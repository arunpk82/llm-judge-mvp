"""
Experiment 31 — Stage 2: Ensemble Graph Traversal

Loads 5-pass extractions from Stage 1.
Builds 5 independent graphs per case.
Traverses each graph with spaCy-parsed response sentences.
Aggregates 5 verdicts into confidence score.

No Gemini calls. spaCy + NetworkX only.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _split_sentences

nlp = spacy.load("en_core_web_sm")


# ========================================================================
# GRAPH BUILDERS — one per pass type
# ========================================================================


def _safe_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val]
    return []


def build_g1_entities(p1: dict) -> nx.DiGraph:
    """Graph from P1: entity nodes with attribute edges."""
    G = nx.DiGraph()
    for ent in _safe_list(p1.get("entities")):
        name = ent.get("name", "").lower().strip()
        if not name:
            continue
        G.add_node(
            name,
            node_type="entity",
            entity_type=ent.get("type", ""),
            attributes=ent.get("attributes") or {},
        )
        for alias in _safe_list(ent.get("aliases")):
            al = alias.lower().strip()
            if al and al != name:
                G.add_node(al, node_type="alias")
                G.add_edge(al, name, relation="alias_of", polarity="affirm")
        # Add attributes as edges
        for attr_k, attr_v in (ent.get("attributes") or {}).items():
            if attr_v and isinstance(attr_v, str):
                val_node = f"attr_{name}_{attr_k}"
                G.add_node(
                    val_node, node_type="attribute", key=attr_k, value=attr_v.lower()
                )
                G.add_edge(name, val_node, relation=attr_k, polarity="affirm")
    return G


def build_g2_events(p2: dict, p1: dict | None = None) -> nx.DiGraph:
    """Graph from P2: entity → action → target paths.
    Uses P1 entities for actor resolution and alias edges."""
    G = nx.DiGraph()

    # Build entity lookup from P1 for actor resolution
    _entity_names = {}
    if p1:
        for ent in _safe_list(p1.get("entities")):
            name = (ent.get("name") or "").lower().strip()
            if name:
                _entity_names[name] = name
                # Add as node with aliases
                G.add_node(name, node_type="entity", entity_type=ent.get("type", ""))
                for alias in _safe_list(ent.get("aliases")):
                    al = alias.lower().strip()
                    if al and al != name:
                        _entity_names[al] = name
                        G.add_node(al, node_type="alias")
                        G.add_edge(al, name, relation="alias_of", polarity="affirm")

    def _resolve_actor(actor_str):
        al = actor_str.lower().strip()
        if al in _entity_names:
            return _entity_names[al]
        for key, main in _entity_names.items():
            if key in al or al in key:
                return main
        al_words = {w for w in al.split() if len(w) > 2}
        best, best_score = None, 0
        for key, main in _entity_names.items():
            key_words = {w for w in key.split() if len(w) > 2}
            overlap = len(al_words & key_words)
            if overlap > best_score:
                best_score = overlap
                best = main
        if best_score >= 1:
            return best
        return al

    for evt in _safe_list(p2.get("events")):
        who = evt.get("who", "")
        action = (evt.get("action") or "").lower().strip()
        if not action:
            continue

        actors = _safe_list(who) if isinstance(who, list) else ([who] if who else [])
        targets = _safe_list(evt.get("target"))

        for actor in actors:
            actor_lower = actor.lower().strip()
            if not actor_lower:
                continue

            # Resolve to P1 entity name
            resolved = _resolve_actor(actor_lower) if _entity_names else actor_lower
            if resolved not in G:
                G.add_node(resolved, node_type="entity")

            action_node = f"evt_{resolved}_{action}_{hash(json.dumps(evt))%9999}"
            G.add_node(action_node, node_type="action", verb=action)
            G.add_edge(
                resolved, action_node, relation="action", polarity="affirm", verb=action
            )

            for tgt in targets:
                if isinstance(tgt, str) and tgt.strip():
                    tgt_lower = tgt.lower().strip()
                    if tgt_lower not in G:
                        G.add_node(tgt_lower, node_type="value")
                    G.add_edge(
                        action_node, tgt_lower, relation="target", polarity="affirm"
                    )

            where = evt.get("where") or ""
            if isinstance(where, str) and where.strip():
                wl = where.lower().strip()
                if wl not in G:
                    G.add_node(wl, node_type="value")
                G.add_edge(action_node, wl, relation="where", polarity="affirm")

            when = evt.get("when") or ""
            if isinstance(when, str) and when.strip():
                wnl = when.lower().strip()
                if wnl not in G:
                    G.add_node(wnl, node_type="value")
                G.add_edge(action_node, wnl, relation="when", polarity="affirm")
    return G


def build_g3_relationships(p3: dict) -> nx.DiGraph:
    """Graph from P3: entity → relationship → entity with NOT edges."""
    G = nx.DiGraph()
    for rel in _safe_list(p3.get("relationships")):
        e1 = (rel.get("entity1") or "").lower().strip()
        e2 = (rel.get("entity2") or "").lower().strip()
        rel_type = (rel.get("relationship") or rel.get("type") or "").lower().strip()
        if not (e1 and e2 and rel_type):
            continue

        if e1 not in G:
            G.add_node(e1, node_type="entity")
        if e2 not in G:
            G.add_node(e2, node_type="entity")
        G.add_edge(e1, e2, relation=rel_type, polarity="affirm")

        for not_rel in _safe_list(rel.get("NOT_relationships") or rel.get("NOT")):
            if isinstance(not_rel, str) and not_rel.strip():
                G.add_edge(e1, e2, relation=not_rel.lower().strip(), polarity="negate")
    return G


def build_g4_numbers(p4: dict) -> nx.DiGraph:
    """Graph from P4: entity → number nodes with context."""
    G = nx.DiGraph()
    for nf in _safe_list(p4.get("numerical_facts")):
        entity = (nf.get("entity") or "").lower().strip()
        number = str(nf.get("number") or "").strip()
        describes = (nf.get("describes") or "").lower().strip()
        unit = (nf.get("unit") or "").lower().strip()
        if not (entity and number):
            continue

        if entity not in G:
            G.add_node(entity, node_type="entity")
        num_node = f"num_{entity}_{describes or number}"
        G.add_node(
            num_node, node_type="number", value=number, describes=describes, unit=unit
        )
        G.add_edge(
            entity,
            num_node,
            relation="has_number",
            polarity="affirm",
            value=number,
            describes=describes,
        )

    for tf in _safe_list(p4.get("temporal_facts")):
        entity = (tf.get("entity") or "").lower().strip()
        date = str(tf.get("date") or tf.get("time") or "").lower().strip()
        event_desc = (tf.get("event") or "").lower().strip()
        if not (entity and date):
            continue

        if entity not in G:
            G.add_node(entity, node_type="entity")
        date_node = f"date_{entity}_{date}"
        G.add_node(date_node, node_type="date", value=date, event=event_desc)
        G.add_edge(
            entity, date_node, relation="has_date", polarity="affirm", value=date
        )
    return G


def build_g5_negations(p5: dict) -> nx.DiGraph:
    """Graph from P5: negation nodes, absent info, boundaries, corrections."""
    G = nx.DiGraph()
    G.add_node("_negation_root", node_type="root")

    for neg in _safe_list(p5.get("explicit_negations")):
        stmt = (neg.get("statement") or str(neg) or "").lower().strip()
        if stmt:
            neg_node = f"neg_{hash(stmt)%99999}"
            G.add_node(neg_node, node_type="negation", statement=stmt)
            G.add_edge(
                "_negation_root", neg_node, relation="negation", polarity="negate"
            )

    for absent in _safe_list(p5.get("absent_information")):
        what = (absent.get("what") or str(absent) or "").lower().strip()
        if what:
            abs_node = f"absent_{hash(what)%99999}"
            G.add_node(abs_node, node_type="absent", what=what)
            G.add_edge("_negation_root", abs_node, relation="absent", polarity="negate")

    for corr in _safe_list(p5.get("corrections")):
        wrong = (corr.get("wrong") or "").lower().strip()
        right = (corr.get("right") or "").lower().strip()
        if wrong and right:
            corr_node = f"corr_{hash(wrong)%99999}"
            G.add_node(corr_node, node_type="correction", wrong=wrong, right=right)
            G.add_edge(
                "_negation_root", corr_node, relation="correction", polarity="negate"
            )
    return G


# ========================================================================
# ENTITY RESOLUTION (shared across all graphs)
# ========================================================================


def _follow_alias(G, node):
    if node in G and G.nodes[node].get("node_type") == "alias":
        for _, target, edata in G.edges(node, data=True):
            if edata.get("relation") == "alias_of":
                return target
    return node


def resolve_entity(G, name):
    name_lower = name.lower().strip()
    if name_lower in G:
        return _follow_alias(G, name_lower)
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "alias":
            for _, target, edata in G.edges(node, data=True):
                if edata.get("relation") == "alias_of":
                    if name_lower in node or node in name_lower:
                        return target
    for node in G.nodes():
        if name_lower in node or node in name_lower:
            nd = G.nodes[node]
            if nd.get("node_type") in ("entity", "alias"):
                return _follow_alias(G, node)
    name_parts = [p for p in name_lower.split() if len(p) > 2]
    for node in G.nodes():
        nd = G.nodes[node]
        if nd.get("node_type") in ("entity", "alias"):
            if any(part in node for part in name_parts):
                return _follow_alias(G, node)
    return None


# ========================================================================
# VERB MATCHING
# ========================================================================

_AUX = {
    "was",
    "were",
    "is",
    "are",
    "has",
    "had",
    "have",
    "been",
    "being",
    "will",
    "would",
    "could",
    "should",
    "can",
    "may",
    "might",
    "do",
    "does",
    "did",
}
_SYNONYMS = {
    "assault": "attack",
    "attack": "assault",
    "announce": "declare",
    "declare": "announce",
    "respond": "react",
    "react": "respond",
    "purchase": "buy",
    "buy": "purchase",
    "stab": "wound",
    "wound": "stab",
    "shut": "close",
    "close": "shut",
    "settle": "resolve",
    "resolve": "settle",
    "file": "submit",
    "submit": "file",
    "issue": "release",
    "release": "issue",
    "recall": "withdraw",
    "withdraw": "recall",
    "warn": "advise",
    "advise": "warn",
    "cite": "reference",
    "reference": "cite",
    "arrest": "detain",
    "detain": "arrest",
    "charge": "indict",
    "indict": "charge",
    "die": "pass",
    "kill": "murder",
    "murder": "kill",
}


def _norm_verb(v):
    words = v.lower().split()
    core = [w for w in words if w not in _AUX]
    return core[0] if core else v.lower()


def verb_match(gv, rv, rvt):
    gv, rv, rvt = gv.lower().strip(), rv.lower().strip(), rvt.lower().strip()
    if gv == rv or gv == rvt:
        return True
    gc = _norm_verb(gv)
    if gc == rv or gc == rvt:
        return True
    if rv in gc or gc in rv or rvt in gc or gc in rvt:
        return True
    if gc.rstrip("ed") == rv or rv.rstrip("ed") == gc:
        return True
    if gc.rstrip("ing") == rv or rv.rstrip("ing") == gc:
        return True
    if gc.rstrip("s") == rv or rv.rstrip("s") == gc:
        return True
    rs = _SYNONYMS.get(rv, "")
    if rs and (rs in gc or gc in rs):
        return True
    return False


# ========================================================================
# PER-GRAPH TRAVERSAL
# ========================================================================

_SKIP_VERBS = {
    "be",
    "have",
    "do",
    "go",
    "get",
    "make",
    "take",
    "come",
    "give",
    "say",
    "know",
    "think",
    "see",
    "seem",
    "become",
    "keep",
    "let",
    "begin",
    "show",
    "try",
    "leave",
    "call",
}
_PRONOUNS = {
    "he",
    "she",
    "they",
    "it",
    "who",
    "whom",
    "which",
    "that",
    "this",
    "these",
    "those",
    "them",
    "his",
    "her",
    "its",
    "their",
    "we",
    "one",
    "someone",
    "everybody",
}


def parse_sentence(sentence):
    doc = nlp(sentence)
    result = {"svo_triples": [], "entities": [], "cardinals": [], "relationships": []}

    for ent in doc.ents:
        result["entities"].append(
            {"text": ent.text, "label": ent.label_, "lower": ent.text.lower()}
        )

    for token in doc:
        if token.dep_ in ("nsubj", "nsubjpass"):
            verb = token.head
            vl = verb.lemma_.lower()
            if vl in _SKIP_VERBS:
                continue

            objects = []
            for child in verb.children:
                if child.dep_ in ("dobj", "attr", "oprd", "acomp"):
                    objects.append(
                        {
                            "text": " ".join(t.text for t in child.subtree).lower(),
                            "head": child.text.lower(),
                        }
                    )
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(
                                {
                                    "text": " ".join(
                                        t.text for t in pobj.subtree
                                    ).lower(),
                                    "head": pobj.text.lower(),
                                    "prep": child.text.lower(),
                                }
                            )
            for child in verb.children:
                if child.dep_ == "xcomp":
                    objects.append(
                        {
                            "text": " ".join(t.text for t in child.subtree).lower(),
                            "head": child.text.lower(),
                        }
                    )

            has_neg = any(c.dep_ == "neg" for c in verb.children)
            subj_parts = [
                t.text
                for t in token.subtree
                if t.dep_ in ("compound", "nsubj", "nsubjpass", "amod")
            ]
            subj_text = " ".join(subj_parts) if subj_parts else token.text

            result["svo_triples"].append(
                {
                    "subject": subj_text.lower(),
                    "verb": vl,
                    "verb_text": verb.text.lower(),
                    "objects": objects,
                    "negated": has_neg,
                    "passive": token.dep_ == "nsubjpass",
                }
            )

    _REL_WORDS = {
        "wife",
        "husband",
        "companion",
        "girlfriend",
        "boyfriend",
        "daughter",
        "son",
        "mother",
        "father",
        "sister",
        "brother",
        "teammate",
        "partner",
        "friend",
        "fiancee",
    }
    for token in doc:
        if token.text.lower() in _REL_WORDS:
            head = token.head
            while head.dep_ not in ("ROOT", "nsubj", "nsubjpass") and head.head != head:
                head = head.head
            result["relationships"].append(
                {"rel_word": token.text.lower(), "context_entity": head.text.lower()}
            )

    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            governing = None
            for t in ent.root.head.subtree:
                if t.pos_ == "NOUN" and t != ent.root:
                    governing = t.text.lower()
                    break
            if not governing and ent.root.head.pos_ == "NOUN":
                governing = ent.root.head.text.lower()
            result["cardinals"].append(
                {"number": ent.text, "governing_noun": governing}
            )

    return result


def traverse_g2(G, parsed):
    """Traverse events graph (G2) — most important for SVO verification."""
    verdicts = []
    for triple in parsed["svo_triples"]:
        subj = triple["subject"]
        verb, verb_text = triple["verb"], triple["verb_text"]
        is_pronoun = bool(set(subj.split()) & _PRONOUNS) and len(subj.split()) <= 3

        entity_node = resolve_entity(G, subj)

        # Pronoun resolution
        if not entity_node and is_pronoun:
            for se in parsed["entities"]:
                if se["label"] in ("PERSON", "ORG"):
                    c = resolve_entity(G, se["lower"])
                    if c:
                        for _, t, d in G.edges(c, data=True):
                            if d.get("relation") == "action" and verb_match(
                                d.get("verb", ""), verb, verb_text
                            ):
                                entity_node = c
                                break
                    if entity_node:
                        break
            if not entity_node:
                best, best_n = None, 0
                for n, nd in G.nodes(data=True):
                    if nd.get("node_type") == "entity":
                        cnt = sum(
                            1
                            for _, _, d in G.edges(n, data=True)
                            if d.get("relation") == "action"
                        )
                        if cnt > best_n:
                            best_n = cnt
                            best = n
                if best:
                    entity_node = best

        if not entity_node:
            verdicts.append(("unknown", f"Subject '{subj}' not found"))
            continue

        is_passive = triple.get("passive", False)

        # Passive voice: search for entity as TARGET, not actor
        if is_passive:
            passive_found = False
            for src_node, tgt_node, data in G.edges(data=True):
                if (
                    data.get("relation") in ("target", "where")
                    and tgt_node == entity_node
                ):
                    action_data = G.nodes.get(src_node, {})
                    if action_data.get("node_type") == "action":
                        av = action_data.get("verb", "")
                        if verb_match(av, verb, verb_text):
                            verdicts.append(
                                (
                                    "grounded",
                                    f"Passive: '{entity_node}' is target of '{verb}'",
                                )
                            )
                            passive_found = True
                            break

            # Also check by substring match on targets
            if not passive_found:
                for node, ndata in G.nodes(data=True):
                    if ndata.get("node_type") == "action":
                        av = ndata.get("verb", "")
                        if verb_match(av, verb, verb_text):
                            for _, tgt, td in G.edges(node, data=True):
                                if td.get("relation") in ("target", "where"):
                                    if entity_node in tgt or tgt in entity_node:
                                        verdicts.append(
                                            (
                                                "grounded",
                                                f"Passive: '{entity_node}' target of '{verb}'",
                                            )
                                        )
                                        passive_found = True
                                        break
                        if passive_found:
                            break

            if passive_found:
                continue
            # Fall through to active check

        # Find action edges
        action_edges = [
            (t, d)
            for _, t, d in G.edges(entity_node, data=True)
            if d.get("relation") == "action"
        ]

        if not action_edges:
            verdicts.append(
                (
                    "flagged",
                    f"Entity '{entity_node}' has NO actions but response claims '{verb}'",
                )
            )
            continue

        matched = False
        for action_node, edge_data in action_edges:
            ev = edge_data.get("verb", "")
            nv = G.nodes[action_node].get("verb", "")
            cv = ev or nv
            if verb_match(cv, verb, verb_text):
                if edge_data.get("polarity") == "negate":
                    verdicts.append(
                        ("flagged", f"'{entity_node}' → '{verb}' is NEGATED")
                    )
                    matched = True
                    break

                if triple["objects"]:
                    obj_ok = False
                    for obj in triple["objects"]:
                        oh = obj["head"]
                        for _, tgt, td in G.edges(action_node, data=True):
                            if td.get("relation") in ("target", "where"):
                                if oh in tgt or tgt in oh:
                                    if td.get("polarity") == "negate":
                                        verdicts.append(
                                            (
                                                "flagged",
                                                f"'{entity_node}'→'{verb}'→'{oh}' NEGATED",
                                            )
                                        )
                                        obj_ok = True
                                        break
                                    else:
                                        obj_ok = True
                                        verdicts.append(
                                            (
                                                "grounded",
                                                f"Path: '{entity_node}'→'{verb}'→'{tgt}'",
                                            )
                                        )
                                        break
                        if obj_ok:
                            break

                    if not obj_ok:
                        graph_tgts = [
                            t
                            for _, t, d in G.edges(action_node, data=True)
                            if d.get("relation") in ("target", "where")
                            and d.get("polarity") == "affirm"
                        ]
                        if graph_tgts:
                            verdicts.append(
                                (
                                    "flagged",
                                    f"'{entity_node}'→'{verb}' targets {graph_tgts[:2]} not {[o['head'] for o in triple['objects']]}",
                                )
                            )
                        else:
                            verdicts.append(
                                (
                                    "flagged",
                                    f"'{entity_node}'→'{verb}' has NO targets but response adds objects",
                                )
                            )
                else:
                    verdicts.append(
                        ("grounded", f"Action '{verb}' confirmed for '{entity_node}'")
                    )
                matched = True
                break

        if not matched:
            verdicts.append(
                ("unknown", f"Action '{verb}' not found for '{entity_node}'")
            )

    # Entity checks
    for ent in parsed["entities"]:
        if ent["label"] in ("PERSON", "ORG", "GPE", "NORP"):
            r = resolve_entity(G, ent["lower"])
            if not r:
                verdicts.append(("flagged", f"Entity '{ent['lower']}' NOT in graph"))
            else:
                verdicts.append(("present", f"Entity '{ent['lower']}' found"))

    return verdicts


def traverse_g3(G, parsed):
    """Traverse relationships graph (G3)."""
    verdicts = []
    for rel in parsed["relationships"]:
        rw = rel["rel_word"]
        ctx = rel["context_entity"]
        entity_node = resolve_entity(G, ctx)
        if not entity_node:
            verdicts.append(("unknown", f"Context '{ctx}' not found for rel '{rw}'"))
            continue
        found = False
        for _, tgt, data in G.edges(entity_node, data=True):
            if rw in data.get("relation", "") or data.get("relation", "") in rw:
                if data.get("polarity") == "negate":
                    verdicts.append(
                        ("flagged", f"Relationship '{rw}' for '{entity_node}' NEGATED")
                    )
                else:
                    verdicts.append(
                        (
                            "grounded",
                            f"Relationship '{rw}' confirmed for '{entity_node}'",
                        )
                    )
                found = True
                break
        if not found:
            # Check if different relationship exists
            existing = [
                d.get("relation")
                for _, _, d in G.edges(entity_node, data=True)
                if d.get("polarity") == "affirm"
            ]
            if existing:
                verdicts.append(
                    (
                        "flagged",
                        f"'{rw}' not found for '{entity_node}'; has {existing[:3]}",
                    )
                )
            else:
                verdicts.append(("unknown", f"No relationships for '{entity_node}'"))
    return verdicts


def traverse_g4(G, parsed):
    """Traverse numbers graph (G4)."""
    verdicts = []
    for card in parsed["cardinals"]:
        num = card["number"]
        noun = card.get("governing_noun")
        if not noun:
            continue
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "number":
                desc = data.get("describes", "")
                val = data.get("value", "")
                if noun in desc or noun in node:
                    if val == num or val == num.replace(",", ""):
                        verdicts.append(
                            ("grounded", f"Count {num} for '{noun}' confirmed")
                        )
                    else:
                        verdicts.append(
                            (
                                "flagged",
                                f"Count: response {num}, graph {val} for '{noun}'",
                            )
                        )
                    break
    return verdicts


def traverse_g5(G, parsed):
    """Traverse negations graph (G5) — check if response contradicts negations."""
    verdicts = []
    sent_text = " ".join(
        t["subject"] + " " + t["verb"] for t in parsed["svo_triples"]
    ).lower()

    for node, data in G.nodes(data=True):
        if data.get("node_type") == "negation":
            stmt = data.get("statement", "")
            # Check keyword overlap
            stmt_words = {w for w in re.findall(r"\w+", stmt) if len(w) > 3}
            sent_words = {w for w in re.findall(r"\w+", sent_text) if len(w) > 3}
            overlap = stmt_words & sent_words
            if len(overlap) >= 3:
                verdicts.append(
                    ("flagged", f"Possible negation contradiction: {stmt[:60]}")
                )

        elif data.get("node_type") == "correction":
            wrong = data.get("wrong", "")
            right = data.get("right", "")
            if wrong and wrong in sent_text:
                verdicts.append(
                    ("flagged", f"Response uses '{wrong}' but source says '{right}'")
                )

        elif data.get("node_type") == "absent":
            what = data.get("what", "")
            if what:
                what_words = {w for w in re.findall(r"\w+", what) if len(w) > 3}
                sent_words_full = {
                    w for w in re.findall(r"\w+", sent_text) if len(w) > 3
                }
                if len(what_words & sent_words_full) >= 2:
                    verdicts.append(
                        ("flagged", f"Response mentions absent info: {what[:60]}")
                    )

    return verdicts


# ========================================================================
# AGGREGATION
# ========================================================================


def aggregate_verdicts(g_verdicts: dict) -> dict:
    """Aggregate verdicts from 5 graphs into a final confidence score."""
    all_verdicts = []
    all_evidence = []
    per_graph = {}

    for gname, verdicts in g_verdicts.items():
        graph_vs = [v for v, _ in verdicts]
        graph_ev = [e for _, e in verdicts]
        all_verdicts.extend(graph_vs)
        all_evidence.extend(graph_ev)

        if "flagged" in graph_vs:
            per_graph[gname] = "flagged"
        elif "grounded" in graph_vs:
            per_graph[gname] = "grounded"
        else:
            per_graph[gname] = "unknown"

    n_flagged = sum(1 for v in per_graph.values() if v == "flagged")
    n_grounded = sum(1 for v in per_graph.values() if v == "grounded")
    n_unknown = sum(1 for v in per_graph.values() if v == "unknown")
    n_graphs = len(per_graph)

    # Confidence rules — flag always wins (specific check beats general)
    if n_flagged >= 2:
        final = "flagged"
        confidence = "high"
    elif n_flagged == 1:
        final = "flagged"
        confidence = "medium"
    elif n_grounded >= 3 and n_flagged == 0:
        final = "grounded"
        confidence = "high"
    elif n_grounded >= 2 and n_flagged == 0:
        final = "grounded"
        confidence = "medium"
    elif n_grounded == 1 and n_flagged == 0 and n_unknown <= 3:
        final = "grounded"
        confidence = "low"
    else:
        final = "unknown"
        confidence = "none"

    return {
        "verdict": final,
        "confidence": confidence,
        "per_graph": per_graph,
        "n_flagged": n_flagged,
        "n_grounded": n_grounded,
        "n_unknown": n_unknown,
        "evidence": all_evidence[:8],
    }


# ========================================================================
# LABELLING
# ========================================================================


def label_sentences(response, span_annotations, response_level):
    sentences = _split_sentences(response)
    labels = []
    for i, sent in enumerate(sentences):
        sent_start = response.find(sent[:50])
        sent_end = sent_start + len(sent) if sent_start >= 0 else -1
        is_hall = False
        hall_type = ""
        if response_level == "fail" and span_annotations:
            for span in span_annotations:
                if sent_start >= 0 and span.start < sent_end and span.end > sent_start:
                    is_hall = True
                    hall_type = span.label_type
                    break
        labels.append(
            {
                "idx": i,
                "sentence": sent,
                "label": "hallucinated" if is_hall else "clean",
                "type": hall_type,
            }
        )
    return labels


@dataclass
class SentenceResult:
    case_id: str
    sentence_idx: int
    sentence: str
    gt_label: str
    gt_type: str
    verdict: str
    confidence: str
    n_flagged: int
    n_grounded: int
    n_unknown: int
    per_graph: str
    evidence: str


def main():
    ft_path = "experiments/exp31_multipass_fact_tables.json"
    if not os.path.exists(ft_path):
        print(f"ERROR: {ft_path} not found.")
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 31 — STAGE 2: Ensemble Graph Traversal")
    print("=" * 70)
    print("  5 graphs per case. spaCy parse. Path traversal. Confidence aggregation.")
    print("=" * 70)

    with open(ft_path) as f:
        all_ft = json.load(f)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"  Cases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results = []
    verdict_counts = {"grounded": 0, "flagged": 0, "unknown": 0}
    confidence_counts = {"high": 0, "medium": 0, "low": 0, "conflict": 0, "none": 0}
    safety_violations = []
    hallucination_catches = []
    t0 = time.time()

    for ci, case in enumerate(cases):
        response = case.request.candidate_answer or ""
        gt = case.ground_truth
        labeled = label_sentences(response, gt.span_annotations, gt.response_level)

        ft_data = all_ft.get(case.case_id, {})
        passes = ft_data.get("passes", {})

        # Build 5 graphs
        graphs = {}
        p1 = passes.get("P1_entities")
        p2 = passes.get("P2_events")
        p3 = passes.get("P3_relationships")
        p4 = passes.get("P4_numbers")
        p5 = passes.get("P5_negations")

        if p1:
            graphs["G1_entities"] = build_g1_entities(p1)
        if p2:
            graphs["G2_events"] = build_g2_events(p2, p1)  # Cross-ref P1 entities
        if p3:
            graphs["G3_relations"] = build_g3_relationships(p3)
        if p4:
            graphs["G4_numbers"] = build_g4_numbers(p4)
        if p5:
            graphs["G5_negations"] = build_g5_negations(p5)

        for sl in labeled:
            sent = sl["sentence"]
            parsed = parse_sentence(sent)

            # Traverse each graph
            g_verdicts = {}
            if "G2_events" in graphs:
                g_verdicts["G2"] = traverse_g2(graphs["G2_events"], parsed)
            if "G3_relations" in graphs:
                g_verdicts["G3"] = traverse_g3(graphs["G3_relations"], parsed)
            if "G4_numbers" in graphs:
                g_verdicts["G4"] = traverse_g4(graphs["G4_numbers"], parsed)
            if "G5_negations" in graphs:
                g_verdicts["G5"] = traverse_g5(graphs["G5_negations"], parsed)
            # G1 entity check via G2 (entities are checked during G2 traversal)

            agg = aggregate_verdicts(g_verdicts)

            verdict_counts[agg["verdict"]] += 1
            confidence_counts[agg["confidence"]] += 1

            if agg["verdict"] == "grounded" and sl["label"] == "hallucinated":
                safety_violations.append(
                    {
                        "case_id": case.case_id,
                        "idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "confidence": agg["confidence"],
                        "per_graph": agg["per_graph"],
                        "evidence": agg["evidence"][:3],
                    }
                )

            if agg["verdict"] == "flagged" and sl["label"] == "hallucinated":
                hallucination_catches.append(
                    {
                        "case_id": case.case_id,
                        "idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "confidence": agg["confidence"],
                        "per_graph": agg["per_graph"],
                        "evidence": agg["evidence"][:3],
                    }
                )

            all_results.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    verdict=agg["verdict"],
                    confidence=agg["confidence"],
                    n_flagged=agg["n_flagged"],
                    n_grounded=agg["n_grounded"],
                    n_unknown=agg["n_unknown"],
                    per_graph=json.dumps(agg["per_graph"]),
                    evidence="; ".join(agg["evidence"][:5]),
                )
            )

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"\n--- {ci+1}/{len(cases)} | {len(all_results)} sents | {elapsed:.1f}s ---"
            )
            print(
                f"  G={verdict_counts['grounded']} F={verdict_counts['flagged']} U={verdict_counts['unknown']}"
            )
            print(
                f"  Safety: {len(safety_violations)}  Catches: {len(hallucination_catches)}"
            )

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"FINAL — {len(all_results)} sentences, {elapsed:.1f}s")
    print(f"{'='*70}")

    print("\n  VERDICTS")
    print(
        f"    Grounded:  {verdict_counts['grounded']:3d} ({verdict_counts['grounded']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Flagged:   {verdict_counts['flagged']:3d} ({verdict_counts['flagged']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Unknown:   {verdict_counts['unknown']:3d} ({verdict_counts['unknown']/len(all_results)*100:.1f}%)"
    )

    print("\n  CONFIDENCE")
    for conf in ["high", "medium", "low", "conflict", "none"]:
        print(f"    {conf:10s}: {confidence_counts[conf]:3d}")

    print(f"\n{'='*70}")
    print("SAFETY VIOLATIONS")
    print(f"{'='*70}")
    if safety_violations:
        print(f"  *** {len(safety_violations)} SAFETY VIOLATIONS ***")
        for sv in safety_violations:
            print(
                f"    {sv['case_id']} S{sv['idx']} GT={sv['gt_type']} conf={sv['confidence']}"
            )
            print(f"      Sentence: {sv['sentence']}")
            print(f"      Per-graph: {sv['per_graph']}")
    else:
        print("  *** 0 SAFETY VIOLATIONS ***")

    print(f"\n{'='*70}")
    print("HALLUCINATION CATCHES")
    print(f"{'='*70}")
    print(
        f"  {len(hallucination_catches)} / 16 caught ({len(hallucination_catches)/16*100:.0f}% recall)"
    )
    for hc in hallucination_catches:
        print(
            f"    {hc['case_id']} S{hc['idx']} GT={hc['gt_type']} conf={hc['confidence']}"
        )
        print(f"      Sentence: {hc['sentence']}")
        print(f"      Per-graph: {hc['per_graph']}")

    # Missed
    caught_ids = {(hc["case_id"], hc["idx"]) for hc in hallucination_catches}
    print(f"\n  Missed ({16-len(hallucination_catches)}):")
    for r in all_results:
        if (
            r.gt_label == "hallucinated"
            and (r.case_id, r.sentence_idx) not in caught_ids
        ):
            print(
                f"    {r.case_id} S{r.sentence_idx} v={r.verdict} conf={r.confidence} graphs={r.per_graph}"
            )
            print(f"      {r.sentence[:80]}")

    # Verdict vs GT
    print(f"\n{'='*70}")
    print("VERDICT vs GROUND TRUTH")
    print(f"{'='*70}")
    for v in ["grounded", "flagged", "unknown"]:
        vr = [r for r in all_results if r.verdict == v]
        vc = sum(1 for r in vr if r.gt_label == "clean")
        vh = sum(1 for r in vr if r.gt_label == "hallucinated")
        print(f"  {v:12s}: {len(vr):3d} ({vc} clean, {vh} hallucinated)")

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    print("  L1 A1:          cleared 21, caught 4/16, 0 safety")
    print("  L2c single:     cleared 43, caught 6/16, 1 safety")
    print("  L1+L2c:         cleared 61, caught 7/16, 1 safety")
    print(
        f"  L2 ensemble:    cleared {verdict_counts['grounded']}, caught {len(hallucination_catches)}/16, {len(safety_violations)} safety"
    )

    # Save
    output = {
        "elapsed_s": round(elapsed, 1),
        "verdict_counts": verdict_counts,
        "confidence_counts": confidence_counts,
        "safety_violations": len(safety_violations),
        "hallucination_catches": len(hallucination_catches),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp31_stage2_ensemble_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp31_stage2_ensemble_results.json")


if __name__ == "__main__":
    main()
