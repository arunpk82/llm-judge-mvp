"""
L2 Patterns — Multi-Pass Knowledge Graph Ensemble.

New module for the hallucination pipeline. Lives alongside hallucination.py.
Called from check_hallucination() when L2 patterns layer is enabled.

Replaces old L3 GraphRAG (_l3_graphrag_check) with:
  - 5 graph builders (G1 entities, G2 events, G3 relationships, G4 numbers, G5 negations)
  - spaCy SVO parsing with pronoun resolution, passive voice, verb synonyms
  - Per-graph traversal (G2, G3, G4, G5)
  - Ensemble aggregation with flag-wins rule

Science Gate results (Exp 31, 50 RAGTruth cases):
  - 61 cleared at 100% precision, 0 safety violations
  - 11/16 hallucinations caught (69% recall)
  - Combined with L1: 78 cleared, 11/16 caught

Layer renaming context:
  Old L3 GraphRAG → New L2 Patterns (this module)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# GRAPH BUILDERS
# ============================================================


def _safe_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        return [val]
    return []


def build_g1_entities(p1: dict) -> dict:
    """Graph from P1: entity nodes with attribute edges."""
    import networkx as nx

    G = nx.DiGraph()
    for ent in _safe_list(p1.get("entities")):
        name = (ent.get("name") or "").lower().strip()
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
        for attr_k, attr_v in (ent.get("attributes") or {}).items():
            if attr_v and isinstance(attr_v, str):
                val_node = f"attr_{name}_{attr_k}"
                G.add_node(
                    val_node, node_type="attribute", key=attr_k, value=attr_v.lower()
                )
                G.add_edge(name, val_node, relation=attr_k, polarity="affirm")
    return G


def build_g2_events(p2: dict, p1: dict | None = None) -> dict:
    """Graph from P2: entity → action → target paths. Cross-references P1 for actor resolution."""
    import networkx as nx

    G = nx.DiGraph()

    _entity_names: dict[str, str] = {}
    if p1:
        for ent in _safe_list(p1.get("entities")):
            name = (ent.get("name") or "").lower().strip()
            if name:
                _entity_names[name] = name
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
        if best_score >= 1 and best is not None:
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
            resolved = _resolve_actor(actor_lower) if _entity_names else actor_lower
            if resolved not in G:
                G.add_node(resolved, node_type="entity")
            action_node = (
                f"evt_{resolved}_{action}_{hash(json.dumps(evt, default=str)) % 9999}"
            )
            G.add_node(action_node, node_type="action", verb=action)
            G.add_edge(
                resolved, action_node, relation="action", polarity="affirm", verb=action
            )
            for tgt in targets:
                if isinstance(tgt, str) and tgt.strip():
                    tl = tgt.lower().strip()
                    if tl not in G:
                        G.add_node(tl, node_type="value")
                    G.add_edge(action_node, tl, relation="target", polarity="affirm")
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


def build_g3_relationships(p3: dict) -> dict:
    """Graph from P3: entity → relationship → entity with NOT edges."""
    import networkx as nx

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


def build_g4_numbers(p4: dict) -> dict:
    """Graph from P4: entity → number nodes with context."""
    import networkx as nx

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
        if not (entity and date):
            continue
        if entity not in G:
            G.add_node(entity, node_type="entity")
        date_node = f"date_{entity}_{date}"
        G.add_node(date_node, node_type="date", value=date)
        G.add_edge(
            entity, date_node, relation="has_date", polarity="affirm", value=date
        )
    return G


def build_g5_negations(p5: dict) -> dict:
    """Graph from P5: negation statements, absent info, corrections."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_node("_negation_root", node_type="root")
    for neg in _safe_list(p5.get("explicit_negations")):
        stmt = (neg.get("statement") or str(neg) or "").lower().strip()
        if stmt:
            G.add_node(f"neg_{hash(stmt)%99999}", node_type="negation", statement=stmt)
            G.add_edge(
                "_negation_root",
                f"neg_{hash(stmt)%99999}",
                relation="negation",
                polarity="negate",
            )
    for absent in _safe_list(p5.get("absent_information")):
        what = (absent.get("what") or str(absent) or "").lower().strip()
        if what:
            G.add_node(f"absent_{hash(what)%99999}", node_type="absent", what=what)
            G.add_edge(
                "_negation_root",
                f"absent_{hash(what)%99999}",
                relation="absent",
                polarity="negate",
            )
    for corr in _safe_list(p5.get("corrections")):
        wrong = (corr.get("wrong") or "").lower().strip()
        right = (corr.get("right") or "").lower().strip()
        if wrong and right:
            G.add_node(
                f"corr_{hash(wrong)%99999}",
                node_type="correction",
                wrong=wrong,
                right=right,
            )
            G.add_edge(
                "_negation_root",
                f"corr_{hash(wrong)%99999}",
                relation="correction",
                polarity="negate",
            )
    return G


def build_all_graphs(fact_tables: dict) -> dict:
    """Build all 5 graphs from multi-pass fact tables."""
    passes = fact_tables.get("passes", fact_tables)
    p1 = passes.get("P1_entities")
    p2 = passes.get("P2_events")
    p3 = passes.get("P3_relationships")
    p4 = passes.get("P4_numbers")
    p5 = passes.get("P5_negations")
    graphs = {}
    if p1:
        graphs["G1"] = build_g1_entities(p1)
    if p2:
        graphs["G2"] = build_g2_events(p2, p1)
    if p3:
        graphs["G3"] = build_g3_relationships(p3)
    if p4:
        graphs["G4"] = build_g4_numbers(p4)
    if p5:
        graphs["G5"] = build_g5_negations(p5)
    return graphs


# ============================================================
# SENTENCE PARSING
# ============================================================

_SKIP_VERBS = frozenset(
    {
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
)
_PRONOUNS = frozenset(
    {
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
)
_AUX = frozenset(
    {
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
)
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
_REL_WORDS = frozenset(
    {
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
)


def _parse_sentence_for_l2(sentence: str, nlp) -> dict:
    """Parse sentence with spaCy for L2 traversal."""
    doc = nlp(sentence)
    result: dict[str, list] = {
        "svo_triples": [],
        "entities": [],
        "cardinals": [],
        "relationships": [],
    }

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


# ============================================================
# ENTITY RESOLUTION + VERB MATCHING
# ============================================================


def _follow_alias(G, node):
    if node in G and G.nodes[node].get("node_type") == "alias":
        for _, target, edata in G.edges(node, data=True):
            if edata.get("relation") == "alias_of":
                return target
    return node


def _resolve_entity(G, name):
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


def _norm_verb(v):
    words = v.lower().split()
    core = [w for w in words if w not in _AUX]
    return core[0] if core else v.lower()


def _verb_match(gv, rv, rvt):
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


# ============================================================
# PER-GRAPH TRAVERSAL
# ============================================================


def _traverse_g2(G, parsed):
    verdicts = []
    for triple in parsed["svo_triples"]:
        subj, verb, verb_text = triple["subject"], triple["verb"], triple["verb_text"]
        is_pronoun = bool(set(subj.split()) & _PRONOUNS) and len(subj.split()) <= 3
        entity_node = _resolve_entity(G, subj)
        if not entity_node and is_pronoun:
            for se in parsed["entities"]:
                if se["label"] in ("PERSON", "ORG"):
                    c = _resolve_entity(G, se["lower"])
                    if c:
                        for _, t, d in G.edges(c, data=True):
                            if d.get("relation") == "action" and _verb_match(
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
        if is_passive:
            passive_found = False
            for src_node, tgt_node, data in G.edges(data=True):
                if (
                    data.get("relation") in ("target", "where")
                    and tgt_node == entity_node
                ):
                    action_data = G.nodes.get(src_node, {})
                    if action_data.get("node_type") == "action":
                        if _verb_match(action_data.get("verb", ""), verb, verb_text):
                            verdicts.append(
                                (
                                    "grounded",
                                    f"Passive: '{entity_node}' target of '{verb}'",
                                )
                            )
                            passive_found = True
                            break
            if not passive_found:
                for node, ndata in G.nodes(data=True):
                    if ndata.get("node_type") == "action" and _verb_match(
                        ndata.get("verb", ""), verb, verb_text
                    ):
                        for _, tgt, td in G.edges(node, data=True):
                            if td.get("relation") in ("target", "where") and (
                                entity_node in tgt or tgt in entity_node
                            ):
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
            ev, nv = edge_data.get("verb", ""), G.nodes[action_node].get("verb", "")
            cv = ev or nv
            if _verb_match(cv, verb, verb_text):
                if edge_data.get("polarity") == "negate":
                    verdicts.append(("flagged", f"'{entity_node}'→'{verb}' NEGATED"))
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
    for ent in parsed["entities"]:
        if ent["label"] in ("PERSON", "ORG", "GPE", "NORP"):
            r = _resolve_entity(G, ent["lower"])
            if not r:
                verdicts.append(("flagged", f"Entity '{ent['lower']}' NOT in graph"))
            else:
                verdicts.append(("present", f"Entity '{ent['lower']}' found"))
    return verdicts


def _traverse_g3(G, parsed):
    verdicts = []
    for rel in parsed["relationships"]:
        rw, ctx = rel["rel_word"], rel["context_entity"]
        entity_node = _resolve_entity(G, ctx)
        if not entity_node:
            verdicts.append(("unknown", f"Context '{ctx}' not found"))
            continue
        found = False
        for _, tgt, data in G.edges(entity_node, data=True):
            if rw in data.get("relation", "") or data.get("relation", "") in rw:
                if data.get("polarity") == "negate":
                    verdicts.append(
                        ("flagged", f"Rel '{rw}' for '{entity_node}' NEGATED")
                    )
                else:
                    verdicts.append(
                        ("grounded", f"Rel '{rw}' confirmed for '{entity_node}'")
                    )
                found = True
                break
        if not found:
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


def _traverse_g4(G, parsed):
    verdicts = []
    for card in parsed["cardinals"]:
        num, noun = card["number"], card.get("governing_noun")
        if not noun:
            continue
        for node, data in G.nodes(data=True):
            if data.get("node_type") == "number":
                desc, val = data.get("describes", ""), data.get("value", "")
                if noun in desc or noun in node:
                    if val == num or val == num.replace(",", ""):
                        verdicts.append(("grounded", f"Count {num} confirmed"))
                    else:
                        verdicts.append(
                            ("flagged", f"Count: response {num}, graph {val}")
                        )
                    break
    return verdicts


def _traverse_g5(G, parsed):
    verdicts = []
    sent_text = " ".join(
        t["subject"] + " " + t["verb"] for t in parsed["svo_triples"]
    ).lower()
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "negation":
            stmt = data.get("statement", "")
            stmt_words = {w for w in re.findall(r"\w+", stmt) if len(w) > 3}
            sent_words = {w for w in re.findall(r"\w+", sent_text) if len(w) > 3}
            if len(stmt_words & sent_words) >= 3:
                verdicts.append(("flagged", f"Negation contradiction: {stmt[:60]}"))
        elif data.get("node_type") == "correction":
            wrong = data.get("wrong", "")
            if wrong and wrong in sent_text:
                verdicts.append(
                    (
                        "flagged",
                        f"Response uses '{wrong}' but source says '{data.get('right','')}'",
                    )
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


# ============================================================
# ENSEMBLE AGGREGATION
# ============================================================


def _aggregate_graph_verdicts(g_verdicts: dict) -> dict:
    """Aggregate verdicts from multiple graphs. Flag always wins over grounded."""
    all_evidence = []
    per_graph = {}

    for gname, verdicts in g_verdicts.items():
        graph_vs = [v for v, _ in verdicts]
        graph_ev = [e for _, e in verdicts]
        all_evidence.extend(graph_ev)
        if "flagged" in graph_vs:
            per_graph[gname] = "flagged"
        elif "grounded" in graph_vs:
            per_graph[gname] = "grounded"
        else:
            per_graph[gname] = "unknown"

    n_flagged = sum(1 for v in per_graph.values() if v == "flagged")
    n_grounded = sum(1 for v in per_graph.values() if v == "grounded")

    # Flag always wins (specific check beats general)
    if n_flagged >= 2:
        final, confidence = "flagged", "high"
    elif n_flagged >= 1:
        final, confidence = "flagged", "medium"
    elif n_grounded >= 3:
        final, confidence = "grounded", "high"
    elif n_grounded >= 2:
        final, confidence = "grounded", "medium"
    elif n_grounded >= 1:
        final, confidence = "grounded", "low"
    else:
        final, confidence = "unknown", "none"

    return {
        "verdict": final,
        "confidence": confidence,
        "per_graph": per_graph,
        "n_flagged": n_flagged,
        "n_grounded": n_grounded,
        "evidence": all_evidence[:8],
    }


# ============================================================
# PUBLIC API — called from hallucination.py
# ============================================================


def l2_ensemble_check(
    sentence: str,
    graphs: dict,
    nlp=None,
) -> dict:
    """L2 Patterns: Ensemble graph traversal for a single sentence.

    This replaces the old _l3_graphrag_check() in hallucination.py.

    Args:
        sentence: Response sentence to verify.
        graphs: Dict of NetworkX DiGraphs (G2, G3, G4, G5) from build_all_graphs().
        nlp: spaCy model instance (reuses _spacy_nlp from hallucination.py).

    Returns:
        {
            "verdict": "grounded" | "flagged" | "unknown",
            "confidence": "high" | "medium" | "low" | "none",
            "per_graph": {"G2": "grounded", "G3": "unknown", ...},
            "evidence": [str],
        }
    """
    if nlp is None:
        # Use the spaCy model from hallucination.py
        from llm_judge.calibration.hallucination import _load_spacy

        _load_spacy()
        from llm_judge.calibration.hallucination import _spacy_nlp as nlp_ref

        nlp = nlp_ref

    parsed = _parse_sentence_for_l2(sentence, nlp)

    g_verdicts = {}
    if "G2" in graphs:
        g_verdicts["G2"] = _traverse_g2(graphs["G2"], parsed)
    if "G3" in graphs:
        g_verdicts["G3"] = _traverse_g3(graphs["G3"], parsed)
    if "G4" in graphs:
        g_verdicts["G4"] = _traverse_g4(graphs["G4"], parsed)
    if "G5" in graphs:
        g_verdicts["G5"] = _traverse_g5(graphs["G5"], parsed)

    if not g_verdicts:
        return {
            "verdict": "unknown",
            "confidence": "none",
            "per_graph": {},
            "evidence": ["L2: No graphs available"],
        }

    return _aggregate_graph_verdicts(g_verdicts)
