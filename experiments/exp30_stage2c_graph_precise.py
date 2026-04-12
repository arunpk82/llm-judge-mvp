"""
Experiment 30 — Stage 2b: Graph-Based Path Traversal

Converts JSON fact tables to NetworkX directed graphs.
Uses spaCy to parse response sentences into SVO triples.
Verifies each triple by traversing the graph for matching paths.

Requires: pip install networkx spacy
           python -m spacy download en_core_web_sm

Loads fact tables from Stage 1 (no Gemini calls).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _split_sentences

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# ========================================================================
# PHASE 1b: Convert JSON fact table to NetworkX graph
# ========================================================================


def json_to_graph(fact_table: dict) -> nx.DiGraph:
    """Convert a JSON fact table to a NetworkX directed graph with typed edges."""
    G = nx.DiGraph()

    # --- Add entities as nodes ---
    for ent in fact_table.get("entities", []):
        name = ent.get("name", "").lower().strip()
        if not name:
            continue
        G.add_node(
            name,
            node_type="entity",
            entity_type=ent.get("type", ""),
            attributes=ent.get("attributes", {}),
        )
        # Add aliases as separate nodes pointing to the main entity
        for alias in ent.get("aliases") or []:
            alias_lower = alias.lower().strip()
            if alias_lower and alias_lower != name:
                G.add_node(alias_lower, node_type="alias")
                G.add_edge(alias_lower, name, relation="alias_of", polarity="affirm")

    # Build a lookup of entity names + aliases for resolving event actors
    _entity_names = {}
    for ent in fact_table.get("entities", []):
        name = ent.get("name", "").lower().strip()
        if name:
            _entity_names[name] = name
            for alias in ent.get("aliases") or []:
                if alias:
                    _entity_names[alias.lower().strip()] = name

    def _resolve_actor(actor_str: str) -> str:
        """Resolve an event actor to an existing entity node name."""
        al = actor_str.lower().strip()
        # Exact match
        if al in _entity_names:
            return _entity_names[al]
        # Substring: "two Atlanta Hawks" → "atlanta hawks"
        for key, main in _entity_names.items():
            if key in al or al in key:
                return main
        # Word overlap: find best match by shared words
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
        return al  # fallback to original

    # --- Add events as edges ---
    for evt in fact_table.get("events", []):
        who = evt.get("who", "")
        action = evt.get("action", "").lower().strip()
        if not action:
            continue

        # Normalise "who" — can be string or list
        actors = [who] if isinstance(who, str) else (who or [])
        targets = evt.get("target") or []
        if isinstance(targets, str):
            targets = [targets]

        for actor in actors:
            actor_lower = actor.lower().strip()
            if not actor_lower:
                continue

            # Resolve actor to existing entity node
            resolved_actor = _resolve_actor(actor_lower)

            # Ensure actor node exists
            if resolved_actor not in G:
                G.add_node(resolved_actor, node_type="entity")

            # Action node (unique per event)
            action_node = f"evt_{resolved_actor}_{action}"
            G.add_node(action_node, node_type="action", verb=action)
            G.add_edge(
                resolved_actor,
                action_node,
                relation="action",
                polarity="affirm",
                verb=action,
            )

            # Targets
            for target in targets:
                if isinstance(target, str) and target.strip():
                    tgt_lower = target.lower().strip()
                    if tgt_lower not in G:
                        G.add_node(tgt_lower, node_type="value")
                    G.add_edge(
                        action_node, tgt_lower, relation="target", polarity="affirm"
                    )

            # NOT targets
            for not_tgt in evt.get("NOT_target") or []:
                if isinstance(not_tgt, str) and not_tgt.strip():
                    ntgt_lower = not_tgt.lower().strip()
                    if ntgt_lower not in G:
                        G.add_node(ntgt_lower, node_type="value")
                    G.add_edge(
                        action_node, ntgt_lower, relation="target", polarity="negate"
                    )

            # NOT actions for this actor
            for not_act in evt.get("NOT_action") or []:
                if isinstance(not_act, str) and not_act.strip():
                    nact_lower = not_act.lower().strip()
                    nact_node = f"evt_{resolved_actor}_{nact_lower}"
                    G.add_node(nact_node, node_type="action", verb=nact_lower)
                    G.add_edge(
                        resolved_actor,
                        nact_node,
                        relation="action",
                        polarity="negate",
                        verb=nact_lower,
                    )

            # Where
            where = evt.get("where", "")
            if isinstance(where, str) and where.strip():
                where_lower = where.lower().strip()
                if where_lower not in G:
                    G.add_node(where_lower, node_type="value")
                G.add_edge(
                    action_node, where_lower, relation="where", polarity="affirm"
                )

            # When
            when = evt.get("when", "")
            if isinstance(when, str) and when.strip():
                when_lower = when.lower().strip()
                if when_lower not in G:
                    G.add_node(when_lower, node_type="value")
                G.add_edge(action_node, when_lower, relation="when", polarity="affirm")

    # --- Add relationships as edges ---
    for rel in fact_table.get("relationships", []):
        e1 = rel.get("entity1", "").lower().strip()
        e2 = rel.get("entity2", "").lower().strip()
        rel_type = rel.get("relationship", rel.get("type", "")).lower().strip()

        if e1 and e2 and rel_type:
            # Resolve to main entity nodes
            e1_resolved = _resolve_actor(e1)
            e2_resolved = _resolve_actor(e2)
            if e1_resolved not in G:
                G.add_node(e1_resolved, node_type="entity")
            if e2_resolved not in G:
                G.add_node(e2_resolved, node_type="entity")
            G.add_edge(e1_resolved, e2_resolved, relation=rel_type, polarity="affirm")

        # NOT relationships
        for not_rel in rel.get("NOT") or []:
            if isinstance(not_rel, str) and not_rel.strip():
                G.add_edge(
                    e1_resolved,
                    e2_resolved,
                    relation=not_rel.lower().strip(),
                    polarity="negate",
                )

    # --- Add numerical facts ---
    for nf in fact_table.get("numerical_facts", []):
        entity = nf.get("entity", "").lower().strip()
        number = str(nf.get("number", "")).strip()
        describes = nf.get("describes", "").lower().strip()

        if entity and number:
            entity_resolved = _resolve_actor(entity)
            if entity_resolved not in G:
                G.add_node(entity_resolved, node_type="entity")
            num_node = (
                f"num_{entity_resolved}_{describes}"
                if describes
                else f"num_{entity_resolved}_{number}"
            )
            G.add_node(num_node, node_type="number", value=number, describes=describes)
            G.add_edge(
                entity_resolved,
                num_node,
                relation="has_number",
                polarity="affirm",
                describes=describes,
                value=number,
            )

    # --- Add negations ---
    for neg in fact_table.get("negations", []):
        statement = neg.get("statement", "") if isinstance(neg, dict) else str(neg)
        if statement:
            neg_node = f"negation_{hash(statement) % 10000}"
            G.add_node(neg_node, node_type="negation", statement=statement.lower())
            G.add_edge("_negations", neg_node, relation="negation", polarity="negate")

    return G


# ========================================================================
# PHASE 2a: spaCy response sentence parsing
# ========================================================================


def parse_sentence(sentence: str) -> dict:
    """Parse a response sentence with spaCy to extract structured claims."""
    doc = nlp(sentence)

    result = {
        "svo_triples": [],
        "entities": [],
        "cardinals": [],
        "relationships": [],
    }

    # --- Extract named entities ---
    for ent in doc.ents:
        result["entities"].append(
            {
                "text": ent.text,
                "label": ent.label_,
                "lower": ent.text.lower(),
            }
        )

    # --- Extract SVO triples from dependency parse ---
    for token in doc:
        # Find subjects
        if token.dep_ in ("nsubj", "nsubjpass"):
            verb = token.head
            verb_lemma = verb.lemma_.lower()

            # Skip ultra-generic verbs that don't represent specific events
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
            if verb_lemma in _SKIP_VERBS:
                continue

            is_passive = token.dep_ == "nsubjpass"

            # Find objects
            objects = []
            for child in verb.children:
                if child.dep_ in ("dobj", "attr", "oprd", "acomp"):
                    obj_text = " ".join(t.text for t in child.subtree).lower()
                    objects.append({"text": obj_text, "head": child.text.lower()})

                # Check for prepositional objects
                if child.dep_ == "prep":
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            obj_text = " ".join(t.text for t in pobj.subtree).lower()
                            objects.append(
                                {
                                    "text": obj_text,
                                    "head": pobj.text.lower(),
                                    "prep": child.text.lower(),
                                }
                            )

            # Check for xcomp (e.g., "charged with attempting to join")
            for child in verb.children:
                if child.dep_ == "xcomp":
                    xcomp_text = " ".join(t.text for t in child.subtree).lower()
                    objects.append({"text": xcomp_text, "head": child.text.lower()})

            # Detect negation on the verb
            has_neg = any(c.dep_ == "neg" for c in verb.children)

            # Get subject with compounds
            subj_parts = [
                t.text
                for t in token.subtree
                if t.dep_ in ("compound", "nsubj", "nsubjpass", "amod")
            ]
            subj_text = " ".join(subj_parts) if subj_parts else token.text

            result["svo_triples"].append(
                {
                    "subject": subj_text.lower(),
                    "verb": verb_lemma,
                    "verb_text": verb.text.lower(),
                    "objects": objects,
                    "negated": has_neg,
                    "passive": is_passive,
                }
            )

    # --- Extract cardinals with their governing noun ---
    for ent in doc.ents:
        if ent.label_ == "CARDINAL":
            # Find the noun this cardinal modifies
            governing_noun = None
            for token in ent.root.head.subtree:
                if token.pos_ == "NOUN" and token != ent.root:
                    governing_noun = token.text.lower()
                    break
            if not governing_noun and ent.root.head.pos_ == "NOUN":
                governing_noun = ent.root.head.text.lower()

            result["cardinals"].append(
                {
                    "number": ent.text,
                    "governing_noun": governing_noun,
                }
            )

    # --- Extract relationship words ---
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
            # Find which entity this relationship modifies
            head = token.head
            while head.dep_ not in ("ROOT", "nsubj", "nsubjpass") and head.head != head:
                head = head.head
            result["relationships"].append(
                {
                    "rel_word": token.text.lower(),
                    "context_entity": head.text.lower(),
                }
            )

    return result


# ========================================================================
# PHASE 2b: Graph path traversal verification
# ========================================================================


def _follow_alias(G: nx.DiGraph, node: str) -> str:
    """If node is an alias, follow alias_of edge to main entity."""
    if G.nodes[node].get("node_type") == "alias":
        for _, target, edata in G.edges(node, data=True):
            if edata.get("relation") == "alias_of":
                return target
    return node


def _resolve_entity(G: nx.DiGraph, name: str) -> str | None:
    """Find entity node in graph by exact match, alias, or substring.
    Always follows alias_of edges to return the MAIN entity node."""
    name_lower = name.lower().strip()

    # Exact match
    if name_lower in G:
        return _follow_alias(G, name_lower)

    # Check aliases
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "alias":
            for _, target, edata in G.edges(node, data=True):
                if edata.get("relation") == "alias_of":
                    if name_lower in node or node in name_lower:
                        return target

    # Substring match — return main entity, not alias
    for node in G.nodes():
        if name_lower in node or node in name_lower:
            node_data = G.nodes[node]
            if node_data.get("node_type") in ("entity", "alias"):
                return _follow_alias(G, node)

    # Word-level partial match
    name_parts = [p for p in name_lower.split() if len(p) > 2]
    for node in G.nodes():
        node_data = G.nodes[node]
        if node_data.get("node_type") in ("entity", "alias"):
            if any(part in node for part in name_parts):
                return _follow_alias(G, node)

    return None


def verify_svo(
    G: nx.DiGraph, triple: dict, sentence_entities: list | None = None
) -> dict:
    """Verify a single SVO triple against the graph.
    sentence_entities: NER entities from the same sentence, used for pronoun resolution.
    """
    result = {"check": "svo", "verdict": "unknown", "evidence": ""}

    subj = triple["subject"]
    verb = triple["verb"]
    verb_text = triple["verb_text"]
    negated = triple["negated"]
    is_passive = triple.get("passive", False)

    # --- Helper functions (defined first, used throughout) ---
    _AUX_VERBS = {
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
        "convict": "sentence",
        "sentence": "convict",
        "charge": "indict",
        "indict": "charge",
        "die": "pass",
        "kill": "murder",
        "murder": "kill",
    }

    def _normalize_verb(v):
        words = v.lower().split()
        core = [w for w in words if w not in _AUX_VERBS]
        return core[0] if core else v.lower()

    def _verb_matches(graph_verb, response_verb, response_verb_text):
        gv = graph_verb.lower().strip()
        rv = response_verb.lower().strip()
        rvt = response_verb_text.lower().strip()

        if gv == rv or gv == rvt:
            return True

        gv_core = _normalize_verb(gv)

        if gv_core == rv or gv_core == rvt:
            return True
        if rv in gv_core or gv_core in rv:
            return True
        if rvt in gv_core or gv_core in rvt:
            return True
        if gv_core.rstrip("ed") == rv or rv.rstrip("ed") == gv_core:
            return True
        if gv_core.rstrip("ing") == rv or rv.rstrip("ing") == gv_core:
            return True
        if gv_core.rstrip("s") == rv or rv.rstrip("s") == gv_core:
            return True

        # Synonym check
        rv_syn = _SYNONYMS.get(rv, "")
        rvt_syn = _SYNONYMS.get(rvt, "")
        if rv_syn and (rv_syn in gv_core or gv_core in rv_syn):
            return True
        if rvt_syn and (rvt_syn in gv_core or gv_core in rvt_syn):
            return True

        return False

    # --- Pronoun resolution ---
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

    subj_words = set(subj.lower().split())
    is_pronoun = bool(subj_words & _PRONOUNS) and len(subj.split()) <= 3

    entity_node = _resolve_entity(G, subj)

    if not entity_node and is_pronoun and sentence_entities:
        # Try each PERSON/ORG entity from the same sentence as the subject
        for sent_ent in sentence_entities:
            if sent_ent["label"] in ("PERSON", "ORG"):
                candidate = _resolve_entity(G, sent_ent["lower"])
                if candidate:
                    # Check if this candidate has actions matching the verb
                    for _, target, data in G.edges(candidate, data=True):
                        if data.get("relation") == "action":
                            edge_verb = data.get("verb", "")
                            node_verb = G.nodes[target].get("verb", "")
                            check_v = edge_verb or node_verb
                            if _verb_matches(check_v, verb, verb_text):
                                entity_node = candidate
                                result["evidence"] = (
                                    f"Pronoun '{subj}' resolved to '{candidate}'; "
                                )
                                break
                    if entity_node:
                        break

        # If still no match, try the entity with most action edges (main character)
        if not entity_node:
            best_entity, best_count = None, 0
            for node, ndata in G.nodes(data=True):
                if ndata.get("node_type") == "entity":
                    action_count = sum(
                        1
                        for _, _, d in G.edges(node, data=True)
                        if d.get("relation") == "action"
                    )
                    if action_count > best_count:
                        best_count = action_count
                        best_entity = node
            if best_entity:
                entity_node = best_entity
                result["evidence"] = (
                    f"Pronoun '{subj}' resolved to main entity '{best_entity}'; "
                )

    if not entity_node:
        result["verdict"] = "unknown"
        result["evidence"] += f"Subject '{subj}' not found in graph"
        return result

    # --- Passive voice: search for entity as TARGET, not actor ---
    if is_passive:
        # Find action nodes that have this entity as a target
        passive_action_edges = []
        for source_node, target_node, data in G.edges(data=True):
            if data.get("relation") == "target" and target_node == entity_node:
                # source_node is the action node — get its parent (actor)
                action_data = G.nodes.get(source_node, {})
                if action_data.get("node_type") == "action":
                    passive_action_edges.append((source_node, action_data))

        # Also search: action nodes where entity_node substring matches any target
        for node, ndata in G.nodes(data=True):
            if ndata.get("node_type") == "action":
                for _, tgt, tdata in G.edges(node, data=True):
                    if tdata.get("relation") == "target":
                        if entity_node in tgt or tgt in entity_node:
                            if (node, ndata) not in passive_action_edges:
                                passive_action_edges.append((node, ndata))

        for action_node, action_data in passive_action_edges:
            action_verb = action_data.get("verb", "")
            if _verb_matches(action_verb, verb, verb_text):
                # Check polarity of the action edge from actor to this action node
                for src, _, edata in G.in_edges(action_node, data=True):
                    if edata.get("relation") == "action":
                        if edata.get("polarity") == "negate":
                            result["verdict"] = "flagged"
                            result[
                                "evidence"
                            ] += f"'{entity_node}' as target of '{verb}' is NEGATED"
                            return result

                result["verdict"] = "grounded"
                result[
                    "evidence"
                ] += f"Passive confirmed: '{entity_node}' is target of '{verb}'"
                return result

        # Passive verb not found — fall through to active check

    # --- Active voice: search for entity as ACTOR ---
    action_edges = []
    for _, target, data in G.edges(entity_node, data=True):
        if data.get("relation") == "action":
            action_edges.append((target, data))

    if not action_edges:
        # Entity exists in graph but has ZERO actions
        # Response claims an action for this entity → fabrication (not synonym gap)
        # Synonyms require something to be synonym OF — absence has no synonyms
        result["verdict"] = "flagged"
        result["evidence"] += (
            f"Entity '{entity_node}' has NO actions in graph but "
            f"response claims '{verb}'"
        )
        return result

    for action_node, edge_data in action_edges:
        edge_verb = edge_data.get("verb", "")
        node_verb = G.nodes[action_node].get("verb", "")
        check_verb = edge_verb or node_verb

        if _verb_matches(check_verb, verb, verb_text):
            # Found matching action
            if edge_data.get("polarity") == "negate":
                result["verdict"] = "flagged"
                result["evidence"] += f"'{entity_node}' → '{verb}' is a NEGATED action"
                return result

            # Action is affirmed — now check objects if present
            if triple["objects"]:
                obj_matched = False
                for obj in triple["objects"]:
                    obj_head = obj["head"]
                    obj_prep = obj.get("prep", "")

                    # Check targets of this action
                    for _, tgt, tdata in G.edges(action_node, data=True):
                        if tdata.get("relation") == "target":
                            if obj_head in tgt or tgt in obj_head:
                                if tdata.get("polarity") == "negate":
                                    result["verdict"] = "flagged"
                                    result["evidence"] += (
                                        f"'{entity_node}' → '{verb}' → '{obj_head}' "
                                        f"matches a NEGATED target"
                                    )
                                    return result
                                else:
                                    obj_matched = True
                                    result["verdict"] = "grounded"
                                    result[
                                        "evidence"
                                    ] += f"Path confirmed: '{entity_node}' → '{verb}' → '{tgt}'"

                    # Check prepositional objects against "where" edges
                    if not obj_matched and obj_prep in (
                        "to",
                        "in",
                        "at",
                        "from",
                        "outside",
                        "near",
                    ):
                        for _, tgt, tdata in G.edges(action_node, data=True):
                            if tdata.get("relation") == "where":
                                if obj_head in tgt or tgt in obj_head:
                                    if tdata.get("polarity") == "negate":
                                        result["verdict"] = "flagged"
                                        result["evidence"] += (
                                            f"'{entity_node}' → '{verb}' → where='{obj_head}' "
                                            f"matches NEGATED location"
                                        )
                                        return result
                                    else:
                                        obj_matched = True
                                        result["verdict"] = "grounded"
                                        result[
                                            "evidence"
                                        ] += f"Path confirmed: '{entity_node}' → '{verb}' → where='{tgt}'"

                    # Check against other attribute edges (negated)
                    if not obj_matched:
                        for _, tgt, tdata in G.edges(action_node, data=True):
                            if tdata.get("polarity") == "negate" and (
                                obj_head in tgt or tgt in obj_head
                            ):
                                result["verdict"] = "flagged"
                                result["evidence"] += (
                                    f"'{entity_node}' → '{verb}' → '{obj_head}' "
                                    f"matches NEGATED {tdata.get('relation', 'attribute')}"
                                )
                                return result

                if obj_matched:
                    return result

                # Objects not matched — check if graph has different targets
                graph_targets = [
                    tgt
                    for _, tgt, tdata in G.edges(action_node, data=True)
                    if tdata.get("relation") in ("target", "where")
                    and tdata.get("polarity") == "affirm"
                ]

                if graph_targets:
                    result["verdict"] = "flagged"
                    result["evidence"] += (
                        f"'{entity_node}' → '{verb}' has targets {graph_targets[:3]} "
                        f"but response says {[o['head'] for o in triple['objects']]}"
                    )
                    return result
                else:
                    # Action exists but has ZERO targets in graph
                    # Response adds targets → fabrication (nothing to be synonym of)
                    result["verdict"] = "flagged"
                    result["evidence"] += (
                        f"'{entity_node}' → '{verb}' has NO targets in graph but "
                        f"response says {[o['head'] for o in triple['objects']]}"
                    )
                    return result

            # Action found, affirmed, no objects in response
            if negated:
                result["verdict"] = "flagged"
                result[
                    "evidence"
                ] += f"Response negates '{verb}' but graph affirms it for '{entity_node}'"
                return result

            result["verdict"] = "grounded"
            result[
                "evidence"
            ] += f"Action '{verb}' confirmed for '{entity_node}' (no objects to verify)"
            return result

    # Verb not found — also check synonyms explicitly
    result["verdict"] = "unknown"
    result["evidence"] += f"Action '{verb}' not found for '{entity_node}'"
    return result


def verify_entity(G: nx.DiGraph, entity: dict) -> dict:
    """Check if a named entity exists in the graph. Returns 'present' (not 'grounded')
    because entity existence alone does not confirm a claim."""
    result = {"check": "entity", "verdict": "unknown", "evidence": ""}
    name = entity["lower"]

    resolved = _resolve_entity(G, name)
    if resolved:
        result["verdict"] = "present"
        result["evidence"] = f"Entity '{name}' found as '{resolved}'"
    else:
        # Only flag if it looks like a proper noun (PERSON, ORG, GPE)
        if entity["label"] in ("PERSON", "ORG", "GPE", "NORP"):
            result["verdict"] = "flagged"
            result["evidence"] = f"Entity '{name}' ({entity['label']}) NOT in graph"
        else:
            result["verdict"] = "unknown"
            result["evidence"] = f"Entity '{name}' ({entity['label']}) not verifiable"

    return result


def verify_relationship(G: nx.DiGraph, rel: dict) -> dict:
    """Verify a relationship word against graph edges."""
    result = {"check": "relationship", "verdict": "unknown", "evidence": ""}
    rel_word = rel["rel_word"]
    context = rel["context_entity"]

    # Find entity in graph
    entity_node = _resolve_entity(G, context)
    if not entity_node:
        result["verdict"] = "unknown"
        result["evidence"] = f"Context entity '{context}' not found"
        return result

    # Check all edges from this entity for relationship matches
    for _, target, data in G.edges(entity_node, data=True):
        rel_type = data.get("relation", "")
        if rel_word in rel_type or rel_type in rel_word:
            if data.get("polarity") == "negate":
                result["verdict"] = "flagged"
                result["evidence"] = (
                    f"Relationship '{rel_word}' for '{entity_node}' is NEGATED"
                )
                return result
            else:
                result["verdict"] = "grounded"
                result["evidence"] = (
                    f"Relationship '{rel_word}' confirmed for '{entity_node}'"
                )
                return result

    # Check if a DIFFERENT relationship exists (mismatch)
    existing_rels = []
    for _, target, data in G.edges(entity_node, data=True):
        rel_type = data.get("relation", "")
        if data.get("polarity") == "affirm" and rel_type not in (
            "action",
            "alias_of",
            "has_number",
            "where",
            "when",
        ):
            existing_rels.append(rel_type)

    if existing_rels:
        result["verdict"] = "flagged"
        result["evidence"] = (
            f"Relationship '{rel_word}' not found for '{entity_node}'; "
            f"graph has: {', '.join(existing_rels[:3])}"
        )
    else:
        result["verdict"] = "unknown"
        result["evidence"] = f"No relationships found for '{entity_node}'"

    return result


def verify_cardinal(G: nx.DiGraph, cardinal: dict) -> dict:
    """Verify a cardinal number against the graph."""
    result = {"check": "cardinal", "verdict": "unknown", "evidence": ""}
    number = cardinal["number"]
    noun = cardinal.get("governing_noun", "")

    if not noun:
        return result

    # Search for number nodes related to this noun
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "number":
            describes = data.get("describes", "")
            value = data.get("value", "")
            if noun in describes or noun in node:
                if value == number or value == number.replace(",", ""):
                    result["verdict"] = "grounded"
                    result["evidence"] = f"Count {number} for '{noun}' confirmed"
                    return result
                else:
                    result["verdict"] = "flagged"
                    result["evidence"] = (
                        f"Count mismatch: response says {number} {noun}, "
                        f"graph says {value}"
                    )
                    return result

    return result


def verify_sentence(G: nx.DiGraph, sentence: str) -> dict:
    """Full verification of a response sentence against the graph."""
    parsed = parse_sentence(sentence)

    all_checks = []

    # Verify SVO triples (pass sentence entities for pronoun resolution)
    for triple in parsed["svo_triples"]:
        check = verify_svo(G, triple, sentence_entities=parsed["entities"])
        all_checks.append(check)

    # Verify named entities
    for entity in parsed["entities"]:
        if entity["label"] in ("PERSON", "ORG", "GPE", "NORP"):
            check = verify_entity(G, entity)
            all_checks.append(check)

    # Verify relationships
    for rel in parsed["relationships"]:
        check = verify_relationship(G, rel)
        all_checks.append(check)

    # Verify cardinals
    for card in parsed["cardinals"]:
        check = verify_cardinal(G, card)
        all_checks.append(check)

    # --- Aggregate verdict ---
    # Separate check types: entity presence is NOT a clearance signal
    verdicts = [c["verdict"] for c in all_checks]
    check_types = [c["check"] for c in all_checks]
    evidence = [c["evidence"] for c in all_checks if c["evidence"]]

    # Count by check type
    svo_checks = [c for c in all_checks if c["check"] == "svo"]
    svo_grounded = sum(1 for c in svo_checks if c["verdict"] == "grounded")
    svo_flagged = sum(1 for c in svo_checks if c["verdict"] == "flagged")
    svo_unknown = sum(1 for c in svo_checks if c["verdict"] == "unknown")

    rel_grounded = sum(
        1
        for c in all_checks
        if c["check"] == "relationship" and c["verdict"] == "grounded"
    )
    rel_flagged = sum(
        1
        for c in all_checks
        if c["check"] == "relationship" and c["verdict"] == "flagged"
    )
    card_grounded = sum(
        1 for c in all_checks if c["check"] == "cardinal" and c["verdict"] == "grounded"
    )
    card_flagged = sum(
        1 for c in all_checks if c["check"] == "cardinal" and c["verdict"] == "flagged"
    )

    structural_grounded = svo_grounded + rel_grounded + card_grounded

    entity_present = sum(
        1 for c in all_checks if c["check"] == "entity" and c["verdict"] == "present"
    )
    entity_flagged = sum(
        1 for c in all_checks if c["check"] == "entity" and c["verdict"] == "flagged"
    )

    any_flagged = (
        svo_flagged > 0 or rel_flagged > 0 or card_flagged > 0 or entity_flagged > 0
    )

    if any_flagged:
        final = "flagged"
    elif svo_grounded >= 1 and svo_unknown == 0:
        # ALL SVO triples confirmed — no unverified clauses in the sentence
        final = "grounded"
    elif rel_grounded >= 1 or card_grounded >= 1:
        # Relationship or cardinal confirmed (and nothing flagged)
        final = "grounded"
    elif svo_grounded >= 1 and svo_unknown > 0:
        # Mixed: some SVO confirmed, some unknown — compound sentence, can't fully clear
        final = "unknown"
    else:
        # Entity presence alone is NOT enough to clear
        final = "unknown"

    return {
        "verdict": final,
        "checks": len(all_checks),
        "grounded_count": structural_grounded,
        "present_count": entity_present,
        "flagged_count": verdicts.count("flagged"),
        "unknown_count": verdicts.count("unknown"),
        "evidence": evidence,
        "parsed_triples": len(parsed["svo_triples"]),
        "parsed_entities": len(parsed["entities"]),
        "parsed_cardinals": len(parsed["cardinals"]),
        "parsed_relationships": len(parsed["relationships"]),
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
    response_level: str
    checks: int
    grounded_count: int
    present_count: int
    flagged_count: int
    unknown_count: int
    parsed_triples: int
    parsed_entities: int
    parsed_cardinals: int
    evidence: str
    l2b_verdict: str


def main():
    fact_tables_path = "experiments/exp30_fact_tables.json"

    if not os.path.exists(fact_tables_path):
        print(f"ERROR: {fact_tables_path} not found.")
        print(
            "Run Stage 1 first: poetry run python experiments/exp30_stage1_extract.py"
        )
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 30 — STAGE 2c: Graph Traversal — Precision Fixes")
    print("=" * 70)
    print(f"  Loading fact tables from: {fact_tables_path}")
    print("  Fix 1: Entity presence ≠ grounded (observation only)")
    print("  Fix 2: Only SVO/relationship/cardinal can clear")
    print("  Fix 3: SVO must verify full path including objects")
    print("  No Gemini calls. Deterministic.")
    print("=" * 70)

    with open(fact_tables_path) as f:
        fact_tables = json.load(f)

    n_valid = sum(1 for v in fact_tables.values() if v.get("fact_table"))
    print(f"  Fact tables loaded: {n_valid} valid")

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"  Cases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results: list[SentenceResult] = []
    verdict_counts = {"grounded": 0, "flagged": 0, "unknown": 0}
    safety_violations = []
    hallucination_catches = []
    graph_stats = []
    t0 = time.time()

    for ci, case in enumerate(cases):
        response = case.request.candidate_answer or ""
        gt = case.ground_truth
        labeled = label_sentences(response, gt.span_annotations, gt.response_level)

        ft_data = fact_tables.get(case.case_id, {})
        ft = ft_data.get("fact_table")

        if not ft:
            for sl in labeled:
                all_results.append(
                    SentenceResult(
                        case_id=case.case_id,
                        sentence_idx=sl["idx"],
                        sentence=sl["sentence"][:150],
                        gt_label=sl["label"],
                        gt_type=sl["type"],
                        response_level=gt.response_level,
                        checks=0,
                        grounded_count=0,
                        flagged_count=0,
                        unknown_count=0,
                        parsed_triples=0,
                        parsed_entities=0,
                        parsed_cardinals=0,
                        evidence="no_fact_table",
                        l2b_verdict="unknown",
                    )
                )
                verdict_counts["unknown"] += 1
            continue

        # Phase 1b: Build graph
        G = json_to_graph(ft)
        graph_stats.append(
            {
                "case_id": case.case_id,
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
            }
        )

        # Phase 2: Verify each sentence
        for sl in labeled:
            sent = sl["sentence"]
            comp = verify_sentence(G, sent)

            verdict_counts[comp["verdict"]] += 1

            if comp["verdict"] == "grounded" and sl["label"] == "hallucinated":
                safety_violations.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "evidence": comp["evidence"][:3],
                    }
                )

            if comp["verdict"] == "flagged" and sl["label"] == "hallucinated":
                hallucination_catches.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "evidence": comp["evidence"][:3],
                    }
                )

            all_results.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    response_level=gt.response_level,
                    checks=comp["checks"],
                    grounded_count=comp["grounded_count"],
                    present_count=comp["present_count"],
                    flagged_count=comp["flagged_count"],
                    unknown_count=comp["unknown_count"],
                    parsed_triples=comp["parsed_triples"],
                    parsed_entities=comp["parsed_entities"],
                    parsed_cardinals=comp["parsed_cardinals"],
                    evidence="; ".join(comp["evidence"][:5]),
                    l2b_verdict=comp["verdict"],
                )
            )

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"\n--- {ci+1}/{len(cases)} cases | {len(all_results)} sents | {elapsed:.1f}s ---"
            )
            print(
                f"  Grounded: {verdict_counts['grounded']}  Flagged: {verdict_counts['flagged']}  Unknown: {verdict_counts['unknown']}"
            )
            print(
                f"  Safety violations: {len(safety_violations)}  Catches: {len(hallucination_catches)}"
            )

    elapsed = time.time() - t0

    # === RESULTS ===
    print(f"\n{'='*70}")
    print(f"FINAL — {len(all_results)} sentences, {elapsed:.1f}s")
    print(f"{'='*70}")

    print("\n  L2b VERDICTS")
    print(
        f"    Grounded:  {verdict_counts['grounded']:3d} ({verdict_counts['grounded']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Flagged:   {verdict_counts['flagged']:3d} ({verdict_counts['flagged']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Unknown:   {verdict_counts['unknown']:3d} ({verdict_counts['unknown']/len(all_results)*100:.1f}%)"
    )

    # Graph stats
    if graph_stats:
        avg_nodes = sum(g["nodes"] for g in graph_stats) / len(graph_stats)
        avg_edges = sum(g["edges"] for g in graph_stats) / len(graph_stats)
        print(
            f"\n  Graph stats: avg {avg_nodes:.0f} nodes, {avg_edges:.0f} edges per case"
        )

    # Safety
    print(f"\n{'='*70}")
    print("SAFETY: Hallucinated sentences incorrectly cleared")
    print(f"{'='*70}")
    if safety_violations:
        print(
            f"  *** {len(safety_violations)} SAFETY VIOLATIONS (was 7 in Stage 2b) ***"
        )
        for sv in safety_violations:
            print(f"    {sv['case_id']} S{sv['sentence_idx']} GT={sv['gt_type']}")
            print(f"      Sentence: {sv['sentence']}")
            print(f"      Evidence: {sv['evidence']}")
    else:
        print("  *** 0 SAFETY VIOLATIONS *** (was 7 in Stage 2b)")

    # Catches
    print(f"\n{'='*70}")
    print("HALLUCINATION CATCHES")
    print(f"{'='*70}")
    print(
        f"  {len(hallucination_catches)} / 16 caught ({len(hallucination_catches)/16*100:.0f}% recall)"
        f"  (was 4/16 in Stage 2b)"
    )
    for hc in hallucination_catches:
        print(f"    {hc['case_id']} S{hc['sentence_idx']} GT={hc['gt_type']}")
        print(f"      Sentence: {hc['sentence']}")
        print(f"      Evidence: {hc['evidence']}")

    # Missed
    caught_ids = {(hc["case_id"], hc["sentence_idx"]) for hc in hallucination_catches}
    print(f"\n  Missed hallucinations ({16 - len(hallucination_catches)}):")
    for r in all_results:
        if (
            r.gt_label == "hallucinated"
            and (r.case_id, r.sentence_idx) not in caught_ids
        ):
            print(
                f"    {r.case_id} S{r.sentence_idx} GT={r.gt_type} verdict={r.l2b_verdict}"
            )
            print(f"      Sentence: {r.sentence[:100]}")
            if r.evidence:
                print(f"      Evidence: {r.evidence[:100]}")

    # Verdict vs GT
    print(f"\n{'='*70}")
    print("VERDICT vs GROUND TRUTH")
    print(f"{'='*70}")
    for verdict in ["grounded", "flagged", "unknown"]:
        vr = [r for r in all_results if r.l2b_verdict == verdict]
        v_clean = sum(1 for r in vr if r.gt_label == "clean")
        v_hall = sum(1 for r in vr if r.gt_label == "hallucinated")
        print(
            f"  {verdict:12s}: {len(vr):3d} total ({v_clean} clean, {v_hall} hallucinated)"
        )

    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON: L1 vs L2 (regex) vs L2b (graph)")
    print(f"{'='*70}")
    print("  L1 (Exp 29b A1):  cleared 21, caught 4/16, 0 safety violations")
    print("  L2 (Exp 30):      cleared 112, caught 7/16, 7 safety violations")
    print("  L2b (graph):      cleared 173, caught 4/16, 7 safety violations")
    print(
        f"  L2c (precise):    cleared {verdict_counts['grounded']}, caught {len(hallucination_catches)}/16, {len(safety_violations)} safety violations"
    )

    # Parse stats
    print(f"\n{'='*70}")
    print("SPACY PARSE STATS")
    print(f"{'='*70}")
    avg_triples = sum(r.parsed_triples for r in all_results) / len(all_results)
    avg_ents = sum(r.parsed_entities for r in all_results) / len(all_results)
    avg_cards = sum(r.parsed_cardinals for r in all_results) / len(all_results)
    print(f"  Avg SVO triples per sentence: {avg_triples:.1f}")
    print(f"  Avg named entities per sentence: {avg_ents:.1f}")
    print(f"  Avg cardinals per sentence: {avg_cards:.1f}")

    # Save
    output = {
        "elapsed_s": round(elapsed, 1),
        "verdict_counts": verdict_counts,
        "safety_violations": len(safety_violations),
        "hallucination_catches": len(hallucination_catches),
        "graph_stats": graph_stats,
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp30_stage2c_graph_precise_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp30_stage2c_graph_precise_results.json")


if __name__ == "__main__":
    main()
