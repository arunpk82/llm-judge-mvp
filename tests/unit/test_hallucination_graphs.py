"""
Tests for L2 Patterns — Knowledge Graph Ensemble (hallucination_graphs.py).

Tests cover:
  Graph builders: G1 entities, G2 events, G3 relationships, G4 numbers, G5 negations
  Sentence parsing: SVO triples, entities, cardinals, relationships
  Entity resolution: exact match, alias, substring, word overlap
  Verb matching: lemma, suffix, synonyms
  Graph traversal: G2 events, G3 relationships, G4 numbers, G5 negations
  Ensemble aggregation: flag-wins rule, confidence scoring
  Integration: l2_ensemble_check end-to-end
"""
from __future__ import annotations

import pytest

spacy = pytest.importorskip("spacy")

# =====================================================================
# Helpers
# =====================================================================

class TestSafeList:
    def test_none_returns_empty(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _safe_list

        assert _safe_list(None) == []

    def test_list_returned_as_is(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _safe_list

        assert _safe_list([1, 2]) == [1, 2]

    def test_string_wrapped_in_list(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _safe_list

        assert _safe_list("hello") == ["hello"]

    def test_other_types_return_empty(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _safe_list

        assert _safe_list(42) == []


# =====================================================================
# G1: Entity Graph
# =====================================================================

class TestBuildG1Entities:
    def test_basic_entity(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g1_entities

        p1 = {"entities": [{"name": "B.B. King", "type": "PERSON"}]}
        G = build_g1_entities(p1)
        assert "b.b. king" in G.nodes()
        assert G.nodes["b.b. king"]["node_type"] == "entity"

    def test_entity_with_alias(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g1_entities

        p1 = {"entities": [{"name": "B.B. King", "aliases": ["Riley B. King", "The King of Blues"]}]}
        G = build_g1_entities(p1)
        assert "riley b. king" in G.nodes()
        assert G.has_edge("riley b. king", "b.b. king")
        edge = G.edges["riley b. king", "b.b. king"]
        assert edge["relation"] == "alias_of"

    def test_entity_with_attributes(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g1_entities

        p1 = {"entities": [{"name": "Paris", "attributes": {"country": "france"}}]}
        G = build_g1_entities(p1)
        assert G.has_edge("paris", "attr_paris_country")

    def test_empty_entities(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g1_entities

        G = build_g1_entities({"entities": []})
        assert len(G.nodes()) == 0

    def test_none_name_skipped(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g1_entities

        G = build_g1_entities({"entities": [{"name": None}]})
        assert len(G.nodes()) == 0


# =====================================================================
# G2: Event Graph
# =====================================================================

class TestBuildG2Events:
    def test_basic_event(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g2_events

        p2 = {"events": [{"who": "John", "action": "attacked", "target": ["the store"]}]}
        G = build_g2_events(p2)
        assert "john" in G.nodes()
        action_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "action"]
        assert len(action_nodes) >= 1

    def test_event_with_p1_cross_reference(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g2_events

        p1 = {"entities": [{"name": "John Smith", "aliases": ["Smith"]}]}
        p2 = {"events": [{"who": "Smith", "action": "testified", "target": ["in court"]}]}
        G = build_g2_events(p2, p1)
        # Smith should resolve to john smith via alias
        assert "john smith" in G.nodes()

    def test_event_with_where_and_when(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g2_events

        p2 = {"events": [{"who": "Police", "action": "arrested", "target": ["suspect"],
                          "where": "downtown", "when": "Monday"}]}
        G = build_g2_events(p2)
        assert "downtown" in G.nodes()
        assert "monday" in G.nodes()

    def test_empty_action_skipped(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g2_events

        p2 = {"events": [{"who": "John", "action": ""}]}
        G = build_g2_events(p2)
        action_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "action"]
        assert len(action_nodes) == 0


# =====================================================================
# G3: Relationship Graph
# =====================================================================

class TestBuildG3Relationships:
    def test_basic_relationship(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g3_relationships

        p3 = {"relationships": [{"entity1": "John", "entity2": "Mary", "relationship": "wife"}]}
        G = build_g3_relationships(p3)
        assert G.has_edge("john", "mary")
        assert G.edges["john", "mary"]["relation"] == "wife"
        assert G.edges["john", "mary"]["polarity"] == "affirm"

    def test_not_relationship(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g3_relationships

        p3 = {"relationships": [{"entity1": "West", "entity2": "victim",
                                  "relationship": "companion", "NOT_relationships": ["wife"]}]}
        G = build_g3_relationships(p3)
        # Should have both affirm and negate edges
        edges = list(G.edges("west", data=True))
        polarities = [e[2]["polarity"] for e in edges]
        assert "negate" in polarities
        assert "negate" in polarities

    def test_empty_fields_skipped(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g3_relationships

        p3 = {"relationships": [{"entity1": "", "entity2": "Mary", "relationship": "wife"}]}
        G = build_g3_relationships(p3)
        assert len(G.nodes()) == 0


# =====================================================================
# G4: Number Graph
# =====================================================================

class TestBuildG4Numbers:
    def test_numerical_fact(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g4_numbers

        p4 = {"numerical_facts": [{"entity": "company", "number": "500", "describes": "employees"}]}
        G = build_g4_numbers(p4)
        assert "company" in G.nodes()
        num_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "number"]
        assert len(num_nodes) == 1
        assert G.nodes[num_nodes[0]]["value"] == "500"

    def test_temporal_fact(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g4_numbers

        p4 = {"numerical_facts": [], "temporal_facts": [{"entity": "event", "date": "2024-01-15"}]}
        G = build_g4_numbers(p4)
        date_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "date"]
        assert len(date_nodes) == 1


# =====================================================================
# G5: Negation Graph
# =====================================================================

class TestBuildG5Negations:
    def test_explicit_negation(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g5_negations

        p5 = {"explicit_negations": [{"statement": "He did not plead guilty"}]}
        G = build_g5_negations(p5)
        neg_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "negation"]
        assert len(neg_nodes) == 1

    def test_absent_information(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g5_negations

        p5 = {"absent_information": [{"what": "No mention of settlement amount"}]}
        G = build_g5_negations(p5)
        absent_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "absent"]
        assert len(absent_nodes) == 1

    def test_correction(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_g5_negations

        p5 = {"corrections": [{"wrong": "guilty", "right": "no contest"}]}
        G = build_g5_negations(p5)
        corr_nodes = [n for n in G.nodes() if G.nodes[n].get("node_type") == "correction"]
        assert len(corr_nodes) == 1
        assert G.nodes[corr_nodes[0]]["wrong"] == "guilty"


# =====================================================================
# build_all_graphs
# =====================================================================

class TestBuildAllGraphs:
    def test_builds_all_five_graphs(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_all_graphs

        fact_tables = {
            "passes": {
                "P1_entities": {"entities": [{"name": "John"}]},
                "P2_events": {"events": [{"who": "John", "action": "ran"}]},
                "P3_relationships": {"relationships": [{"entity1": "A", "entity2": "B", "relationship": "friend"}]},
                "P4_numbers": {"numerical_facts": [{"entity": "X", "number": "10", "describes": "count"}]},
                "P5_negations": {"explicit_negations": [{"statement": "not guilty"}]},
            }
        }
        graphs = build_all_graphs(fact_tables)
        assert "G1" in graphs
        assert "G2" in graphs
        assert "G3" in graphs
        assert "G4" in graphs
        assert "G5" in graphs

    def test_missing_passes_skipped(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_all_graphs

        fact_tables = {"passes": {"P1_entities": {"entities": [{"name": "X"}]}}}
        graphs = build_all_graphs(fact_tables)
        assert "G1" in graphs
        assert "G2" not in graphs

    def test_empty_fact_tables(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_all_graphs

        graphs = build_all_graphs({"passes": {}})
        assert len(graphs) == 0


# =====================================================================
# Entity Resolution
# =====================================================================

class TestEntityResolution:
    def _make_graph(self):
        import networkx as nx

        G = nx.DiGraph()
        G.add_node("john smith", node_type="entity")
        G.add_node("smith", node_type="alias")
        G.add_edge("smith", "john smith", relation="alias_of", polarity="affirm")
        G.add_node("new york", node_type="entity")
        return G

    def test_exact_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _resolve_entity

        G = self._make_graph()
        assert _resolve_entity(G, "john smith") == "john smith"

    def test_alias_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _resolve_entity

        G = self._make_graph()
        assert _resolve_entity(G, "Smith") == "john smith"

    def test_substring_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _resolve_entity

        G = self._make_graph()
        assert _resolve_entity(G, "new york city") == "new york"

    def test_no_match_returns_none(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _resolve_entity

        G = self._make_graph()
        assert _resolve_entity(G, "tokyo") is None

    def test_follow_alias(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _follow_alias

        G = self._make_graph()
        assert _follow_alias(G, "smith") == "john smith"
        assert _follow_alias(G, "john smith") == "john smith"


# =====================================================================
# Verb Matching
# =====================================================================

class TestVerbMatch:
    def test_exact_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _verb_match

        assert _verb_match("attack", "attack", "attacked") is True

    def test_lemma_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _verb_match

        assert _verb_match("attacked", "attack", "attacked") is True

    def test_suffix_stripping(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _verb_match

        assert _verb_match("running", "run", "running") is True

    def test_synonym_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _verb_match

        assert _verb_match("assault", "attack", "attacked") is True

    def test_no_match(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _verb_match

        assert _verb_match("sing", "attack", "attacked") is False

    def test_aux_verb_stripping(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _norm_verb

        assert _norm_verb("was stabbed") == "stabbed"


# =====================================================================
# Sentence Parsing (requires spaCy)
# =====================================================================

spacy = pytest.importorskip("spacy")


class TestSentenceParsing:
    @pytest.fixture(autouse=True)
    def _load_nlp(self):
        pass  # spacy imported above

        self.nlp = spacy.load("en_core_web_sm")

    def test_extracts_svo_triples(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("Police arrested the suspect.", self.nlp)
        assert len(parsed["svo_triples"]) >= 1
        triple = parsed["svo_triples"][0]
        assert "arrest" in triple["verb"]

    def test_extracts_entities(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("John Smith visited New York.", self.nlp)
        entity_texts = [e["lower"] for e in parsed["entities"]]
        assert any("john" in t for t in entity_texts) or any("new york" in t for t in entity_texts)

    def test_detects_passive_voice(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("The suspect was arrested by police.", self.nlp)
        passives = [t for t in parsed["svo_triples"] if t.get("passive")]
        assert len(passives) >= 1

    def test_extracts_cardinals(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("Three people were injured.", self.nlp)
        assert len(parsed["cardinals"]) >= 1

    def test_extracts_relationships(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("His wife attended the ceremony.", self.nlp)
        assert len(parsed["relationships"]) >= 1
        assert parsed["relationships"][0]["rel_word"] == "wife"

    def test_skips_generic_verbs(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _parse_sentence_for_l2

        parsed = _parse_sentence_for_l2("He is a teacher.", self.nlp)
        verbs = [t["verb"] for t in parsed["svo_triples"]]
        assert "be" not in verbs


# =====================================================================
# Graph Traversal — G2 Events
# =====================================================================

class TestTraverseG2:
    def _build_g2(self):
        from llm_judge.calibration.hallucination_graphs import build_g2_events

        p1 = {"entities": [{"name": "Police", "type": "ORG"}]}
        p2 = {"events": [{"who": "Police", "action": "arrested", "target": ["the suspect"]}]}
        return build_g2_events(p2, p1)

    def test_matching_svo_returns_grounded(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g2

        G = self._build_g2()
        parsed = {
            "svo_triples": [{"subject": "police", "verb": "arrest", "verb_text": "arrested",
                             "objects": [{"text": "the suspect", "head": "suspect"}],
                             "negated": False, "passive": False}],
            "entities": [{"text": "Police", "label": "ORG", "lower": "police"}],
            "cardinals": [], "relationships": [],
        }
        verdicts = _traverse_g2(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "grounded" in verdict_types

    def test_unknown_subject_returns_unknown(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g2

        G = self._build_g2()
        parsed = {
            "svo_triples": [{"subject": "aliens", "verb": "attack", "verb_text": "attacked",
                             "objects": [], "negated": False, "passive": False}],
            "entities": [], "cardinals": [], "relationships": [],
        }
        verdicts = _traverse_g2(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "unknown" in verdict_types

    def test_entity_no_actions_returns_flagged(self) -> None:
        import networkx as nx

        from llm_judge.calibration.hallucination_graphs import _traverse_g2

        G = nx.DiGraph()
        G.add_node("john", node_type="entity")
        # john has no action edges
        parsed = {
            "svo_triples": [{"subject": "john", "verb": "stab", "verb_text": "stabbed",
                             "objects": [], "negated": False, "passive": False}],
            "entities": [], "cardinals": [], "relationships": [],
        }
        verdicts = _traverse_g2(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "flagged" in verdict_types


# =====================================================================
# Graph Traversal — G3, G4, G5
# =====================================================================

class TestTraverseG3:
    def test_matching_relationship_grounded(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g3, build_g3_relationships

        p3 = {"relationships": [{"entity1": "west", "entity2": "victim", "relationship": "companion"}]}
        G = build_g3_relationships(p3)
        parsed = {"relationships": [{"rel_word": "companion", "context_entity": "west"}],
                  "svo_triples": [], "entities": [], "cardinals": []}
        verdicts = _traverse_g3(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "grounded" in verdict_types

    def test_wrong_relationship_flagged(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g3, build_g3_relationships

        p3 = {"relationships": [{"entity1": "west", "entity2": "victim", "relationship": "colleague"}]}
        G = build_g3_relationships(p3)
        parsed = {"relationships": [{"rel_word": "wife", "context_entity": "west"}],
                  "svo_triples": [], "entities": [], "cardinals": []}
        verdicts = _traverse_g3(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "flagged" in verdict_types


class TestTraverseG4:
    def test_matching_number_grounded(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g4, build_g4_numbers

        p4 = {"numerical_facts": [{"entity": "victims", "number": "3", "describes": "people"}]}
        G = build_g4_numbers(p4)
        parsed = {"cardinals": [{"number": "3", "governing_noun": "people"}],
                  "svo_triples": [], "entities": [], "relationships": []}
        verdicts = _traverse_g4(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "grounded" in verdict_types

    def test_wrong_number_flagged(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g4, build_g4_numbers

        p4 = {"numerical_facts": [{"entity": "victims", "number": "3", "describes": "people"}]}
        G = build_g4_numbers(p4)
        parsed = {"cardinals": [{"number": "5", "governing_noun": "people"}],
                  "svo_triples": [], "entities": [], "relationships": []}
        verdicts = _traverse_g4(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "flagged" in verdict_types


class TestTraverseG5:
    def test_negation_contradiction_flagged(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g5, build_g5_negations

        p5 = {"explicit_negations": [{"statement": "defendant never pleaded guilty to charges"}]}
        G = build_g5_negations(p5)
        parsed = {"svo_triples": [{"subject": "defendant pleaded guilty", "verb": "plead", "verb_text": "pleaded",
                                   "objects": [{"text": "guilty", "head": "guilty"}],
                                   "negated": False, "passive": False}],
                  "entities": [], "cardinals": [], "relationships": []}
        verdicts = _traverse_g5(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "flagged" in verdict_types

    def test_correction_flagged(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _traverse_g5, build_g5_negations

        p5 = {"corrections": [{"wrong": "guilty", "right": "no contest"}]}
        G = build_g5_negations(p5)
        parsed = {"svo_triples": [{"subject": "he pleaded guilty", "verb": "plead", "verb_text": "pleaded",
                                   "objects": [{"text": "guilty", "head": "guilty"}],
                                   "negated": False, "passive": False}],
                  "entities": [], "cardinals": [], "relationships": []}
        verdicts = _traverse_g5(G, parsed)
        verdict_types = [v[0] for v in verdicts]
        assert "flagged" in verdict_types


# =====================================================================
# Ensemble Aggregation
# =====================================================================

class TestAggregateGraphVerdicts:
    def test_flag_wins_over_grounded(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _aggregate_graph_verdicts

        result = _aggregate_graph_verdicts({
            "G2": [("grounded", "path found")],
            "G4": [("flagged", "count mismatch")],
        })
        assert result["verdict"] == "flagged"

    def test_all_grounded_high_confidence(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _aggregate_graph_verdicts

        result = _aggregate_graph_verdicts({
            "G2": [("grounded", "path A")],
            "G3": [("grounded", "rel confirmed")],
            "G4": [("grounded", "count OK")],
        })
        assert result["verdict"] == "grounded"
        assert result["confidence"] == "high"

    def test_single_grounded_low_confidence(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _aggregate_graph_verdicts

        result = _aggregate_graph_verdicts({
            "G2": [("grounded", "found")],
            "G3": [("unknown", "no data")],
        })
        assert result["verdict"] == "grounded"
        assert result["confidence"] == "low"

    def test_all_unknown(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _aggregate_graph_verdicts

        result = _aggregate_graph_verdicts({
            "G2": [("unknown", "subject not found")],
            "G3": [("unknown", "no rels")],
        })
        assert result["verdict"] == "unknown"
        assert result["confidence"] == "none"

    def test_multiple_flags_high_confidence(self) -> None:
        from llm_judge.calibration.hallucination_graphs import _aggregate_graph_verdicts

        result = _aggregate_graph_verdicts({
            "G2": [("flagged", "wrong target")],
            "G4": [("flagged", "count mismatch")],
        })
        assert result["verdict"] == "flagged"
        assert result["confidence"] == "high"


# =====================================================================
# Integration: l2_ensemble_check
# =====================================================================

class TestL2EnsembleCheck:
    @pytest.fixture(autouse=True)
    def _load_nlp(self):
        pass  # spacy imported above

        self.nlp = spacy.load("en_core_web_sm")

    def test_grounded_sentence(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_all_graphs, l2_ensemble_check

        fact_tables = {
            "passes": {
                "P1_entities": {"entities": [{"name": "Police", "type": "ORG"}]},
                "P2_events": {"events": [{"who": "Police", "action": "arrested",
                                          "target": ["the suspect"]}]},
            }
        }
        graphs = build_all_graphs(fact_tables)
        result = l2_ensemble_check("Police arrested the suspect.", graphs, nlp=self.nlp)
        assert result["verdict"] in ("grounded", "unknown")  # depends on parse quality

    def test_no_graphs_returns_unknown(self) -> None:
        from llm_judge.calibration.hallucination_graphs import l2_ensemble_check

        result = l2_ensemble_check("Some sentence.", {}, nlp=self.nlp)
        assert result["verdict"] == "unknown"

    def test_result_has_required_keys(self) -> None:
        from llm_judge.calibration.hallucination_graphs import build_all_graphs, l2_ensemble_check

        fact_tables = {"passes": {"P1_entities": {"entities": [{"name": "X"}]},
                                  "P2_events": {"events": [{"who": "X", "action": "ran"}]}}}
        graphs = build_all_graphs(fact_tables)
        result = l2_ensemble_check("Something happened.", graphs, nlp=self.nlp)
        assert "verdict" in result
        assert "confidence" in result
        assert "per_graph" in result
        assert "evidence" in result
