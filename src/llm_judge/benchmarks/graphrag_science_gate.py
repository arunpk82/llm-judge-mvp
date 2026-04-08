"""
Experiment 11: GraphRAG Science Gate — spaCy SVO Extraction & Comparison.

Approach: Syntax-based (AST-like) triplet extraction using dependency parsing.
No neural generation — only extracts what the text actually says.
  1. Parse source and response with spaCy dependency parser
  2. Extract (Subject, Verb, Object) + prepositional modifiers from parse tree
  3. Compare triplets: predicate mismatch = fact reassignment, no match = fabrication

Tests on BOTH false negatives AND true negatives.

Usage:
    poetry run python -m llm_judge.benchmarks.graphrag_science_gate --max-fn 10 --max-tn 10
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _compute_grounding_ratio, _split_sentences
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)
SPACY_MODEL = "en_core_web_sm"


@dataclass
class Triplet:
    subject: str
    predicate: str
    obj: str
    modifiers: dict[str, str] = field(default_factory=dict)

    def __str__(self):
        mods = ", ".join(f"{k}={v}" for k, v in self.modifiers.items())
        base = f"({self.subject}, {self.predicate}, {self.obj})"
        return f"{base} [{mods}]" if mods else base


@dataclass
class TripletComparison:
    response_triplet: Triplet
    best_match: Triplet | None
    match_type: str  # exact, predicate_mismatch, modifier_mismatch, no_match
    detail: str


@dataclass
class SentenceResult:
    sentence: str
    sentence_idx: int
    response_triplets: list[Triplet]
    comparisons: list[TripletComparison]
    has_mismatch: bool
    has_fabrication: bool


@dataclass
class CaseResult:
    case_id: str
    ground_truth: str
    gate1_ratio: float
    gate1_min_sim: float
    source_triplets_count: int
    response_triplets_count: int
    predicate_mismatches: int
    modifier_mismatches: int
    fabrications: int
    graphrag_decision: str
    correct: bool
    sentences: list[SentenceResult]


def load_spacy_model():
    """Load spaCy model for dependency parsing."""
    import spacy

    print(f"Loading spaCy model: {SPACY_MODEL}")
    start = time.time()
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError:
        print(f"  Downloading {SPACY_MODEL}...")
        spacy.cli.download(SPACY_MODEL)
        nlp = spacy.load(SPACY_MODEL)
    print(f"  Loaded in {time.time() - start:.1f}s")
    return nlp


def _get_subtree_text(token) -> str:
    """Get the full text of a token's subtree (compound nouns, etc)."""
    subtree = sorted(token.subtree, key=lambda t: t.i)
    # Filter to relevant dependents (compounds, modifiers, determiners)
    parts = []
    for t in subtree:
        if t.dep_ in ("compound", "amod", "nummod", "det", "poss", "case",
                       "flat", "flat:name", "appos") or t == token:
            parts.append(t)
        elif t.dep_ == "punct":
            continue
        elif t.dep_ in ("prep", "agent", "relcl", "advcl", "ccomp", "xcomp",
                         "acl", "conj", "cc"):
            break  # Don't include clausal dependents in entity text
        else:
            parts.append(t)
    parts = sorted(set(parts), key=lambda t: t.i)
    return " ".join(t.text for t in parts).strip()


def _get_entity_text(token) -> str:
    """Get named entity text if token is part of one, otherwise compound text."""
    if token.ent_type_:
        # Return the full named entity span
        for ent in token.doc.ents:
            if token.i >= ent.start and token.i < ent.end:
                return ent.text
    # Fallback: get compound noun phrase
    compounds = [t for t in token.lefts if t.dep_ in ("compound", "amod", "nummod")]
    if compounds:
        return " ".join(t.text for t in sorted(compounds + [token], key=lambda t: t.i))
    return token.text


def extract_svo_triplets(text: str, nlp: Any) -> list[Triplet]:
    """
    Extract Subject-Verb-Object triplets from text using dependency parsing.
    Also extracts prepositional modifiers (on March 26, in New York, etc).
    """
    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        for token in sent:
            # Find verb roots and auxiliary constructions
            if token.pos_ not in ("VERB", "AUX"):
                continue
            if token.dep_ in ("aux", "auxpass"):
                continue  # Skip auxiliary verbs, process the main verb

            verb = token
            verb_text = verb.lemma_

            # Handle passive: "was arrested" → verb = "arrest"
            if any(child.dep_ == "auxpass" for child in verb.children):
                verb_text = verb.lemma_

            # Find subjects
            subjects = []
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(_get_entity_text(child))
                elif child.dep_ == "conj" and any(
                    c.dep_ in ("nsubj", "nsubjpass") for c in child.head.children
                ):
                    # Coordinated subject
                    pass

            if not subjects:
                # Check parent for subject (in clausal complements)
                if verb.dep_ in ("xcomp", "ccomp", "advcl"):
                    for child in verb.head.children:
                        if child.dep_ in ("nsubj", "nsubjpass"):
                            subjects.append(_get_entity_text(child))

            # Find objects
            objects = []
            for child in verb.children:
                if child.dep_ in ("dobj", "attr", "oprd"):
                    objects.append(_get_entity_text(child))
                elif child.dep_ == "prep":
                    # Prepositional objects: "arrested on March 26"
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            objects.append(_get_entity_text(pobj))

            # Extract prepositional modifiers
            modifiers: dict[str, str] = {}
            for child in verb.children:
                if child.dep_ == "prep":
                    prep_text = child.text.lower()
                    for pobj in child.children:
                        if pobj.dep_ == "pobj":
                            mod_text = _get_entity_text(pobj)
                            modifiers[prep_text] = mod_text

            # Also check for temporal/location NER
            for child in verb.children:
                if child.dep_ in ("npadvmod", "tmod"):
                    modifiers["time"] = _get_entity_text(child)

            # Build triplets
            if subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        triplets.append(Triplet(
                            subject=subj, predicate=verb_text,
                            obj=obj, modifiers=modifiers,
                        ))
            elif subjects and modifiers:
                # Intransitive with modifier: "she was arrested on March 26"
                for subj in subjects:
                    for prep, mod_val in modifiers.items():
                        triplets.append(Triplet(
                            subject=subj, predicate=verb_text,
                            obj=mod_val, modifiers={prep: mod_val},
                        ))
            elif subjects and not objects:
                # Intransitive: "the airline is investigating"
                for subj in subjects:
                    triplets.append(Triplet(
                        subject=subj, predicate=verb_text,
                        obj="(intransitive)", modifiers=modifiers,
                    ))

    return triplets


def _normalize(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _entity_overlap(a: str, b: str) -> float:
    wa, wb = set(_normalize(a).split()), set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / min(len(wa), len(wb))


def compare_triplets(
    resp_triplets: list[Triplet],
    src_triplets: list[Triplet],
) -> list[TripletComparison]:
    comparisons = []
    for rt in resp_triplets:
        if rt.obj == "(intransitive)":
            continue  # Skip intransitive verbs for comparison

        best_match, best_type = None, "no_match"
        best_detail = f"No source triplet matches {rt}"
        best_score = 0.0

        for st in src_triplets:
            if st.obj == "(intransitive)":
                continue

            # Check subject overlap
            subj_sim = _entity_overlap(rt.subject, st.subject)
            # Check if objects share meaning
            obj_sim = _entity_overlap(rt.obj, st.obj)

            # Also check cross-match (subject-object swap)
            cross1 = _entity_overlap(rt.subject, st.obj)
            cross2 = _entity_overlap(rt.obj, st.subject)

            fwd_score = (subj_sim + obj_sim) / 2
            cross_score = (cross1 + cross2) / 2
            score = max(fwd_score, cross_score)

            if score < 0.3:
                continue

            if score > best_score:
                best_score = score
                best_match = st

                # Check verb/predicate similarity
                pred_sim = _entity_overlap(rt.predicate, st.predicate)

                if pred_sim >= 0.5 and score >= 0.5:
                    # Check modifiers
                    mod_mismatch = False
                    for key in rt.modifiers:
                        if key in st.modifiers:
                            if _entity_overlap(rt.modifiers[key], st.modifiers[key]) < 0.3:
                                mod_mismatch = True
                                break

                    if mod_mismatch:
                        best_type = "modifier_mismatch"
                        best_detail = (
                            f"Same verb+entities, different modifier: "
                            f"response={rt.modifiers} vs source={st.modifiers}"
                        )
                    else:
                        best_type = "exact"
                        best_detail = f"Matches: {st}"
                elif score >= 0.5:
                    best_type = "predicate_mismatch"
                    best_detail = (
                        f"Same entities ({rt.subject}/{rt.obj}), "
                        f"different verb: response='{rt.predicate}' vs source='{st.predicate}'"
                    )
                else:
                    best_type = "partial"
                    best_detail = f"Partial match with {st}"

        comparisons.append(TripletComparison(
            response_triplet=rt, best_match=best_match,
            match_type=best_type, detail=best_detail,
        ))
    return comparisons


def find_test_cases(max_cases=500, max_fn=10, max_tn=10):
    adapter = RAGTruthAdapter()
    get_embedding_provider()  # init singleton
    fns: list[tuple] = []
    tns: list[tuple] = []
    for case in adapter.load_cases(max_cases=max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue
        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        ctx = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        resp = case.request.candidate_answer
        ratio, min_sim = _compute_grounding_ratio(resp, ctx, similarity_threshold=0.60)
        if ratio >= 0.80 and min_sim >= 0.30:
            src = "\n".join(ctx_parts) if ctx_parts else ctx
            entry = (case, src, resp, ratio, min_sim)
            if gt == "fail" and len(fns) < max_fn:
                fns.append(entry)
            elif gt == "pass" and len(tns) < max_tn:
                tns.append(entry)
        if len(fns) >= max_fn and len(tns) >= max_tn:
            break
    return fns, tns


def run_graphrag_on_case(case, source_doc, response, ratio, min_sim, gt, nlp):
    source_triplets = extract_svo_triplets(source_doc, nlp)
    resp_sents = _split_sentences(response)
    sent_results = []
    total_mm, total_mod_mm, total_fab, total_rt = 0, 0, 0, 0

    for i, sent in enumerate(resp_sents):
        rt = extract_svo_triplets(sent, nlp)
        total_rt += len(rt)
        comps = compare_triplets(rt, source_triplets)
        has_mm = any(c.match_type == "predicate_mismatch" for c in comps)
        has_mod_mm = any(c.match_type == "modifier_mismatch" for c in comps)
        has_fab = any(c.match_type == "no_match" for c in comps)
        total_mm += sum(1 for c in comps if c.match_type == "predicate_mismatch")
        total_mod_mm += sum(1 for c in comps if c.match_type == "modifier_mismatch")
        total_fab += sum(1 for c in comps if c.match_type == "no_match")
        sent_results.append(SentenceResult(
            sentence=sent, sentence_idx=i, response_triplets=rt,
            comparisons=comps, has_mismatch=(has_mm or has_mod_mm),
            has_fabrication=has_fab,
        ))

    has_issue = total_mm > 0 or total_mod_mm > 0 or total_fab > 0
    decision = "fail" if has_issue else "pass"
    correct = (decision == gt)
    return CaseResult(
        case_id=case.case_id, ground_truth=gt,
        gate1_ratio=ratio, gate1_min_sim=min_sim,
        source_triplets_count=len(source_triplets),
        response_triplets_count=total_rt,
        predicate_mismatches=total_mm, modifier_mismatches=total_mod_mm,
        fabrications=total_fab,
        graphrag_decision=decision, correct=correct,
        sentences=sent_results,
    )


def print_report(fn_results, tn_results):
    fn_caught = sum(1 for r in fn_results if r.graphrag_decision == "fail")
    fn_total = len(fn_results)
    tn_correct = sum(1 for r in tn_results if r.graphrag_decision == "pass")
    tn_total = len(tn_results)
    tn_fp = tn_total - tn_correct
    all_results = fn_results + tn_results

    print(f"\n{'='*70}")
    print("EXPERIMENT 11: SCIENCE GATE — GRAPHRAG (spaCy SVO)")
    print(f"{'='*70}")
    print("\nFALSE NEGATIVES (gt=fail, should catch):")
    print(f"  Tested: {fn_total}")
    print(f"  Caught: {fn_caught}/{fn_total}")
    print("\nTRUE NEGATIVES (gt=pass, should NOT flag):")
    print(f"  Tested: {tn_total}")
    print(f"  Correctly passed: {tn_correct}/{tn_total}")
    print(f"  Wrongly flagged (FP): {tn_fp}/{tn_total}")

    total = fn_total + tn_total
    total_correct = fn_caught + tn_correct
    print(f"\nOVERALL ACCURACY: {total_correct}/{total} ({total_correct/max(1,total)*100:.0f}%)")
    print(f"  Detection rate (recall): {fn_caught}/{fn_total} ({fn_caught/max(1,fn_total)*100:.0f}%)")
    print(f"  False positive rate: {tn_fp}/{tn_total} ({tn_fp/max(1,tn_total)*100:.0f}%)")
    pass_criteria = fn_caught >= fn_total * 0.7 and tn_fp <= tn_total * 0.3
    print("  Pass criteria: catch >= 70% AND FP <= 30%")
    print(f"  Result: {'PASS' if pass_criteria else 'FAIL'}")

    print("\nTRIPLET STATISTICS")
    print(f"  Source triplets: {sum(r.source_triplets_count for r in all_results)}")
    print(f"  Response triplets: {sum(r.response_triplets_count for r in all_results)}")
    print(f"  Predicate mismatches: {sum(r.predicate_mismatches for r in all_results)}")
    print(f"  Modifier mismatches: {sum(r.modifier_mismatches for r in all_results)}")
    print(f"  Fabrications (no match): {sum(r.fabrications for r in all_results)}")
    print(f"{'='*70}")

    for label, results in [("FALSE NEGATIVES (gt=fail)", fn_results), ("TRUE NEGATIVES (gt=pass)", tn_results)]:
        print(f"\n--- {label} ---")
        for r in results:
            if r.ground_truth == "fail":
                marker = "CAUGHT" if r.graphrag_decision == "fail" else "MISSED"
            else:
                marker = "OK" if r.graphrag_decision == "pass" else "FALSE POSITIVE"
            print(f"\n  {r.case_id} [{marker}] gt={r.ground_truth} decision={r.graphrag_decision}")
            print(f"    Source triplets: {r.source_triplets_count}, Response triplets: {r.response_triplets_count}")
            print(f"    Pred mismatch: {r.predicate_mismatches}, Mod mismatch: {r.modifier_mismatches}, Fab: {r.fabrications}")
            for s in r.sentences:
                if not s.response_triplets:
                    print(f"    [{s.sentence_idx+1}] No triplets extracted")
                    print(f"        {s.sentence[:120]}")
                    continue
                flag = ""
                if s.has_mismatch:
                    flag = " <<<< MISMATCH"
                elif s.has_fabrication:
                    flag = " <<<< NO_MATCH"
                print(f"    [{s.sentence_idx+1}]{flag}")
                print(f"        {s.sentence[:120]}")
                for c in s.comparisons:
                    print(f"        [{c.match_type.upper()}] {c.response_triplet}")
                    if c.best_match:
                        print(f"          vs source: {c.best_match}")
                    print(f"          {c.detail}")


def main():
    parser = argparse.ArgumentParser(description="Experiment 11: GraphRAG (spaCy SVO)")
    parser.add_argument("--max-fn", type=int, default=10)
    parser.add_argument("--max-tn", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Finding test cases (FN + TN from Gate 1 PASS)...")
    fn_cases, tn_cases = find_test_cases(
        max_cases=args.max_cases, max_fn=args.max_fn, max_tn=args.max_tn,
    )
    print(f"Found {len(fn_cases)} false negatives + {len(tn_cases)} true negatives")

    if not fn_cases and not tn_cases:
        print("No test cases found.")
        return

    nlp = load_spacy_model()

    fn_results, tn_results = [], []
    for idx, (case, src, resp, ratio, min_sim) in enumerate(fn_cases):
        print(f"\nFN {idx+1}/{len(fn_cases)}: {case.case_id}")
        start = time.time()
        r = run_graphrag_on_case(case, src, resp, ratio, min_sim, "fail", nlp)
        elapsed = time.time() - start
        marker = "CAUGHT" if r.graphrag_decision == "fail" else "MISSED"
        print(f"  {r.graphrag_decision} ({elapsed:.1f}s) {marker} "
              f"(src={r.source_triplets_count} resp={r.response_triplets_count} "
              f"mm={r.predicate_mismatches} mod={r.modifier_mismatches} fab={r.fabrications})")
        fn_results.append(r)

    for idx, (case, src, resp, ratio, min_sim) in enumerate(tn_cases):
        print(f"\nTN {idx+1}/{len(tn_cases)}: {case.case_id}")
        start = time.time()
        r = run_graphrag_on_case(case, src, resp, ratio, min_sim, "pass", nlp)
        elapsed = time.time() - start
        marker = "OK" if r.graphrag_decision == "pass" else "FALSE POSITIVE"
        print(f"  {r.graphrag_decision} ({elapsed:.1f}s) {marker} "
              f"(src={r.source_triplets_count} resp={r.response_triplets_count} "
              f"mm={r.predicate_mismatches} mod={r.modifier_mismatches} fab={r.fabrications})")
        tn_results.append(r)

    print_report(fn_results, tn_results)

    fn_caught = sum(1 for r in fn_results if r.graphrag_decision == "fail")
    tn_fp = sum(1 for r in tn_results if r.graphrag_decision == "fail")
    save_data = {
        "experiment": "Experiment 11: GraphRAG Science Gate (spaCy SVO)",
        "model": SPACY_MODEL,
        "fn_tested": len(fn_results), "fn_caught": fn_caught,
        "tn_tested": len(tn_results), "tn_false_positives": tn_fp,
        "detection_rate": round(fn_caught / max(1, len(fn_results)), 3),
        "false_positive_rate": round(tn_fp / max(1, len(tn_results)), 3),
        "cases": [
            {"case_id": r.case_id, "ground_truth": r.ground_truth,
             "correct": r.correct, "decision": r.graphrag_decision,
             "source_triplets": r.source_triplets_count,
             "response_triplets": r.response_triplets_count,
             "pred_mismatches": r.predicate_mismatches,
             "mod_mismatches": r.modifier_mismatches,
             "fabrications": r.fabrications}
            for r in fn_results + tn_results],
    }
    save_path = output_dir / "graphrag_science_gate_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")

if __name__ == "__main__":
    main()
