"""
Experiment 30 — Stage 2: Compare Response Sentences Against Fact Tables

Loads pre-extracted fact tables from Stage 1.
Runs deterministic comparison of each response sentence against the fact table.
No Gemini calls. No ML models. Pure comparison logic.

If comparison logic needs tuning, rerun only this stage.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _split_sentences

# === Extraction helpers (regex, no ML) ===

_NUM_PATTERN = re.compile(r"\b(\d[\d,]*\.?\d*)\b")
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "he",
    "she",
    "they",
    "his",
    "her",
    "him",
    "their",
    "them",
    "we",
    "our",
    "you",
    "your",
    "who",
    "whom",
    "which",
    "what",
    "and",
    "but",
    "or",
    "said",
    "says",
    "also",
    "just",
    "very",
    "too",
    "so",
    "than",
    "then",
    "now",
    "here",
    "there",
    "after",
    "before",
    "about",
    "into",
    "through",
    "during",
    "between",
    "under",
    "above",
}
_RELATIONSHIP_WORDS = {
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
    "colleague",
    "friend",
    "fiancee",
    "fiance",
    "widow",
    "ex-wife",
    "ex-husband",
}


def extract_sentence_entities(sentence: str) -> list[str]:
    ents = []
    for m in _ENTITY_PATTERN.finditer(sentence):
        ent = m.group(0)
        if ent.lower() not in _STOPWORDS and len(ent) > 2:
            ents.append(ent)
    return ents


def extract_sentence_numbers(sentence: str) -> list[str]:
    return [n.replace(",", "") for n in _NUM_PATTERN.findall(sentence)]


# === Fact table helpers ===


def build_entity_lookup(ft: dict) -> dict[str, dict]:
    """entity_name_lower -> entity info, including aliases."""
    lookup = {}
    for ent in ft.get("entities", []):
        name = ent.get("name", "")
        if name:
            lookup[name.lower()] = ent
            for alias in ent.get("aliases", []):
                if alias:
                    lookup[alias.lower()] = ent
    return lookup


def entity_in_lookup(name: str, lookup: dict) -> tuple[bool, str | None]:
    """Check if entity name is in lookup (exact, substring, or partial word match)."""
    name_lower = name.lower()

    # Exact match
    if name_lower in lookup:
        return True, name_lower

    # Substring: "Copeland" in "Chris Copeland"
    for key in lookup:
        if name_lower in key or key in name_lower:
            return True, key

    # Partial word match: any word of name in any key
    name_parts = [p.lower() for p in name.split() if len(p) > 2]
    for key in lookup:
        if any(part in key for part in name_parts):
            return True, key

    return False, None


# === Comparison engine ===


def compare_sentence(sentence: str, fact_table: dict) -> dict:
    """Compare one response sentence against the fact table."""
    result = {
        "checks_performed": 0,
        "checks_passed": 0,
        "checks_failed": 0,
        "mismatches": [],
        "matches": [],
        "unknown": [],
        "verdict": "unknown",
    }

    sent_entities = extract_sentence_entities(sentence)
    sent_numbers = extract_sentence_numbers(sentence)
    sent_lower = sentence.lower()

    entity_lookup = build_entity_lookup(fact_table)
    num_facts = fact_table.get("numerical_facts", [])
    events = fact_table.get("events", [])
    relationships = fact_table.get("relationships", [])
    negations = fact_table.get("negations", [])

    # --- C1: Entity existence ---
    for ent in sent_entities:
        result["checks_performed"] += 1
        found, matched_key = entity_in_lookup(ent, entity_lookup)
        if found:
            result["checks_passed"] += 1
            result["matches"].append(f"Entity '{ent}' found as '{matched_key}'")
        else:
            result["checks_failed"] += 1
            result["mismatches"].append(f"Entity '{ent}' NOT in fact table")

    # --- C2: Number verification ---
    for num in sent_numbers:
        result["checks_performed"] += 1
        num_found = False

        # Check numerical_facts
        for nf in num_facts:
            nf_num = str(nf.get("number", "")).replace(",", "").replace(".0", "")
            if nf_num == num:
                num_found = True
                nf_entity = str(nf.get("entity", "")).lower()
                nf_describes = str(nf.get("describes", "")).lower()
                # Verify entity context matches
                if nf_entity and any(
                    p in sent_lower for p in nf_entity.split() if len(p) > 2
                ):
                    result["checks_passed"] += 1
                    result["matches"].append(
                        f"Number '{num}' matches for '{nf_entity}' ({nf_describes})"
                    )
                else:
                    result["checks_passed"] += 1
                    result["matches"].append(
                        f"Number '{num}' in fact table (context: {nf_describes[:40]})"
                    )
                break

        if not num_found:
            # Check in events and temporal facts
            all_facts_str = json.dumps(
                events + fact_table.get("temporal_facts", [])
            ).lower()
            if num in all_facts_str:
                num_found = True
                result["checks_passed"] += 1
                result["matches"].append(f"Number '{num}' found in events/temporal")

        if not num_found:
            result["checks_failed"] += 1
            result["mismatches"].append(f"Number '{num}' NOT in fact table")

    # --- C3: Relationship verification ---
    for rel_word in _RELATIONSHIP_WORDS:
        if rel_word in sent_lower:
            result["checks_performed"] += 1

            # Check relationships list
            rel_found = False
            all_rels_str = json.dumps(relationships).lower()
            if rel_word in all_rels_str:
                rel_found = True
                result["checks_passed"] += 1
                result["matches"].append(f"Relationship '{rel_word}' confirmed")
            else:
                # Check entity attributes
                for ent_info in entity_lookup.values():
                    attrs_str = json.dumps(ent_info.get("attributes", {})).lower()
                    if rel_word in attrs_str:
                        rel_found = True
                        result["checks_passed"] += 1
                        result["matches"].append(
                            f"Relationship '{rel_word}' in entity attributes"
                        )
                        break

            if not rel_found:
                # Check if a DIFFERENT relationship is stated
                different_rel = None
                for alt_word in _RELATIONSHIP_WORDS:
                    if alt_word != rel_word and alt_word in all_rels_str:
                        # Check if it involves the same entities
                        for ent in sent_entities:
                            ent_lower = ent.lower()
                            for rel in relationships:
                                rel_str = json.dumps(rel).lower()
                                if ent_lower in rel_str and alt_word in rel_str:
                                    different_rel = alt_word
                                    break
                            if different_rel:
                                break
                    if different_rel:
                        break

                if different_rel:
                    result["checks_failed"] += 1
                    result["mismatches"].append(
                        f"Relationship '{rel_word}' mismatch — fact table says '{different_rel}'"
                    )
                else:
                    result["unknown"].append(
                        f"Relationship '{rel_word}' not verifiable"
                    )

    # --- C4: Negation/contradiction ---
    for neg in negations:
        neg_str = json.dumps(neg).lower() if isinstance(neg, dict) else str(neg).lower()
        neg_keywords = [
            w for w in re.findall(r"\w+", neg_str) if len(w) > 3 and w not in _STOPWORDS
        ]
        overlap = sum(1 for kw in neg_keywords if kw in sent_lower)
        if overlap >= 3:
            result["checks_performed"] += 1
            has_neg = any(
                w in sent_lower.split()
                for w in {
                    "not",
                    "no",
                    "never",
                    "neither",
                    "nor",
                    "n't",
                    "don't",
                    "doesn't",
                    "didn't",
                    "won't",
                    "wouldn't",
                    "can't",
                    "couldn't",
                    "isn't",
                    "aren't",
                    "wasn't",
                    "weren't",
                    "hasn't",
                    "haven't",
                    "hadn't",
                }
            )
            neg_in_source = any(
                w in neg_str for w in ["not", "no", "never", "denied", "false"]
            )

            if has_neg == neg_in_source:
                result["checks_passed"] += 1
                result["matches"].append("Negation alignment with source")
            else:
                result["checks_failed"] += 1
                result["mismatches"].append(
                    f"Possible negation contradiction: {neg_str[:60]}"
                )

    # --- C5: Action/event verification ---
    # Check if key verbs in sentence match events in fact table
    action_verbs = {
        "arrested",
        "charged",
        "stabbed",
        "killed",
        "convicted",
        "sentenced",
        "pleaded",
        "settled",
        "filed",
        "recalled",
        "hospitalized",
        "died",
        "married",
        "divorced",
        "released",
        "resigned",
        "fired",
        "hired",
        "announced",
        "confirmed",
        "denied",
        "apologized",
        "sued",
    }
    for verb in action_verbs:
        if verb in sent_lower:
            result["checks_performed"] += 1
            events_str = json.dumps(events).lower()
            if verb in events_str:
                # Check if the entity doing the action matches
                entity_match = False
                for ent in sent_entities:
                    if ent.lower() in events_str:
                        entity_match = True
                        break
                if entity_match:
                    result["checks_passed"] += 1
                    result["matches"].append(
                        f"Action '{verb}' with matching entity confirmed"
                    )
                else:
                    result["checks_passed"] += 1
                    result["matches"].append(
                        f"Action '{verb}' exists in events (entity unclear)"
                    )
            else:
                # Verb not in any event — might be fabricated action
                result["unknown"].append(f"Action '{verb}' not in events")

    # --- Verdict ---
    if result["checks_performed"] == 0:
        result["verdict"] = "unknown"
    elif result["checks_failed"] > 0:
        result["verdict"] = "flagged"
    elif result["checks_passed"] >= 2:
        result["verdict"] = "grounded"
    else:
        result["verdict"] = "unknown"

    return result


# === Labelling ===


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
    source_len: int
    checks_performed: int
    checks_passed: int
    checks_failed: int
    mismatches: str
    matches: str
    l2_verdict: str


def main():
    fact_tables_path = "experiments/exp30_fact_tables.json"

    if not os.path.exists(fact_tables_path):
        print(f"ERROR: {fact_tables_path} not found.")
        print(
            "Run Stage 1 first: poetry run python experiments/exp30_stage1_extract.py"
        )
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 30 — STAGE 2: Compare Sentences Against Fact Tables")
    print("=" * 70)
    print(f"  Loading fact tables from: {fact_tables_path}")
    print("  No Gemini calls. No ML models. Pure deterministic comparison.")
    print("=" * 70)

    with open(fact_tables_path) as f:
        fact_tables = json.load(f)

    n_valid = sum(1 for v in fact_tables.values() if v.get("fact_table"))
    n_failed = sum(1 for v in fact_tables.values() if not v.get("fact_table"))
    print(f"  Fact tables loaded: {n_valid} valid, {n_failed} failed")

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"  Cases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results: list[SentenceResult] = []
    verdict_counts = {"grounded": 0, "flagged": 0, "unknown": 0}
    safety_violations = []
    hallucination_catches = []
    t0 = time.time()

    for ci, case in enumerate(cases):
        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )
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
                        source_len=len(source),
                        checks_performed=0,
                        checks_passed=0,
                        checks_failed=0,
                        mismatches="no_fact_table",
                        matches="",
                        l2_verdict="unknown",
                    )
                )
                verdict_counts["unknown"] += 1
            continue

        for sl in labeled:
            sent = sl["sentence"]
            comp = compare_sentence(sent, ft)

            verdict_counts[comp["verdict"]] += 1

            if comp["verdict"] == "grounded" and sl["label"] == "hallucinated":
                safety_violations.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "matches": comp["matches"][:3],
                    }
                )

            if comp["verdict"] == "flagged" and sl["label"] == "hallucinated":
                hallucination_catches.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "mismatches": comp["mismatches"][:3],
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
                    source_len=len(source),
                    checks_performed=comp["checks_performed"],
                    checks_passed=comp["checks_passed"],
                    checks_failed=comp["checks_failed"],
                    mismatches="; ".join(comp["mismatches"][:5]),
                    matches="; ".join(comp["matches"][:5]),
                    l2_verdict=comp["verdict"],
                )
            )

        if (ci + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(
                f"\n--- {ci+1}/{len(cases)} cases | {len(all_results)} sents | {elapsed*1000:.0f}ms ---"
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
    print(
        f"FINAL — {len(all_results)} sentences, {elapsed*1000:.0f}ms (no Gemini calls)"
    )
    print(f"{'='*70}")

    print("\n  L2 VERDICTS")
    print(
        f"    Grounded:  {verdict_counts['grounded']:3d} ({verdict_counts['grounded']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Flagged:   {verdict_counts['flagged']:3d} ({verdict_counts['flagged']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Unknown:   {verdict_counts['unknown']:3d} ({verdict_counts['unknown']/len(all_results)*100:.1f}%)"
    )

    # Safety
    print(f"\n{'='*70}")
    print("SAFETY: Hallucinated sentences incorrectly cleared")
    print(f"{'='*70}")
    if safety_violations:
        print(f"  *** {len(safety_violations)} SAFETY VIOLATIONS ***")
        for sv in safety_violations:
            print(f"    {sv['case_id']} S{sv['sentence_idx']} GT={sv['gt_type']}")
            print(f"      Sentence: {sv['sentence']}")
            print(f"      Matches: {sv['matches']}")
    else:
        print("  *** 0 SAFETY VIOLATIONS ***")
        print("  L2 never cleared a hallucinated sentence.")

    # Catches
    print(f"\n{'='*70}")
    print("HALLUCINATION CATCHES")
    print(f"{'='*70}")
    print(
        f"  {len(hallucination_catches)} / 16 hallucinations caught ({len(hallucination_catches)/16*100:.0f}% recall)"
    )
    for hc in hallucination_catches:
        print(f"    {hc['case_id']} S{hc['sentence_idx']} GT={hc['gt_type']}")
        print(f"      Sentence: {hc['sentence']}")
        print(f"      Mismatches: {hc['mismatches']}")

    # Missed hallucinations
    caught_ids = {(hc["case_id"], hc["sentence_idx"]) for hc in hallucination_catches}
    print(f"\n  Missed hallucinations ({16 - len(hallucination_catches)}):")
    for r in all_results:
        if (
            r.gt_label == "hallucinated"
            and (r.case_id, r.sentence_idx) not in caught_ids
        ):
            print(
                f"    {r.case_id} S{r.sentence_idx} GT={r.gt_type} verdict={r.l2_verdict}"
            )
            print(f"      Sentence: {r.sentence[:100]}")
            if r.matches:
                print(f"      Matches: {r.matches[:100]}")

    # Verdict vs ground truth
    print(f"\n{'='*70}")
    print("VERDICT vs GROUND TRUTH")
    print(f"{'='*70}")
    for verdict in ["grounded", "flagged", "unknown"]:
        vr = [r for r in all_results if r.l2_verdict == verdict]
        v_clean = sum(1 for r in vr if r.gt_label == "clean")
        v_hall = sum(1 for r in vr if r.gt_label == "hallucinated")
        print(
            f"  {verdict:12s}: {len(vr):3d} total ({v_clean} clean, {v_hall} hallucinated)"
        )

    # Comparison with L1
    print(f"\n{'='*70}")
    print("COMPARISON: L1 vs L2")
    print(f"{'='*70}")
    print("  L1 (Exp 29b A1 only): cleared 21, caught 4/16 (B flags)")
    print(
        f"  L2 (Exp 30):          cleared {verdict_counts['grounded']}, caught {len(hallucination_catches)}/16"
    )
    print(f"  L2 safety violations: {len(safety_violations)}")

    # Checks distribution
    print(f"\n{'='*70}")
    print("CHECKS DISTRIBUTION")
    print(f"{'='*70}")
    check_counts = [r.checks_performed for r in all_results]
    print(f"  Avg checks per sentence: {sum(check_counts)/len(check_counts):.1f}")
    for bucket, lo, hi in [
        ("0 checks", 0, 0),
        ("1-2", 1, 2),
        ("3-5", 3, 5),
        ("6+", 6, 100),
    ]:
        cnt = sum(1 for c in check_counts if lo <= c <= hi)
        print(f"    {bucket}: {cnt} sentences")

    # Save
    output = {
        "elapsed_ms": round(elapsed * 1000, 1),
        "verdict_counts": verdict_counts,
        "safety_violations": len(safety_violations),
        "hallucination_catches": len(hallucination_catches),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp30_l2_comparison_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp30_l2_comparison_results.json")


if __name__ == "__main__":
    main()
