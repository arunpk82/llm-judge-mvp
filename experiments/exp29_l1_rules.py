"""
Experiment 29: Enhanced L1 Rules Layer

Measures how many sentences a proper rules-based L1 can confidently clear
as grounded, and how many hallucinations the flag checks catch.

All checks are deterministic, zero-cost, sub-millisecond.
No ML models. No API calls. Pure rules.

Same 50 RAGTruth cases, 283 sentences.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _split_sentences

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
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "also",
    "about",
    "up",
    "out",
    "if",
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
    "because",
    "while",
    "although",
    "however",
    "since",
    "until",
    "unless",
    "said",
    "says",
    "according",
    "told",
    "stated",
    "reported",
    "added",
    "noted",
}
_NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "nobody",
    "nothing",
    "nowhere",
    "n't",
    "don't",
    "doesn't",
    "didn't",
    "won't",
    "wouldn't",
    "can't",
    "couldn't",
    "shouldn't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "hasn't",
    "haven't",
    "hadn't",
}
_STRONG_QUALIFIERS = {
    "all": "some",
    "every": "some",
    "always": "sometimes",
    "never": "rarely",
    "none": "few",
    "only": "one of",
    "first": "one of the first",
    "largest": "one of the largest",
    "best": "one of the best",
    "worst": "one of the worst",
}


def extract_numbers(text):
    raw = _NUM_PATTERN.findall(text)
    return [n.replace(",", "") for n in raw]


def extract_entities(text):
    ents = []
    for m in _ENTITY_PATTERN.finditer(text):
        if m.start() == 0:
            continue
        if m.start() >= 2 and text[m.start() - 2] == ".":
            continue
        ent = m.group(0)
        if ent.lower() in _STOPWORDS:
            continue
        if len(ent) > 2:
            ents.append(ent)
    return ents


def extract_content_words(text):
    words = set(re.findall(r"\w+", text.lower()))
    return {w for w in words if w not in _STOPWORDS and len(w) > 2 and not w.isdigit()}


# === CATEGORY A: CONFIRM GROUNDED ===


def check_a1_string_match(sentence, source_sentences, source_full):
    norm_sent = re.sub(r"\s+", " ", sentence.lower().strip())
    norm_source = re.sub(r"\s+", " ", source_full.lower().strip())
    if norm_sent in norm_source:
        return True
    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return False
    for src_sent in source_sentences:
        norm_src = re.sub(r"\s+", " ", src_sent.lower().strip())
        if SequenceMatcher(None, norm_sent, norm_src).ratio() > 0.85:
            return True
        src_tokens = set(re.findall(r"\w+", src_sent.lower()))
        if src_tokens:
            jaccard = len(sent_tokens & src_tokens) / len(sent_tokens | src_tokens)
            if jaccard > 0.80:
                return True
    return False


def check_a2_entity_coverage(sentence, source_full):
    sent_entities = extract_entities(sentence)
    if not sent_entities:
        return False, [], []
    source_lower = source_full.lower()
    found, missing = [], []
    for ent in sent_entities:
        (found if ent.lower() in source_lower else missing).append(ent)
    return (len(missing) == 0 and len(found) >= 2), found, missing


def check_a3_number_verification(sentence, source_full):
    sent_nums = extract_numbers(sentence)
    if not sent_nums:
        return False, [], []
    source_nums = set(extract_numbers(source_full))
    found, missing = [], []
    for n in sent_nums:
        (found if n in source_nums else missing).append(n)
    return (len(missing) == 0 and len(found) >= 1), found, missing


def check_a4_token_coverage(sentence, source_full, threshold=0.90):
    sent_words = extract_content_words(sentence)
    if len(sent_words) < 3:
        return False, 0.0
    source_words = extract_content_words(source_full)
    covered = sent_words & source_words
    coverage = len(covered) / len(sent_words)
    return coverage >= threshold, round(coverage, 3)


# === CATEGORY B: FLAG SUSPICIOUS ===


def check_b1_number_mismatch(sentence, source_full):
    sent_nums = extract_numbers(sentence)
    if not sent_nums:
        return False, ""
    source_nums = set(extract_numbers(source_full))
    for n in sent_nums:
        if n not in source_nums:
            return True, f"Number '{n}' not in source"
    return False, ""


def check_b2_entity_not_in_source(sentence, source_full):
    sent_entities = extract_entities(sentence)
    source_lower = source_full.lower()
    for ent in sent_entities:
        if ent.lower() not in source_lower:
            return True, f"Entity '{ent}' not in source"
    return False, ""


def check_b3_negation_flip(sentence, source_sentences):
    sent_lower = sentence.lower()
    sent_has_neg = any(neg in sent_lower.split() for neg in _NEGATION_WORDS)
    sent_content = extract_content_words(sentence)
    for src_sent in source_sentences:
        src_lower = src_sent.lower()
        src_content = extract_content_words(src_sent)
        if len(sent_content & src_content) < 3:
            continue
        src_has_neg = any(neg in src_lower.split() for neg in _NEGATION_WORDS)
        if sent_has_neg != src_has_neg:
            return (
                True,
                f"Negation mismatch: sent_neg={sent_has_neg}, src_neg={src_has_neg}",
            )
    return False, ""


def check_b4_qualifier_mismatch(sentence, source_full):
    sent_lower = sentence.lower()
    source_lower = source_full.lower()
    for strong, weak in _STRONG_QUALIFIERS.items():
        if f" {strong} " in f" {sent_lower} ":
            if f" {strong} " not in f" {source_lower} ":
                return True, f"Strong qualifier '{strong}' not in source"
    return False, ""


# === LABELLING ===


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
    a1_string_match: bool
    a2_entity_coverage: bool
    a2_found: str
    a2_missing: str
    a3_number_match: bool
    a3_found: str
    a3_missing: str
    a4_token_coverage: bool
    a4_coverage_pct: float
    any_a_pass: bool
    b1_number_mismatch: bool
    b1_detail: str
    b2_entity_missing: bool
    b2_detail: str
    b3_negation_flip: bool
    b3_detail: str
    b4_qualifier_mismatch: bool
    b4_detail: str
    any_b_flag: bool
    l1_verdict: str


def main():
    print("=" * 70)
    print("EXPERIMENT 29: Enhanced L1 Rules Layer")
    print("=" * 70)
    print("  Category A: Confirm grounded (clear sentence, skip downstream)")
    print("    A1: String match    A2: Entity coverage")
    print("    A3: Number match    A4: Token coverage (90%+)")
    print("  Category B: Flag suspicious (pass evidence downstream)")
    print("    B1: Number mismatch  B2: Entity not in source")
    print("    B3: Negation flip    B4: Qualifier mismatch")
    print("  Same 50 cases, 283 sentences.")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"\nCases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results = []
    t0 = time.time()
    a_counts = {"a1": 0, "a2": 0, "a3": 0, "a4": 0, "any_a": 0}
    b_counts = {"b1": 0, "b2": 0, "b3": 0, "b4": 0, "any_b": 0}
    verdict_counts = {"grounded": 0, "flagged": 0, "pass_downstream": 0}
    safety_violations = []
    b_catches = []

    for ci, case in enumerate(cases):
        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )
        response = case.request.candidate_answer or ""
        gt = case.ground_truth
        labeled = label_sentences(response, gt.span_annotations, gt.response_level)
        source_sents = _split_sentences(source)

        for sl in labeled:
            sent = sl["sentence"]
            a1 = check_a1_string_match(sent, source_sents, source)
            a2_pass, a2_found, a2_missing = check_a2_entity_coverage(sent, source)
            a3_pass, a3_found, a3_missing = check_a3_number_verification(sent, source)
            a4_pass, a4_pct = check_a4_token_coverage(sent, source)
            any_a = a1 or a2_pass or a3_pass or a4_pass

            if a1:
                a_counts["a1"] += 1
            if a2_pass:
                a_counts["a2"] += 1
            if a3_pass:
                a_counts["a3"] += 1
            if a4_pass:
                a_counts["a4"] += 1
            if any_a:
                a_counts["any_a"] += 1

            b1_flag, b1_det = check_b1_number_mismatch(sent, source)
            b2_flag, b2_det = check_b2_entity_not_in_source(sent, source)
            b3_flag, b3_det = check_b3_negation_flip(sent, source_sents)
            b4_flag, b4_det = check_b4_qualifier_mismatch(sent, source)
            any_b = b1_flag or b2_flag or b3_flag or b4_flag

            if b1_flag:
                b_counts["b1"] += 1
            if b2_flag:
                b_counts["b2"] += 1
            if b3_flag:
                b_counts["b3"] += 1
            if b4_flag:
                b_counts["b4"] += 1
            if any_b:
                b_counts["any_b"] += 1

            if any_a and not any_b:
                verdict = "grounded"
            elif any_b:
                verdict = "flagged"
            else:
                verdict = "pass_downstream"
            verdict_counts[verdict] += 1

            if verdict == "grounded" and sl["label"] == "hallucinated":
                safety_violations.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "gt_type": sl["type"],
                        "a1": a1,
                        "a2": a2_pass,
                        "a3": a3_pass,
                        "a4": a4_pass,
                        "a4_pct": a4_pct,
                    }
                )

            if any_b and sl["label"] == "hallucinated":
                b_catches.append(
                    {
                        "case_id": case.case_id,
                        "sentence_idx": sl["idx"],
                        "sentence": sent[:120],
                        "b1": b1_det,
                        "b2": b2_det,
                        "b3": b3_det,
                        "b4": b4_det,
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
                    a1_string_match=a1,
                    a2_entity_coverage=a2_pass,
                    a2_found=",".join(a2_found[:5]),
                    a2_missing=",".join(a2_missing[:5]),
                    a3_number_match=a3_pass,
                    a3_found=",".join(a3_found[:5]),
                    a3_missing=",".join(a3_missing[:5]),
                    a4_token_coverage=a4_pass,
                    a4_coverage_pct=a4_pct,
                    any_a_pass=any_a,
                    b1_number_mismatch=b1_flag,
                    b1_detail=b1_det,
                    b2_entity_missing=b2_flag,
                    b2_detail=b2_det,
                    b3_negation_flip=b3_flag,
                    b3_detail=b3_det,
                    b4_qualifier_mismatch=b4_flag,
                    b4_detail=b4_det,
                    any_b_flag=any_b,
                    l1_verdict=verdict,
                )
            )

    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"RESULTS — {len(all_results)} sentences, {elapsed*1000:.0f}ms total")
    print(f"{'='*70}")

    print("\n  CATEGORY A: Confirm Grounded")
    print(
        f"    A1 String match:     {a_counts['a1']:3d} / {len(all_results)} ({a_counts['a1']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    A2 Entity coverage:  {a_counts['a2']:3d} / {len(all_results)} ({a_counts['a2']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    A3 Number match:     {a_counts['a3']:3d} / {len(all_results)} ({a_counts['a3']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    A4 Token coverage:   {a_counts['a4']:3d} / {len(all_results)} ({a_counts['a4']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    ANY A pass:          {a_counts['any_a']:3d} / {len(all_results)} ({a_counts['any_a']/len(all_results)*100:.1f}%)"
    )

    print("\n  CATEGORY B: Flag Suspicious")
    print(f"    B1 Number mismatch:  {b_counts['b1']:3d}")
    print(f"    B2 Entity missing:   {b_counts['b2']:3d}")
    print(f"    B3 Negation flip:    {b_counts['b3']:3d}")
    print(f"    B4 Qualifier issue:  {b_counts['b4']:3d}")
    print(f"    ANY B flag:          {b_counts['any_b']:3d}")

    print("\n  L1 VERDICTS")
    print(
        f"    Grounded (skip downstream): {verdict_counts['grounded']:3d} ({verdict_counts['grounded']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Flagged (pass to L3/L4):    {verdict_counts['flagged']:3d} ({verdict_counts['flagged']/len(all_results)*100:.1f}%)"
    )
    print(
        f"    Pass downstream (no signal):{verdict_counts['pass_downstream']:3d} ({verdict_counts['pass_downstream']/len(all_results)*100:.1f}%)"
    )

    print(f"\n{'='*70}")
    print("SAFETY: Hallucinated sentences incorrectly cleared by L1")
    print(f"{'='*70}")
    if safety_violations:
        print(f"  *** {len(safety_violations)} SAFETY VIOLATIONS ***")
        for sv in safety_violations:
            print(f"    {sv['case_id']} S{sv['sentence_idx']} GT={sv['gt_type']}")
            print(f"      Sentence: {sv['sentence']}")
            print(
                f"      Cleared by: A1={sv['a1']} A2={sv['a2']} A3={sv['a3']} A4={sv['a4']} (cov={sv['a4_pct']:.3f})"
            )
    else:
        print("  No safety violations. L1 never cleared a hallucinated sentence.")

    print(f"\n{'='*70}")
    print("B FLAG CATCHES: Hallucinations caught by flag checks")
    print(f"{'='*70}")
    if b_catches:
        print(f"  {len(b_catches)} hallucinations flagged:")
        for bc in b_catches:
            flags = [v for k, v in bc.items() if k.startswith("b") and v]
            print(f"    {bc['case_id']} S{bc['sentence_idx']}")
            print(f"      Sentence: {bc['sentence']}")
            print(f"      Flags: {'; '.join(flags)}")
    else:
        print("  No hallucinations caught by B flags.")

    print(f"\n{'='*70}")
    print("L1 IMPACT ON DOWNSTREAM")
    print(f"{'='*70}")
    cleared = [r for r in all_results if r.l1_verdict == "grounded"]
    remaining = [r for r in all_results if r.l1_verdict != "grounded"]
    cleared_clean = sum(1 for r in cleared if r.gt_label == "clean")
    cleared_hall = sum(1 for r in cleared if r.gt_label == "hallucinated")

    print(
        f"  Cleared at L1: {len(cleared)} sentences ({cleared_clean} clean, {cleared_hall} hallucinated)"
    )
    print(f"  Remaining for L2+: {len(remaining)} sentences")
    print(
        f"  Reduction: {len(cleared)/len(all_results)*100:.1f}% of sentences handled at L1"
    )
    if cleared_hall > 0:
        print(f"\n  *** WARNING: {cleared_hall} hallucinated sentences cleared! ***")
    else:
        print("\n  L1 precision: 100% (no hallucinated sentences cleared)")

    # Check overlap
    print(f"\n{'='*70}")
    print("CHECK OVERLAP ANALYSIS")
    print(f"{'='*70}")
    a1_set = {(r.case_id, r.sentence_idx) for r in all_results if r.a1_string_match}
    a2_set = {(r.case_id, r.sentence_idx) for r in all_results if r.a2_entity_coverage}
    a3_set = {(r.case_id, r.sentence_idx) for r in all_results if r.a3_number_match}
    a4_set = {(r.case_id, r.sentence_idx) for r in all_results if r.a4_token_coverage}
    print(f"  A1 only: {len(a1_set - a2_set - a3_set - a4_set)}")
    print(f"  A2 only: {len(a2_set - a1_set - a3_set - a4_set)}")
    print(f"  A3 only: {len(a3_set - a1_set - a2_set - a4_set)}")
    print(f"  A4 only: {len(a4_set - a1_set - a2_set - a3_set)}")

    coverages = [r.a4_coverage_pct for r in all_results if r.a4_coverage_pct > 0]
    if coverages:
        print("  A4 coverage distribution:")
        buckets = {"<50%": 0, "50-70%": 0, "70-80%": 0, "80-90%": 0, "90%+": 0}
        for c in coverages:
            if c < 0.50:
                buckets["<50%"] += 1
            elif c < 0.70:
                buckets["50-70%"] += 1
            elif c < 0.80:
                buckets["70-80%"] += 1
            elif c < 0.90:
                buckets["80-90%"] += 1
            else:
                buckets["90%+"] += 1
        for k, v in buckets.items():
            print(f"    {k}: {v} sentences")

    output = {
        "elapsed_ms": round(elapsed * 1000, 1),
        "a_counts": a_counts,
        "b_counts": b_counts,
        "verdict_counts": verdict_counts,
        "safety_violations": len(safety_violations),
        "b_catches": len(b_catches),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp29_l1_rules_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp29_l1_rules_results.json")


if __name__ == "__main__":
    main()
