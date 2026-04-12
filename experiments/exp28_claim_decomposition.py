"""
Experiment 28: Claim Decomposition

Baseline: Exp 26b (chunking, Response F1=0.636)
Change: Decompose multi-claim sentences into atomic claims before MiniCheck.
Sentence grounded ONLY if ALL atomic claims pass (strict aggregation).

Same 50 cases, 283 sentences.
"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _chunk_source,
    _split_sentences,
)

# --- Claim decomposition (regex-based, no LLM) ---

_SPLIT_SEMICOLONS = re.compile(r";\s+")
_SPLIT_CONJUNCTIONS = re.compile(r",\s+(?:and|as well as|while|but)\s+", re.I)
_SPLIT_INCLUDING = re.compile(r",\s+(?:including|such as)\s+", re.I)
_SPLIT_CLAUSES = re.compile(r",\s+(?:which|where|who|that)\s+", re.I)

MIN_CLAIM_LENGTH = 15


def decompose_claims(sentence: str) -> list[str]:
    """Split a sentence into atomic claims. Returns original if no split possible."""
    parts = [sentence]

    # Phase 1: semicolons (strongest signal)
    new = []
    for p in parts:
        segs = _SPLIT_SEMICOLONS.split(p)
        if len(segs) > 1 and all(len(s.strip()) > MIN_CLAIM_LENGTH for s in segs):
            new.extend(s.strip() for s in segs)
        else:
            new.append(p)
    parts = new

    # Phase 2: conjunctions
    new = []
    for p in parts:
        for pattern in [_SPLIT_CONJUNCTIONS, _SPLIT_INCLUDING]:
            segs = pattern.split(p)
            if len(segs) > 1 and all(len(s.strip()) > MIN_CLAIM_LENGTH for s in segs):
                new.extend(s.strip() for s in segs)
                break
        else:
            new.append(p)
    parts = new

    # Phase 3: relative clauses
    new = []
    for p in parts:
        segs = _SPLIT_CLAUSES.split(p)
        if len(segs) > 1 and all(len(s.strip()) > MIN_CLAIM_LENGTH for s in segs):
            new.extend(s.strip() for s in segs)
        else:
            new.append(p)
    parts = new

    # Filter short fragments
    parts = [p for p in parts if len(p.strip()) > MIN_CLAIM_LENGTH]

    if len(parts) <= 1:
        return [sentence]

    return parts


# --- MiniCheck with chunking (same as Exp 26b) ---


def run_minicheck_chunked(claim: str, source_doc: str) -> float:
    """MiniCheck with chunking. Returns 1.0 or 0.0."""
    try:
        import torch

        import llm_judge.calibration.hallucination as hal

        hal._load_minicheck()

        chunks = _chunk_source(source_doc)
        for chunk in chunks:
            prompt = hal._MINICHECK_PROMPT.format(document=chunk, claim=claim)
            inputs = hal._mc_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            with torch.no_grad():
                outputs = hal._mc_model.generate(**inputs, max_new_tokens=5)
            generated = hal._mc_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            if generated == "1":
                return 1.0
        return 0.0
    except Exception as e:
        print(f"    MC error: {str(e)[:80]}")
        return -1.0


# --- Labelling ---


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
    # Baseline (26b: chunking, full sentence)
    baseline_score: float
    # Exp 28 (decomposed claims + chunking)
    decomposed: bool
    num_claims: int
    claims: str  # pipe-separated
    claim_scores: str  # comma-separated
    exp28_score: float  # 1.0 if ALL claims pass, 0.0 if ANY fails


MC_THRESHOLD = 0.5


class LiveStats:
    def __init__(self):
        self.cases = 0
        self.sents = 0
        self.decomposed_count = 0
        self.avg_claims = 0.0
        # Baseline
        self.b_tp = 0
        self.b_fp = 0
        self.b_fn = 0
        self.b_tn = 0
        # Exp 28
        self.e_tp = 0
        self.e_fp = 0
        self.e_fn = 0
        self.e_tn = 0

    def add(self, gt, baseline, exp28):
        for score, counters in [(baseline, "b"), (exp28, "e")]:
            if score < 0:
                continue
            grounded = score >= MC_THRESHOLD
            if gt == "hallucinated" and not grounded:
                setattr(self, f"{counters}_tp", getattr(self, f"{counters}_tp") + 1)
            elif gt == "clean" and not grounded:
                setattr(self, f"{counters}_fp", getattr(self, f"{counters}_fp") + 1)
            elif gt == "hallucinated" and grounded:
                setattr(self, f"{counters}_fn", getattr(self, f"{counters}_fn") + 1)
            else:
                setattr(self, f"{counters}_tn", getattr(self, f"{counters}_tn") + 1)

    def _m(self, tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return p, r, 2 * p * r / max(0.001, p + r)

    def show(self, elapsed, total):
        bp, br, bf = self._m(self.b_tp, self.b_fp, self.b_fn)
        ep, er, ef = self._m(self.e_tp, self.e_fp, self.e_fn)
        fp_d = self.e_fp - self.b_fp
        fn_d = self.e_fn - self.b_fn

        print(
            f"\n--- {self.cases}/{total} cases | {self.sents} sents | {elapsed:.0f}s ---"
        )
        print(
            f"  Decomposed: {self.decomposed_count}/{self.sents} sentences ({self.avg_claims:.1f} avg claims when split)"
        )
        print(
            f"  26b Baseline : TP={self.b_tp:3d} FP={self.b_fp:3d} FN={self.b_fn:3d} TN={self.b_tn:3d} | P={bp:.3f} R={br:.3f} F1={bf:.3f}"
        )
        print(
            f"  28 Decompose : TP={self.e_tp:3d} FP={self.e_fp:3d} FN={self.e_fn:3d} TN={self.e_tn:3d} | P={ep:.3f} R={er:.3f} F1={ef:.3f}"
        )
        print(f"     Delta:      FP {fp_d:+d}   FN {fn_d:+d}")


def resp_metrics(results, field):
    cids = sorted(set(r.case_id for r in results))
    tp = fp = fn = tn = 0
    for cid in cids:
        cr = [r for r in results if r.case_id == cid]
        gt_fail = cr[0].response_level == "fail"
        flagged = any(
            getattr(r, field) >= 0 and getattr(r, field) < MC_THRESHOLD for r in cr
        )
        if gt_fail and flagged:
            tp += 1
        elif not gt_fail and flagged:
            fp += 1
        elif gt_fail and not flagged:
            fn += 1
        else:
            tn += 1
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    f1 = 2 * p * r / max(0.001, p + r)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p": round(p, 3),
        "r": round(r, 3),
        "f1": round(f1, 3),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 28: Claim Decomposition")
    print("=" * 70)
    print("  Baseline: Exp 26b (chunking, full sentence)")
    print("  Exp 28:   Decompose multi-claim sentences, verify each atomic claim")
    print("  Strict aggregation: ALL claims must pass for sentence to be grounded")
    print("  Same 50 cases, 283 sentences. MiniCheck threshold=0.5")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"\nCases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results: list[SentenceResult] = []
    stats = LiveStats()
    total_claims_when_split = 0
    split_count = 0
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

        for sl in labeled:
            sent = sl["sentence"]

            # --- Baseline: Exp 26b (chunking, full sentence) ---
            baseline_score = run_minicheck_chunked(sent, source)

            # --- Exp 28: Decompose + verify each ---
            claims = decompose_claims(sent)
            was_decomposed = len(claims) > 1

            if was_decomposed:
                split_count += 1
                total_claims_when_split += len(claims)
                # Verify each atomic claim
                claim_scores = []
                for claim in claims:
                    cs = run_minicheck_chunked(claim, source)
                    claim_scores.append(cs)

                # Strict: ALL must pass
                if any(s < 0 for s in claim_scores):
                    exp28_score = -1.0  # error
                elif all(s >= MC_THRESHOLD for s in claim_scores):
                    exp28_score = 1.0  # all grounded
                else:
                    exp28_score = 0.0  # at least one failed
            else:
                # Single claim: same as baseline
                claim_scores = [baseline_score]
                exp28_score = baseline_score

            stats.add(sl["label"], baseline_score, exp28_score)
            stats.decomposed_count += 1 if was_decomposed else 0
            stats.avg_claims = total_claims_when_split / max(1, split_count)

            all_results.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    response_level=gt.response_level,
                    source_len=len(source),
                    baseline_score=baseline_score,
                    decomposed=was_decomposed,
                    num_claims=len(claims),
                    claims=" | ".join(c[:80] for c in claims),
                    claim_scores=",".join(f"{s:.0f}" for s in claim_scores),
                    exp28_score=exp28_score,
                )
            )

        stats.cases = ci + 1
        stats.sents = len(all_results)
        if (ci + 1) % 5 == 0:
            stats.show(time.time() - t0, len(cases))

    # === FINAL ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(
        f"FINAL \u2014 {len(all_results)} sentences, {len(cases)} cases, {elapsed:.0f}s"
    )
    print(f"{'='*70}")
    stats.show(elapsed, len(cases))

    # Response-level
    mr_base = resp_metrics(all_results, "baseline_score")
    mr_exp28 = resp_metrics(all_results, "exp28_score")

    print(f"\n{'='*70}")
    print("RESPONSE-LEVEL COMPARISON")
    print(f"{'='*70}")
    print(
        f"  26b Baseline: TP={mr_base['tp']} FP={mr_base['fp']} FN={mr_base['fn']} TN={mr_base['tn']} | P={mr_base['p']:.3f} R={mr_base['r']:.3f} F1={mr_base['f1']:.3f}"
    )
    print(
        f"  28 Decompose: TP={mr_exp28['tp']} FP={mr_exp28['fp']} FN={mr_exp28['fn']} TN={mr_exp28['tn']} | P={mr_exp28['p']:.3f} R={mr_exp28['r']:.3f} F1={mr_exp28['f1']:.3f}"
    )
    print("\n  Published baselines:")
    print("    GPT-4-turbo:    0.634")
    print("    FT Llama-2-13B: 0.787")
    print(f"    *** 26b Baseline: {mr_base['f1']:.3f} ***")
    print(f"    *** 28 Decompose: {mr_exp28['f1']:.3f} ***")

    # What changed
    print(f"\n{'='*70}")
    print("WHAT CHANGED \u2014 sentence-by-sentence")
    print(f"{'='*70}")

    fixed = [
        r
        for r in all_results
        if r.baseline_score < MC_THRESHOLD and r.exp28_score >= MC_THRESHOLD
    ]
    broken = [
        r
        for r in all_results
        if r.baseline_score >= MC_THRESHOLD and r.exp28_score < MC_THRESHOLD
    ]

    print(f"\n  FIXED by decomposition ({len(fixed)} sentences now grounded):")
    for r in fixed:
        t = " [TRUNC]" if r.source_len > 3500 else ""
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} claims={r.num_claims}{t} | {r.sentence[:90]}"
        )
        print(f"      Claims: {r.claims[:120]}")
        print(f"      Scores: {r.claim_scores}")

    print(f"\n  BROKEN by decomposition ({len(broken)} sentences now ungrounded):")
    for r in broken:
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} claims={r.num_claims} | {r.sentence[:90]}"
        )
        print(f"      Claims: {r.claims[:120]}")
        print(f"      Scores: {r.claim_scores}")

    # Decomposition stats
    decomposed = [r for r in all_results if r.decomposed]
    print(f"\n{'='*70}")
    print("DECOMPOSITION STATISTICS")
    print(f"{'='*70}")
    print(f"  Total sentences: {len(all_results)}")
    print(
        f"  Decomposed: {len(decomposed)} ({len(decomposed)/len(all_results)*100:.1f}%)"
    )
    print(f"  Average claims per decomposed sentence: {stats.avg_claims:.1f}")
    print("  Claim count distribution:")
    from collections import Counter

    dist = Counter(r.num_claims for r in decomposed)
    for n, cnt in sorted(dist.items()):
        print(f"    {n} claims: {cnt} sentences")

    # Save
    output = {
        "config": {"mc_threshold": MC_THRESHOLD, "min_claim_length": MIN_CLAIM_LENGTH},
        "elapsed_s": round(elapsed, 1),
        "baseline_response": mr_base,
        "exp28_response": mr_exp28,
        "decomposed_count": len(decomposed),
        "avg_claims": round(stats.avg_claims, 2),
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp28_claim_decomposition_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp28_claim_decomposition_results.json")


if __name__ == "__main__":
    main()
