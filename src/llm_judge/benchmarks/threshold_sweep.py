"""
Experiment 2: Threshold Sweep for Property 1.1 Groundedness.

Purpose: Find the optimal combination of (sentence_similarity_threshold,
grounding_ratio_threshold) that maximises F1 on the CS domain ground truth.

Approach: Compute all sentence-level cosine similarities ONCE, then apply
each threshold combination post-hoc without re-computing embeddings.
This makes the 78-combination sweep fast (~seconds after initial embedding).

Usage:
    poetry run python -m llm_judge.benchmarks.threshold_sweep --max-cases 500
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
OUTPUT_DIR = Path("experiments")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


@dataclass
class SweepCase:
    """Pre-computed similarities for a single case."""
    case_id: str
    expected: str  # "pass" or "fail"
    injection_type: str
    num_response_sentences: int
    num_context_sentences: int
    # max similarity for each response sentence against all context sentences
    sentence_max_similarities: list[float]


@dataclass
class ThresholdResult:
    """Result for a single threshold combination."""
    sim_threshold: float
    ratio_threshold: float
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    fire_count: int = 0
    total: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def fire_rate(self) -> float:
        return self.fire_count / self.total if self.total > 0 else 0.0


def compute_similarities(max_cases: int = 500) -> list[SweepCase]:
    """Step 1: Load cases and compute all sentence-level similarities."""
    from llm_judge.benchmarks.master_gt import MasterGroundTruthAdapter
    from llm_judge.properties import get_embedding_provider

    adapter = MasterGroundTruthAdapter()
    provider = get_embedding_provider()

    cases: list[SweepCase] = []
    count = 0

    print(f"Computing sentence-level similarities for up to {max_cases} cases...")
    t0 = time.time()

    for bc in adapter.load_cases(max_cases=max_cases):
        expected = bc.ground_truth.property_labels.get("1.1")
        if expected is None:
            continue

        response_text = bc.request.candidate_answer
        context_parts = list(bc.request.source_context or [])
        conversation = " ".join(msg.content for msg in bc.request.conversation)
        context = conversation + ("\n\n" + "\n".join(context_parts) if context_parts else "")

        response_sents = _split_sentences(response_text)
        context_sents = _split_sentences(context)

        if not response_sents or not context_sents:
            continue

        # Encode all sentences
        resp_embs = provider.encode(response_sents)
        ctx_embs = provider.encode(context_sents)

        # Compute max similarity for each response sentence
        max_sims = []
        for r_emb in resp_embs:
            max_sim = provider.max_similarity(r_emb, ctx_embs)
            max_sims.append(round(max_sim, 4))

        cases.append(SweepCase(
            case_id=bc.case_id,
            expected=expected,
            injection_type=bc.metadata.get("injection_type", ""),
            num_response_sentences=len(response_sents),
            num_context_sentences=len(context_sents),
            sentence_max_similarities=max_sims,
        ))

        count += 1
        if count % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {count} cases embedded ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"Computed similarities for {len(cases)} cases in {elapsed:.1f}s")
    return cases


def sweep_thresholds(
    cases: list[SweepCase],
    sim_thresholds: list[float] | None = None,
    ratio_thresholds: list[float] | None = None,
) -> list[ThresholdResult]:
    """Step 2: Apply every threshold combination post-hoc."""

    if sim_thresholds is None:
        sim_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    if ratio_thresholds is None:
        ratio_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    results: list[ThresholdResult] = []

    for sim_t in sim_thresholds:
        for ratio_t in ratio_thresholds:
            r = ThresholdResult(sim_threshold=sim_t, ratio_threshold=ratio_t)

            for case in cases:
                # Apply sentence-level threshold
                grounded = sum(1 for s in case.sentence_max_similarities if s >= sim_t)
                grounding_ratio = grounded / len(case.sentence_max_similarities)

                # Apply ratio threshold
                predicted = "fail" if grounding_ratio < ratio_t else "pass"

                r.total += 1
                if predicted == "fail":
                    r.fire_count += 1

                if predicted == "fail" and case.expected == "fail":
                    r.tp += 1
                elif predicted == "fail" and case.expected == "pass":
                    r.fp += 1
                elif predicted == "pass" and case.expected == "pass":
                    r.tn += 1
                elif predicted == "pass" and case.expected == "fail":
                    r.fn += 1

            results.append(r)

    return results


def print_report(
    results: list[ThresholdResult],
    cases: list[SweepCase],
) -> ThresholdResult:
    """Step 3: Print full experiment report and return best result."""

    # Sort by F1 descending
    results.sort(key=lambda r: r.f1, reverse=True)
    best = results[0]

    # Distribution analysis
    all_sims = []
    fail_sims = []
    pass_sims = []
    for case in cases:
        for s in case.sentence_max_similarities:
            all_sims.append(s)
            if case.expected == "fail":
                fail_sims.append(s)
            else:
                pass_sims.append(s)

    def percentiles(vals):
        if not vals:
            return {}
        sv = sorted(vals)
        n = len(sv)
        return {
            "count": n,
            "min": sv[0],
            "p10": sv[int(n*0.1)],
            "p25": sv[int(n*0.25)],
            "median": sv[int(n*0.5)],
            "p75": sv[int(n*0.75)],
            "p90": sv[int(n*0.9)],
            "max": sv[-1],
            "mean": round(sum(sv)/n, 4),
        }

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: THRESHOLD SWEEP — PROPERTY 1.1 GROUNDEDNESS")
    print("=" * 80)

    # Purpose
    print("\nPURPOSE")
    print("  Find the optimal (sentence_similarity_threshold, grounding_ratio_threshold)")
    print("  combination that maximises F1 on CS domain ground truth.")

    # Data collected
    print("\nDATA COLLECTED")
    print(f"  Cases with 1.1 ground truth: {len(cases)}")
    print(f"  Fail cases: {sum(1 for c in cases if c.expected == 'fail')}")
    print(f"  Pass cases: {sum(1 for c in cases if c.expected == 'pass')}")
    print(f"  Total sentence similarities computed: {len(all_sims)}")
    print(f"  Threshold combinations tested: {len(results)}")

    # Similarity distribution
    print("\nSIMILARITY DISTRIBUTION")
    print(f"  All sentences:  {percentiles(all_sims)}")
    print(f"  Fail sentences: {percentiles(fail_sims)}")
    print(f"  Pass sentences: {percentiles(pass_sims)}")

    # Top 10 results
    print("\nTOP 10 THRESHOLD COMBINATIONS (by F1)")
    print(f"  {'Sim Thresh':>10} {'Ratio Thresh':>12} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for r in results[:10]:
        print(f"  {r.sim_threshold:>10.2f} {r.ratio_threshold:>12.2f} {r.f1:>8.3f} {r.precision:>8.3f} {r.recall:>8.3f} {r.fire_rate:>7.1%} {r.tp:>5} {r.fp:>5} {r.tn:>5} {r.fn:>5}")

    # Bottom 5 for contrast
    print("\n  WORST 5 (for contrast)")
    for r in results[-5:]:
        print(f"  {r.sim_threshold:>10.2f} {r.ratio_threshold:>12.2f} {r.f1:>8.3f} {r.precision:>8.3f} {r.recall:>8.3f} {r.fire_rate:>7.1%} {r.tp:>5} {r.fp:>5} {r.tn:>5} {r.fn:>5}")

    # Experiment 1 comparison point
    print("\n  EXPERIMENT 1 COMPARISON (sim=0.50, ratio=0.30)")
    exp1 = [r for r in results if abs(r.sim_threshold - 0.50) < 0.01 and abs(r.ratio_threshold - 0.30) < 0.01]
    if exp1:
        r = exp1[0]
        print(f"  {r.sim_threshold:>10.2f} {r.ratio_threshold:>12.2f} {r.f1:>8.3f} {r.precision:>8.3f} {r.recall:>8.3f} {r.fire_rate:>7.1%} {r.tp:>5} {r.fp:>5} {r.tn:>5} {r.fn:>5}")

    # Best result detail
    print("\nBEST RESULT")
    print(f"  Sentence similarity threshold: {best.sim_threshold}")
    print(f"  Grounding ratio threshold:     {best.ratio_threshold}")
    print(f"  F1:        {best.f1:.3f}")
    print(f"  Precision: {best.precision:.3f}")
    print(f"  Recall:    {best.recall:.3f}")
    print(f"  Fire rate: {best.fire_rate:.1%}")
    print(f"  TP={best.tp}  FP={best.fp}  TN={best.tn}  FN={best.fn}")

    # Delta from Experiment 1
    if exp1:
        e1 = exp1[0]
        print("\n  DELTA FROM EXPERIMENT 1")
        print(f"  F1:        {e1.f1:.3f} \u2192 {best.f1:.3f} ({best.f1 - e1.f1:+.3f})")
        print(f"  Precision: {e1.precision:.3f} \u2192 {best.precision:.3f} ({best.precision - e1.precision:+.3f})")
        print(f"  Recall:    {e1.recall:.3f} \u2192 {best.recall:.3f} ({best.recall - e1.recall:+.3f})")
        print(f"  Fire rate: {e1.fire_rate:.1%} \u2192 {best.fire_rate:.1%}")

    # Observations
    print("\nOBSERVATIONS")

    # Find precision-recall tradeoff extremes
    high_p = max(results, key=lambda r: r.precision if r.f1 > 0.1 else 0)
    high_r = max(results, key=lambda r: r.recall if r.f1 > 0.1 else 0)
    print(f"  1. Highest precision (with F1>0.1): sim={high_p.sim_threshold}, ratio={high_p.ratio_threshold} \u2192 P={high_p.precision:.3f} R={high_p.recall:.3f}")
    print(f"  2. Highest recall    (with F1>0.1): sim={high_r.sim_threshold}, ratio={high_r.ratio_threshold} \u2192 P={high_r.precision:.3f} R={high_r.recall:.3f}")

    # Check if pass/fail distributions are separable
    pass_mean = sum(pass_sims) / len(pass_sims) if pass_sims else 0
    fail_mean = sum(fail_sims) / len(fail_sims) if fail_sims else 0
    print(f"  3. Mean similarity \u2014 pass: {pass_mean:.4f}, fail: {fail_mean:.4f}, gap: {pass_mean - fail_mean:.4f}")

    print(f"\n{'=' * 80}")

    # Save full results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "threshold_sweep_results.json"
    save_data = {
        "experiment": "Experiment 2: Threshold Sweep",
        "cases_evaluated": len(cases),
        "fail_cases": sum(1 for c in cases if c.expected == "fail"),
        "pass_cases": sum(1 for c in cases if c.expected == "pass"),
        "combinations_tested": len(results),
        "best": {
            "sim_threshold": best.sim_threshold,
            "ratio_threshold": best.ratio_threshold,
            "f1": round(best.f1, 4),
            "precision": round(best.precision, 4),
            "recall": round(best.recall, 4),
            "fire_rate": round(best.fire_rate, 4),
        },
        "similarity_distribution": {
            "all": percentiles(all_sims),
            "fail": percentiles(fail_sims),
            "pass": percentiles(pass_sims),
        },
        "all_results": [
            {
                "sim": r.sim_threshold, "ratio": r.ratio_threshold,
                "f1": round(r.f1, 4), "p": round(r.precision, 4),
                "r": round(r.recall, 4), "fire": round(r.fire_rate, 4),
                "tp": r.tp, "fp": r.fp, "tn": r.tn, "fn": r.fn,
            }
            for r in results
        ],
    }
    with save_path.open("w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved: {save_path}")

    return best


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 2: Threshold Sweep")
    parser.add_argument("--max-cases", type=int, default=500)
    args = parser.parse_args()

    # Step 1: Compute similarities (expensive, done once)
    cases = compute_similarities(max_cases=args.max_cases)

    # Step 2: Sweep thresholds (fast, post-hoc)
    results = sweep_thresholds(cases)

    # Step 3: Report
    print_report(results, cases)


if __name__ == "__main__":
    main()
