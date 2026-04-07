"""
Experiment 5: Comprehensive Threshold Sweep with RAG Context.

With real KB documentation as source_context, the similarity distribution
has fundamentally changed. This sweep re-evaluates all parameters:

  1. Sentence similarity threshold (what counts as "grounded")
  2. Grounding ratio threshold (what % of sentences must be grounded)
  3. Min-sentence similarity threshold (flag if ANY sentence is below)
  4. Dual threshold (ratio OR min-sentence — either can trigger fail)

Computes all similarities once, then applies every combination post-hoc.

Usage:
    poetry run python -m llm_judge.benchmarks.sweep_rag --max-cases 500
"""
from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
OUTPUT_DIR = Path("experiments")


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


@dataclass
class SweepCase:
    case_id: str
    expected: str
    injection_type: str
    num_response_sents: int
    num_context_sents: int
    sentence_max_sims: list[float]  # max sim for each response sentence

    @property
    def min_sim(self) -> float:
        return min(self.sentence_max_sims) if self.sentence_max_sims else 1.0

    @property
    def mean_sim(self) -> float:
        return sum(self.sentence_max_sims) / len(self.sentence_max_sims) if self.sentence_max_sims else 1.0

    def grounding_ratio(self, sim_threshold: float) -> float:
        if not self.sentence_max_sims:
            return 1.0
        grounded = sum(1 for s in self.sentence_max_sims if s >= sim_threshold)
        return grounded / len(self.sentence_max_sims)


def compute_similarities(max_cases: int = 500) -> list[SweepCase]:
    from llm_judge.benchmarks.master_gt import MasterGroundTruthAdapter
    from llm_judge.properties import get_embedding_provider

    adapter = MasterGroundTruthAdapter()
    provider = get_embedding_provider()
    cases: list[SweepCase] = []

    print(f"Step 1: Computing sentence-level similarities ({max_cases} cases max)...")
    t0 = time.time()
    count = 0

    for bc in adapter.load_cases(max_cases=max_cases):
        expected = bc.ground_truth.property_labels.get("1.1")
        if expected is None:
            continue

        response_text = bc.request.candidate_answer
        context_parts = list(bc.request.source_context or [])
        conversation = " ".join(msg.content for msg in bc.request.conversation)
        context = conversation + ("\n\n" + "\n".join(context_parts) if context_parts else "")

        resp_sents = _split_sentences(response_text)
        ctx_sents = _split_sentences(context)

        if not resp_sents or not ctx_sents:
            continue

        resp_embs = provider.encode(resp_sents)
        ctx_embs = provider.encode(ctx_sents)

        max_sims = [
            round(provider.max_similarity(r_emb, ctx_embs), 4)
            for r_emb in resp_embs
        ]

        cases.append(SweepCase(
            case_id=bc.case_id,
            expected=expected,
            injection_type=bc.metadata.get("injection_type", ""),
            num_response_sents=len(resp_sents),
            num_context_sents=len(ctx_sents),
            sentence_max_sims=max_sims,
        ))

        count += 1
        if count % 100 == 0:
            print(f"  {count} cases ({time.time()-t0:.1f}s)")

    print(f"  Done: {len(cases)} cases in {time.time()-t0:.1f}s")
    return cases


def sweep_all(cases: list[SweepCase]) -> dict[str, Any]:
    """Sweep all threshold combinations across three strategies."""

    sim_thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    ratio_thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    min_sim_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    results: dict[str, list[dict]] = {
        "ratio_only": [],
        "min_sim_only": [],
        "dual": [],
    }

    n = len(cases)

    # Strategy 1: Ratio only (same as Experiment 2 but with RAG context)
    for st in sim_thresholds:
        for rt in ratio_thresholds:
            tp = fp = tn = fn = 0
            for c in cases:
                ratio = c.grounding_ratio(st)
                pred = "fail" if ratio < rt else "pass"
                if pred == "fail" and c.expected == "fail":
                    tp += 1
                elif pred == "fail" and c.expected == "pass":
                    fp += 1
                elif pred == "pass" and c.expected == "pass":
                    tn += 1
                else:
                    fn += 1
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
            results["ratio_only"].append({
                "sim": st, "ratio": rt, "f1": round(f1, 4),
                "p": round(p, 4), "r": round(r, 4),
                "fire": round((tp + fp) / n, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            })

    # Strategy 2: Min-sentence only (new)
    for mst in min_sim_thresholds:
        tp = fp = tn = fn = 0
        for c in cases:
            pred = "fail" if c.min_sim < mst else "pass"
            if pred == "fail" and c.expected == "fail":
                tp += 1
            elif pred == "fail" and c.expected == "pass":
                fp += 1
            elif pred == "pass" and c.expected == "pass":
                tn += 1
            else:
                fn += 1
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        results["min_sim_only"].append({
            "min_sim": mst, "f1": round(f1, 4),
            "p": round(p, 4), "r": round(r, 4),
            "fire": round((tp + fp) / n, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        })

    # Strategy 3: Dual threshold (ratio OR min-sentence — either triggers fail)
    for st in sim_thresholds:
        for rt in ratio_thresholds:
            for mst in min_sim_thresholds:
                tp = fp = tn = fn = 0
                for c in cases:
                    ratio = c.grounding_ratio(st)
                    fail_ratio = ratio < rt
                    fail_min = c.min_sim < mst
                    pred = "fail" if (fail_ratio or fail_min) else "pass"
                    if pred == "fail" and c.expected == "fail":
                        tp += 1
                    elif pred == "fail" and c.expected == "pass":
                        fp += 1
                    elif pred == "pass" and c.expected == "pass":
                        tn += 1
                    else:
                        fn += 1
                p = tp / (tp + fp) if (tp + fp) > 0 else 0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
                results["dual"].append({
                    "sim": st, "ratio": rt, "min_sim": mst,
                    "f1": round(f1, 4), "p": round(p, 4), "r": round(r, 4),
                    "fire": round((tp + fp) / n, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                })

    return results


def print_report(cases: list[SweepCase], results: dict[str, Any]) -> None:
    n = len(cases)
    fail_count = sum(1 for c in cases if c.expected == "fail")  # noqa: F841
    pass_count = n - fail_count

    # Similarity distributions WITH RAG context
    all_sims = [s for c in cases for s in c.sentence_max_sims]
    fail_sims = [s for c in cases if c.expected == "fail" for s in c.sentence_max_sims]
    pass_sims = [s for c in cases if c.expected == "pass" for s in c.sentence_max_sims]
    fail_mins = [c.min_sim for c in cases if c.expected == "fail"]
    pass_mins = [c.min_sim for c in cases if c.expected == "pass"]

    def stats(vals):
        if not vals:
            return "n/a"
        sv = sorted(vals)
        n = len(sv)
        return f"n={n} min={sv[0]:.3f} p10={sv[int(n*.1)]:.3f} p25={sv[int(n*.25)]:.3f} med={sv[n//2]:.3f} p75={sv[int(n*.75)]:.3f} p90={sv[int(n*.9)]:.3f} max={sv[-1]:.3f} mean={sum(sv)/n:.3f}"

    print(f"\n{'='*80}")
    print("EXPERIMENT 5: COMPREHENSIVE THRESHOLD SWEEP WITH RAG CONTEXT")
    print(f"{'='*80}")

    print("\nDATA SUMMARY")
    print(f"  Cases: {n} (fail={fail_count}, pass={pass_count})")
    print(f"  Total sentence similarities: {len(all_sims)}")
    print("  Strategies tested: 3 (ratio-only, min-sim-only, dual)")
    print(f"  Combinations: ratio={len(results['ratio_only'])}, min-sim={len(results['min_sim_only'])}, dual={len(results['dual'])}")

    print("\nSIMILARITY DISTRIBUTIONS (with RAG context)")
    print(f"  All sentences:  {stats(all_sims)}")
    print(f"  Pass sentences: {stats(pass_sims)}")
    print(f"  Fail sentences: {stats(fail_sims)}")
    print(f"  Pass min-sim:   {stats(pass_mins)}")
    print(f"  Fail min-sim:   {stats(fail_mins)}")

    # Per injection type
    by_type: dict[str, list[float]] = defaultdict(list)
    by_type_min: dict[str, list[float]] = defaultdict(list)
    for c in cases:
        t = c.injection_type or "clean"
        by_type[t].extend(c.sentence_max_sims)
        by_type_min[t].append(c.min_sim)

    print("\nPER INJECTION TYPE (min-sim)")
    for t in sorted(by_type_min.keys()):
        vals = by_type_min[t]
        sv = sorted(vals)
        print(f"  {t:<25} n={len(sv):<4} min={sv[0]:.3f} med={sv[len(sv)//2]:.3f} mean={sum(sv)/len(sv):.3f}")

    # Strategy 1: Ratio only
    ratio_sorted = sorted(results["ratio_only"], key=lambda x: x["f1"], reverse=True)
    print("\nSTRATEGY 1: RATIO ONLY (top 5)")
    print(f"  {'Sim':>6} {'Ratio':>6} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for r in ratio_sorted[:5]:
        print(f"  {r['sim']:>6.2f} {r['ratio']:>6.2f} {r['f1']:>8.3f} {r['p']:>8.3f} {r['r']:>8.3f} {r['fire']:>7.1%} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}")

    # Strategy 2: Min-sim only
    min_sorted = sorted(results["min_sim_only"], key=lambda x: x["f1"], reverse=True)
    print("\nSTRATEGY 2: MIN-SENTENCE ONLY (all)")
    print(f"  {'MinSim':>6} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print(f"  {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for r in min_sorted:
        print(f"  {r['min_sim']:>6.2f} {r['f1']:>8.3f} {r['p']:>8.3f} {r['r']:>8.3f} {r['fire']:>7.1%} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}")

    # Strategy 3: Dual threshold
    dual_sorted = sorted(results["dual"], key=lambda x: x["f1"], reverse=True)
    print("\nSTRATEGY 3: DUAL THRESHOLD (top 10)")
    print(f"  {'Sim':>6} {'Ratio':>6} {'MinSim':>6} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8} {'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print(f"  {'-'*6} {'-'*6} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*5} {'-'*5} {'-'*5} {'-'*5}")
    for r in dual_sorted[:10]:
        print(f"  {r['sim']:>6.2f} {r['ratio']:>6.2f} {r['min_sim']:>6.2f} {r['f1']:>8.3f} {r['p']:>8.3f} {r['r']:>8.3f} {r['fire']:>7.1%} {r['tp']:>5} {r['fp']:>5} {r['tn']:>5} {r['fn']:>5}")

    # Best from each strategy
    best_ratio = ratio_sorted[0]
    best_min = min_sorted[0]
    best_dual = dual_sorted[0]

    print("\nBEST FROM EACH STRATEGY")
    print(f"  Ratio only: sim={best_ratio['sim']}, ratio={best_ratio['ratio']}  → F1={best_ratio['f1']:.3f} P={best_ratio['p']:.3f} R={best_ratio['r']:.3f}")
    print(f"  Min-sim:    min_sim={best_min['min_sim']}                → F1={best_min['f1']:.3f} P={best_min['p']:.3f} R={best_min['r']:.3f}")
    print(f"  Dual:       sim={best_dual['sim']}, ratio={best_dual['ratio']}, min_sim={best_dual['min_sim']} → F1={best_dual['f1']:.3f} P={best_dual['p']:.3f} R={best_dual['r']:.3f}")

    # Compare with previous experiments
    print("\nPROGRESS")
    print(f"  {'Experiment':<35} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Baseline (token overlap)':<35} {'0.461':>8} {'0.300':>8} {'1.000':>8} {'99.6%':>8}")
    print(f"  {'Exp 2 (tuned, no RAG)':<35} {'0.570':>8} {'0.447':>8} {'0.787':>8} {'54.2%':>8}")
    print(f"  {'Exp 4 (ratio, with RAG)':<35} {best_ratio['f1']:>8.3f} {best_ratio['p']:>8.3f} {best_ratio['r']:>8.3f} {best_ratio['fire']:>7.1%}")
    print(f"  {'Exp 5 best (dual, with RAG)':<35} {best_dual['f1']:>8.3f} {best_dual['p']:>8.3f} {best_dual['r']:>8.3f} {best_dual['fire']:>7.1%}")
    print(f"  {'Published (GPT-4)':<35} {'0.635':>8} {'—':>8} {'—':>8} {'—':>8}")

    print(f"\n{'='*80}")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "sweep_rag_results.json"
    save_data = {
        "experiment": "Experiment 5: Comprehensive Sweep with RAG Context",
        "cases": n, "fail_cases": fail_count, "pass_cases": pass_count,
        "distributions": {
            "all_sims": {"mean": round(sum(all_sims)/len(all_sims), 4), "count": len(all_sims)},
            "pass_min_sims": {"mean": round(sum(pass_mins)/len(pass_mins), 4) if pass_mins else 0},
            "fail_min_sims": {"mean": round(sum(fail_mins)/len(fail_mins), 4) if fail_mins else 0},
        },
        "best_ratio": best_ratio,
        "best_min_sim": best_min,
        "best_dual": best_dual,
        "all_ratio": ratio_sorted[:20],
        "all_min_sim": min_sorted,
        "all_dual": dual_sorted[:30],
    }
    with save_path.open("w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 5: Comprehensive Sweep with RAG")
    parser.add_argument("--max-cases", type=int, default=500)
    args = parser.parse_args()

    cases = compute_similarities(max_cases=args.max_cases)
    results = sweep_all(cases)
    print_report(cases, results)


if __name__ == "__main__":
    main()
