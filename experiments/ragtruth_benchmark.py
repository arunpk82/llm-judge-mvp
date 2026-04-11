"""
RAGTruth Benchmark Runner — Response-Level Hallucination Detection.

Runs the full 5-layer pipeline on RAGTruth test cases and computes
precision, recall, F1 comparable to published baselines:
  - GPT-4-turbo: F1 = 63.4%
  - Fine-tuned Llama-2-13B: F1 = 78.7%
  - Human inter-annotator agreement: F1 ≈ 78%

Usage:
    # Quick benchmark (50 cases, ~20 min with MiniCheck on CPU)
    poetry run python experiments/ragtruth_benchmark.py --max-cases 50

    # Full benchmark (all test cases, ~2 hours)
    poetry run python experiments/ragtruth_benchmark.py --max-cases 500

    # Without MiniCheck (DeBERTa only, faster)
    poetry run python experiments/ragtruth_benchmark.py --no-minicheck
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import check_hallucination


@dataclass
class BenchmarkResult:
    case_id: str
    ground_truth: str  # "pass" or "fail"
    predicted: str  # "pass" or "fail"
    gate1_decision: str
    gate2_decision: str
    layer_stats: dict[str, int]
    correct: bool


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGTruth Benchmark Runner")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument(
        "--no-minicheck", action="store_true", help="Skip MiniCheck, use DeBERTa only"
    )
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = RAGTruthAdapter()
    results: list[BenchmarkResult] = []
    start_time = time.time()

    # If --no-minicheck, monkeypatch to skip MiniCheck
    if args.no_minicheck:
        import llm_judge.calibration.hallucination as hal

        hal._l2a_minicheck = lambda sentence, source_doc: False
        hal._load_minicheck = lambda: None
        print("MiniCheck DISABLED — using DeBERTa NLI only")

    total = 0
    skipped = 0

    for case in adapter.load_cases(max_cases=args.max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            skipped += 1
            continue

        total += 1

        # Build context from source_context
        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        source_doc = "\n".join(ctx_parts) if ctx_parts else context
        response = case.request.candidate_answer

        # Run full pipeline (L0 → L1 → L2a → L2b → L3 → L4)
        result = check_hallucination(
            response=response,
            context=context,
            source_context=source_doc,
            case_id=case.case_id,
            gate2_routing="pass",  # enable full pipeline
        )

        # Response-level decision:
        # With gate2_routing="pass", the full pipeline runs and gate2_decision
        # is the authoritative verdict from L2/L3/L4 analysis.
        # Gate 1 fail only stops the pipeline when gate2_routing="none".
        if result.gate2_decision == "fail":
            predicted = "fail"
        elif result.gate2_decision == "pass":
            predicted = "pass"
        elif result.gate1_decision in ("fail", "ambiguous"):
            # Pipeline didn't reach L2+ (gate2_routing="none" or no sentences)
            predicted = "fail"
        else:
            predicted = "pass"

        results.append(
            BenchmarkResult(
                case_id=case.case_id,
                ground_truth=gt,
                predicted=predicted,
                gate1_decision=result.gate1_decision,
                gate2_decision=result.gate2_decision,
                layer_stats=result.layer_stats,
                correct=(predicted == gt),
            )
        )

        if total % 10 == 0:
            elapsed = time.time() - start_time
            correct_so_far = sum(1 for r in results if r.correct)
            print(
                f"  {total} cases | {correct_so_far}/{total} correct "
                f"({correct_so_far/total*100:.0f}%) | {elapsed:.0f}s"
            )

    elapsed = time.time() - start_time

    # Compute metrics
    tp = sum(1 for r in results if r.ground_truth == "fail" and r.predicted == "fail")
    fp = sum(1 for r in results if r.ground_truth == "pass" and r.predicted == "fail")
    fn = sum(1 for r in results if r.ground_truth == "fail" and r.predicted == "pass")
    tn = sum(1 for r in results if r.ground_truth == "pass" and r.predicted == "pass")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / len(results) if results else 0.0

    gt_fail = sum(1 for r in results if r.ground_truth == "fail")
    gt_pass = sum(1 for r in results if r.ground_truth == "pass")

    # Layer stats aggregation
    total_l0 = sum(r.layer_stats.get("L0", 0) for r in results)
    total_l2a = sum(r.layer_stats.get("L2a_minicheck", 0) for r in results)
    total_l2b = sum(r.layer_stats.get("L2b_nli", 0) for r in results)
    total_l3 = sum(r.layer_stats.get("L3", 0) for r in results)
    total_l4s = sum(r.layer_stats.get("L4_supported", 0) for r in results)
    total_l4u = sum(r.layer_stats.get("L4_unsupported", 0) for r in results)
    total_g1f = sum(1 for r in results if r.gate1_decision in ("fail", "ambiguous"))

    print(f"\n{'='*70}")
    print("RAGTRUTH BENCHMARK — RESPONSE-LEVEL HALLUCINATION DETECTION")
    print(f"{'='*70}")
    print(f"Cases: {total} (skipped {skipped} without GT)")
    print(f"Ground truth: {gt_fail} hallucinated, {gt_pass} clean")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/max(1,total):.1f}s/case)")
    print(f"MiniCheck: {'DISABLED' if args.no_minicheck else 'ENABLED'}")

    print("\nCONFUSION MATRIX:")
    print("               Predicted FAIL  Predicted PASS")
    print(f"  GT=FAIL (H)    TP={tp:>4}         FN={fn:>4}")
    print(f"  GT=PASS (C)    FP={fp:>4}         TN={tn:>4}")

    print("\nMETRICS:")
    print(f"  Precision:  {precision:.3f}  ({tp}/{tp+fp})")
    print(f"  Recall:     {recall:.3f}  ({tp}/{tp+fn})")
    print(f"  F1 Score:   {f1:.3f}")
    print(f"  Accuracy:   {accuracy:.3f}  ({tp+tn}/{len(results)})")

    print("\nCOMPARISON TO PUBLISHED BASELINES (response-level F1):")
    print("  GPT-4-turbo:             0.634")
    print("  SelfCheckGPT (GPT-3.5):  0.588")
    print("  FT Llama-2-13B:          0.787")
    print("  Human agreement:         ~0.780")
    print(f"  *** Our pipeline:        {f1:.3f} ***")

    delta = f1 - 0.634
    if delta > 0:
        print(f"  → {delta:+.3f} vs GPT-4-turbo ({delta/0.634*100:+.0f}%)")
    else:
        print(f"  → {delta:+.3f} vs GPT-4-turbo")

    print("\nLAYER DISTRIBUTION:")
    print(f"  Gate 1 FAIL:     {total_g1f:>4} cases")
    print(f"  L0 deterministic: {total_l0:>4} sentences")
    print(f"  L2a MiniCheck:    {total_l2a:>4} sentences")
    print(f"  L2b DeBERTa:      {total_l2b:>4} sentences")
    print(f"  L3 GraphRAG:      {total_l3:>4} sentences")
    print(f"  L4 supported:     {total_l4s:>4} sentences")
    print(f"  L4 unsupported:   {total_l4u:>4} sentences")

    # Show misclassifications
    false_negatives = [
        r for r in results if r.ground_truth == "fail" and r.predicted == "pass"
    ]
    false_positives = [
        r for r in results if r.ground_truth == "pass" and r.predicted == "fail"
    ]

    if false_negatives:
        print(f"\nFALSE NEGATIVES (hallucinations we missed, {len(false_negatives)}):")
        for r in false_negatives[:5]:
            print(
                f"  {r.case_id}: G1={r.gate1_decision} G2={r.gate2_decision} "
                f"stats={r.layer_stats}"
            )

    if false_positives:
        print(f"\nFALSE POSITIVES (clean responses flagged, {len(false_positives)}):")
        for r in false_positives[:5]:
            print(
                f"  {r.case_id}: G1={r.gate1_decision} G2={r.gate2_decision} "
                f"stats={r.layer_stats}"
            )

    print(f"{'='*70}")

    # Save results
    save_data = {
        "benchmark": "RAGTruth Response-Level Hallucination Detection",
        "total_cases": total,
        "minicheck_enabled": not args.no_minicheck,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "gt_fail": gt_fail,
        "gt_pass": gt_pass,
        "elapsed_seconds": round(elapsed, 1),
        "comparison": {
            "gpt4_turbo_f1": 0.634,
            "selfcheckgpt_f1": 0.588,
            "ft_llama2_13b_f1": 0.787,
            "our_pipeline_f1": round(f1, 4),
        },
        "layer_totals": {
            "gate1_fail": total_g1f,
            "L0": total_l0,
            "L2a_minicheck": total_l2a,
            "L2b_nli": total_l2b,
            "L3": total_l3,
            "L4_supported": total_l4s,
            "L4_unsupported": total_l4u,
        },
        "cases": [asdict(r) for r in results],
    }
    save_path = output_dir / "ragtruth_benchmark_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
