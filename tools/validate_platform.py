#!/usr/bin/env python3
"""
Layer 2 Validation Runner — Platform vs Human Expert Agreement

Usage:
    poetry run python tools/validate_platform.py --dataset datasets/validation/cs_validation_scored.jsonl

What it does:
    1. Loads a human-scored validation dataset (JSONL with human_scores + human_decision)
    2. Runs each case through the platform's deterministic scorer
    3. Produces a per-dimension agreement report
    4. Recommends property lifecycle status (auto-gated / calibrate / human-gated)
    5. Saves detailed results to reports/validation/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_judge.deterministic_judge import DeterministicJudge  # noqa: E402
from llm_judge.schemas import Message, PredictRequest  # noqa: E402


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def load_validation_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    for case in cases:
        assert "case_id" in case, f"Missing case_id in {case}"
        assert "human_scores" in case, f"Missing human_scores in {case.get('case_id', '?')}"
        assert "human_decision" in case, f"Missing human_decision in {case.get('case_id', '?')}"
    return cases


def run_platform(cases: list[dict]) -> list[dict]:
    judge = DeterministicJudge()
    results = []
    for case in cases:
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in case["conversation"]
        ]
        request = PredictRequest(
            conversation=messages,
            candidate_answer=case["candidate_answer"],
            rubric_id=case["rubric_id"],
        )
        response = judge.evaluate(request)
        results.append({
            "case_id": case["case_id"],
            "platform_scores": response.scores,
            "platform_decision": response.decision,
            "platform_overall": response.overall_score,
            "platform_confidence": response.confidence,
            "platform_flags": response.flags,
            "human_scores": case["human_scores"],
            "human_decision": case["human_decision"],
            "rationale": case.get("rationale", ""),
        })
    return results


def analyze(results: list[dict]) -> dict:
    dims = ["relevance", "clarity", "correctness", "tone"]
    n = len(results)

    tp = sum(
        1 for r in results
        if r["human_decision"] == "pass" and r["platform_decision"] == "pass"
    )
    tn = sum(
        1 for r in results
        if r["human_decision"] == "fail" and r["platform_decision"] == "fail"
    )
    fp = sum(
        1 for r in results
        if r["human_decision"] == "fail" and r["platform_decision"] == "pass"
    )
    fn = sum(
        1 for r in results
        if r["human_decision"] == "pass" and r["platform_decision"] == "fail"
    )

    decision_agreement = (tp + tn) / n
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    fpr = fp / max(1, fp + tn)

    dim_analysis = {}
    for dim in dims:
        exact = 0
        within1 = 0
        within2 = 0
        total_delta = 0
        higher = 0
        lower = 0
        equal = 0

        for r in results:
            h = r["human_scores"][dim]
            p = r["platform_scores"][dim]
            delta = p - h
            total_delta += delta

            if delta == 0:
                exact += 1
                equal += 1
            elif delta > 0:
                higher += 1
            else:
                lower += 1

            if abs(delta) <= 1:
                within1 += 1
            if abs(delta) <= 2:
                within2 += 1

        within1_pct = within1 / n * 100
        if within1_pct >= 70:
            lifecycle = "AUTO-GATED"
        elif within1_pct >= 50:
            lifecycle = "CALIBRATE"
        else:
            lifecycle = "HUMAN-GATED"

        dim_analysis[dim] = {
            "exact_match": exact,
            "within_1": within1,
            "within_1_pct": round(within1_pct, 1),
            "within_2": within2,
            "avg_delta": round(total_delta / n, 2),
            "platform_higher": higher,
            "platform_lower": lower,
            "equal": equal,
            "lifecycle_recommendation": lifecycle,
        }

    worst = []
    for r in results:
        for dim in dims:
            delta = r["platform_scores"][dim] - r["human_scores"][dim]
            if abs(delta) >= 3:
                worst.append({
                    "case_id": r["case_id"],
                    "dimension": dim,
                    "human": r["human_scores"][dim],
                    "platform": r["platform_scores"][dim],
                    "delta": delta,
                    "rationale": r["rationale"],
                })

    decision_disagreements = []
    for r in results:
        if r["human_decision"] != r["platform_decision"]:
            h_avg = sum(r["human_scores"].values()) / 4
            p_avg = sum(r["platform_scores"].values()) / 4
            decision_disagreements.append({
                "case_id": r["case_id"],
                "human_decision": r["human_decision"],
                "platform_decision": r["platform_decision"],
                "human_avg": round(h_avg, 1),
                "platform_avg": round(p_avg, 1),
                "rationale": r["rationale"],
            })

    return {
        "timestamp": _utc_now_iso(),
        "total_cases": n,
        "decision": {
            "agreement": round(decision_agreement * 100, 1),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "false_positive_rate": round(fpr * 100, 1),
        },
        "dimensions": dim_analysis,
        "worst_disagreements": worst,
        "decision_disagreements": decision_disagreements,
    }


def print_report(analysis: dict) -> None:
    print("=" * 70)
    print("LAYER 2 VALIDATION REPORT")
    print(f"Platform vs Human Expert — {analysis['total_cases']} Cases")
    print(f"Timestamp: {analysis['timestamp']}")
    print("=" * 70)

    d = analysis["decision"]
    print("\n--- DECISION AGREEMENT ---")
    print(f"Overall: {d['agreement']:.0f}%")
    print("                    Platform PASS    Platform FAIL")
    print(
        f"  Human PASS           {d['true_positives']:>3} (TP)"
        f"         {d['false_negatives']:>3} (FN)"
    )
    print(
        f"  Human FAIL           {d['false_positives']:>3} (FP)"
        f"         {d['true_negatives']:>3} (TN)"
    )
    print(
        f"  Precision: {d['precision']:.0f}%  |  "
        f"Recall: {d['recall']:.0f}%  |  "
        f"FPR: {d['false_positive_rate']:.0f}%"
    )

    print("\n--- PER-DIMENSION AGREEMENT ---")
    for dim, stats in analysis["dimensions"].items():
        n = analysis["total_cases"]
        print(f"\n  {dim.upper()}")
        print(f"    Within ±1:  {stats['within_1']}/{n} ({stats['within_1_pct']:.0f}%)")
        if stats["avg_delta"] > 0:
            avg_dir = "platform scores higher"
        elif stats["avg_delta"] < 0:
            avg_dir = "platform scores lower"
        else:
            avg_dir = "balanced"
        print(f"    Avg delta:  {stats['avg_delta']:+.2f} ({avg_dir})")
        print(f"    → {stats['lifecycle_recommendation']}")

    if analysis["worst_disagreements"]:
        print("\n--- WORST DISAGREEMENTS (|delta| >= 3) ---")
        for w in analysis["worst_disagreements"]:
            print(
                f"  {w['case_id']} {w['dimension']}: "
                f"Human={w['human']}, Platform={w['platform']} "
                f"(Δ={w['delta']:+d}) — {w['rationale'][:55]}"
            )

    if analysis["decision_disagreements"]:
        print(
            f"\n--- DECISION DISAGREEMENTS "
            f"({len(analysis['decision_disagreements'])} cases) ---"
        )
        for dd in analysis["decision_disagreements"]:
            print(
                f"  {dd['case_id']}: "
                f"Human={dd['human_decision']}({dd['human_avg']}) "
                f"Platform={dd['platform_decision']}({dd['platform_avg']}) "
                f"— {dd['rationale'][:50]}"
            )

    print("\n--- PROPERTY LIFECYCLE ---")
    for dim, stats in analysis["dimensions"].items():
        print(
            f"  {dim:>12}: {stats['within_1_pct']:.0f}% within ±1 "
            f"→ {stats['lifecycle_recommendation']}"
        )

    print(f"\n{'=' * 70}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Layer 2 Validation — compare platform judgment vs human expert",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to human-scored validation dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/validation",
        help="Directory for validation results (default: reports/validation/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output (still writes report file)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = load_validation_dataset(dataset_path)
    if not args.quiet:
        print(f"Loaded {len(cases)} validation cases from {dataset_path}")

    results = run_platform(cases)
    analysis = analyze(results)

    results_path = output_dir / "validation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    report_path = output_dir / "validation_report.json"
    with report_path.open("w") as f:
        json.dump(analysis, f, indent=2)

    if not args.quiet:
        print_report(analysis)
        print(f"\nResults saved to: {results_path}")
        print(f"Report saved to:  {report_path}")

    human_gated = sum(
        1 for d in analysis["dimensions"].values()
        if d["lifecycle_recommendation"] == "HUMAN-GATED"
    )

    if human_gated > 0:
        if not args.quiet:
            print(
                f"\n⚠  {human_gated} dimension(s) need HUMAN-GATING "
                "— platform decision should not be trusted without human review"
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
