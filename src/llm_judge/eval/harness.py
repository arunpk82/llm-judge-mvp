# src/llm_judge/eval/harness.py

from __future__ import annotations

import asyncio
import json
from typing import Any

from llm_judge.paths import ensure_dir, state_root
from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import PredictRequest


def _sanitize_json_line(line: str) -> str:
    """
    Make JSONL ingestion more robust by removing stray control characters that
    sometimes sneak in during copy/paste or file generation.
    """
    # Keep \t, \n, \r. Remove other ASCII control chars 0x00-0x1F.
    cleaned = []
    for ch in line:
        code = ord(ch)
        if code < 32 and ch not in ("\t", "\n", "\r"):
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Load golden evaluation examples from JSONL."""
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            line = _sanitize_json_line(line)
            rows.append(json.loads(line))
    return rows


def mean_absolute_error(pred: int, truth: int) -> float:
    return abs(pred - truth)


def _bucket_name(confidence: float) -> str:
    if confidence >= 0.8:
        return "high (0.8-1.0)"
    if confidence >= 0.5:
        return "mid  (0.5-0.8)"
    return "low  (<0.5)"


async def run_eval(path: str = "datasets/golden/v1.jsonl") -> None:
    rows = load_jsonl(path)

    print(f"\n📌 Running evaluation on {len(rows)} golden examples...\n")

    engine = get_judge_engine()

    correct = 0

    # Confusion matrix counters
    tp = fp = tn = fn = 0

    # Per-dimension MAE tracking
    mae_totals: dict[str, float] = {
        "relevance": 0.0,
        "clarity": 0.0,
        "correctness": 0.0,
        "tone": 0.0,
    }
    mae_count = 0

    # Confidence bucket tracking
    buckets: dict[str, dict[str, int]] = {
        "high (0.8-1.0)": {"correct": 0, "total": 0},
        "mid  (0.5-0.8)": {"correct": 0, "total": 0},
        "low  (<0.5)": {"correct": 0, "total": 0},
    }

    for i, row in enumerate(rows, start=1):
        req = PredictRequest(
            conversation=row["conversation"],
            candidate_answer=row["candidate_answer"],
            rubric_id=row["rubric_id"],
        )

        # IMPORTANT: use engine contract (async)
        result = engine.evaluate(req)

        human_decision = row["human_decision"]
        judge_decision = result.decision

        is_correct = human_decision == judge_decision
        if is_correct:
            correct += 1

        # Confusion matrix update
        if human_decision == "pass" and judge_decision == "pass":
            tp += 1
        elif human_decision == "pass" and judge_decision == "fail":
            fn += 1
        elif human_decision == "fail" and judge_decision == "pass":
            fp += 1
        elif human_decision == "fail" and judge_decision == "fail":
            tn += 1

        # Confidence calibration bucket
        conf = float(result.confidence)
        bname = _bucket_name(conf)
        buckets[bname]["total"] += 1
        if is_correct:
            buckets[bname]["correct"] += 1

        # MAE calculation (if human_scores provided)
        if "human_scores" in row and isinstance(row["human_scores"], dict):
            mae_count += 1
            for dim, truth_val in row["human_scores"].items():
                pred_val = result.scores.get(dim, 3)
                mae_totals[str(dim)] += mean_absolute_error(
                    int(pred_val), int(truth_val)
                )

        print(
            f"[{i}] Human={human_decision} | Judge={judge_decision} "
            f"| Overall={result.overall_score:.2f} | Correct={is_correct}"
        )
        print(
            f"    Scores={result.scores} | Flags={result.flags} | Confidence={result.confidence:.2f}"
        )

        # Optional mismatch debug block if you want it (kept minimal)
        if not is_correct:
            print("    --- mismatch debug ---")
            print(f"    rubric_id={row.get('rubric_id')}")
            print(
                f"    conversation_last={row.get('conversation', [])[-1] if row.get('conversation') else None}"
            )
            print(f"    candidate_answer={row.get('candidate_answer')}")
            print("    ----------------------")

    # Summary
    accuracy = correct / len(rows) if rows else 0.0

    print("\n==============================")
    print(f"✅ Accuracy: {accuracy:.2%} ({correct}/{len(rows)})")
    print("==============================\n")

    print("📊 Confusion Matrix (Pass/Fail)")
    print("------------------------------")
    print(f"TP (pass→pass): {tp}")
    print(f"FN (pass→fail): {fn}")
    print(f"FP (fail→pass): {fp}")
    print(f"TN (fail→fail): {tn}")

    # MAE Report
    if mae_count > 0:
        print("\n📉 Per-Dimension MAE vs Human Scores")
        print("----------------------------------")
        for dim, total_err in mae_totals.items():
            print(f"{dim:12s}: {total_err / mae_count:.2f}")

    # Confidence Calibration Report
    print("\n🎯 Confidence Calibration Report")
    print("--------------------------------")
    for name, stats in buckets.items():
        if stats["total"] == 0:
            continue
        bucket_acc = stats["correct"] / stats["total"]
        print(f"{name:15s}: {bucket_acc:.2%} ({stats['correct']}/{stats['total']})")

    # Export report artifact
    report: dict[str, Any] = {
        "dataset_path": path,
        "n_examples": len(rows),
        "accuracy": accuracy,
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "mae": {
            dim: (mae_totals[dim] / mae_count if mae_count > 0 else None)
            for dim in mae_totals
        },
        "calibration": {
            name: (stats["correct"] / stats["total"] if stats["total"] > 0 else None)
            for name, stats in buckets.items()
        },
    }

    out_path = ensure_dir(state_root()) / "eval_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Saved evaluation report → {out_path}")
    print("\n🚀 Evaluation complete.\n")


if __name__ == "__main__":
    asyncio.run(run_eval())
