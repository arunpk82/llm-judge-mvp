#!/usr/bin/env python3
"""
Gate 2 Validation — LLM Judge on Cases Where Gate 1 Disagrees

Usage:
    GEMINI_API_KEY=your-key poetry run python tools/validate_gate2.py \
        --dataset datasets/validation/cs_validation_scored.jsonl

What it does:
    1. Runs Gate 1 (deterministic) on all cases
    2. Identifies cases where Gate 1 disagrees with human expert
    3. Runs Gate 2 (LLM judge) on ONLY those disagreement cases
    4. Compares Gate 1, Gate 2, and human scores side by side
    5. Produces a combined report showing where Gate 2 improves on Gate 1

This demonstrates the sequential gate model:
    - Gate 1 runs on everything (fast, cheap)
    - Gate 2 runs only where Gate 1 lacks confidence (slow, better)
    - The combined pipeline should beat either gate alone

Supported engines:
    JUDGE_ENGINE=gemini   GEMINI_API_KEY=...   (default)
    JUDGE_ENGINE=llm      LLM_API_KEY=...      (OpenAI-compatible)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_judge.deterministic_judge import DeterministicJudge  # noqa: E402
from llm_judge.schemas import Message, PredictRequest  # noqa: E402


def load_dataset(path: Path) -> list[dict]:
    cases = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def run_gate1(cases: list[dict]) -> list[dict]:
    """Run deterministic scorer on all cases."""
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
            "gate1_scores": response.scores,
            "gate1_decision": response.decision,
            "gate1_flags": response.flags,
            "human_scores": case["human_scores"],
            "human_decision": case["human_decision"],
            "rationale": case.get("rationale", ""),
            "conversation": case["conversation"],
            "candidate_answer": case["candidate_answer"],
            "rubric_id": case["rubric_id"],
        })
    return results


def run_gate2(
    disagreements: list[dict],
    engine: str,
) -> list[dict]:
    """Run LLM judge on disagreement cases only."""
    from llm_judge.llm_judge import LLMJudge
    judge = LLMJudge(engine=engine)

    results = []
    for i, case in enumerate(disagreements):
        messages = [
            Message(role=m["role"], content=m["content"])
            for m in case["conversation"]
        ]
        request = PredictRequest(
            conversation=messages,
            candidate_answer=case["candidate_answer"],
            rubric_id=case["rubric_id"],
        )
        print(f"  [{i + 1}/{len(disagreements)}] {case['case_id']}...", end=" ")
        start = time.time()
        try:
            response = judge.evaluate(request)
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")
            case["gate2_scores"] = response.scores
            case["gate2_decision"] = response.decision
            case["gate2_confidence"] = response.confidence
            case["gate2_flags"] = response.flags
            case["gate2_explanations"] = response.explanations
            case["gate2_error"] = None
        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            case["gate2_scores"] = None
            case["gate2_decision"] = None
            case["gate2_confidence"] = None
            case["gate2_flags"] = []
            case["gate2_explanations"] = None
            case["gate2_error"] = str(e)

        results.append(case)

        # Rate limit: 1 second between calls for free tier
        if i < len(disagreements) - 1:
            time.sleep(1.0)

    return results


def print_comparison(results: list[dict]) -> None:
    """Print side-by-side comparison of Gate 1, Gate 2, and human scores."""
    dims = ["relevance", "clarity", "correctness", "tone"]

    print("\n" + "=" * 90)
    print("GATE 2 VALIDATION REPORT")
    print(f"LLM judge on {len(results)} disagreement cases")
    print("=" * 90)

    # Per-case comparison
    print(f"\n{'Case':<10} {'Dim':<12} {'Human':>6} {'Gate1':>6} {'Gate2':>6} "
          f"{'G1Δ':>5} {'G2Δ':>5} {'Winner':<8}")
    print("-" * 80)

    gate1_wins = 0
    gate2_wins = 0
    ties = 0
    gate2_errors = 0

    for r in results:
        if r["gate2_error"]:
            print(f"{r['case_id']:<10} {'ERROR':<12} — Gate 2 failed: {r['gate2_error'][:40]}")
            gate2_errors += 1
            continue

        for dim in dims:
            h = r["human_scores"][dim]
            g1 = r["gate1_scores"][dim]
            g2 = r["gate2_scores"][dim]
            d1 = abs(g1 - h)
            d2 = abs(g2 - h)

            if d2 < d1:
                winner = "Gate2"
                gate2_wins += 1
            elif d1 < d2:
                winner = "Gate1"
                gate1_wins += 1
            else:
                winner = "tie"
                ties += 1

            print(
                f"{r['case_id']:<10} {dim:<12} {h:>6} {g1:>6} {g2:>6} "
                f"{d1:>5} {d2:>5} {winner:<8}"
            )

        # Decision comparison
        hd = r["human_decision"]
        g1d = r["gate1_decision"]
        g2d = r["gate2_decision"]
        g1_match = "✓" if g1d == hd else "✗"
        g2_match = "✓" if g2d == hd else "✗"
        print(
            f"{'':<10} {'DECISION':<12} {hd:>6} {g1d + g1_match:>6} "
            f"{g2d + g2_match:>6} {'':>5} {'':>5}"
        )
        print(f"{'':<10} {'rationale':<12} {r['rationale'][:55]}")
        print()

    # Summary
    total_dims = gate1_wins + gate2_wins + ties
    print("=" * 80)
    print("SUMMARY")
    print(f"  Gate 2 closer to human: {gate2_wins}/{total_dims} dimensions")
    print(f"  Gate 1 closer to human: {gate1_wins}/{total_dims} dimensions")
    print(f"  Ties:                   {ties}/{total_dims} dimensions")
    if gate2_errors > 0:
        print(f"  Gate 2 errors:          {gate2_errors} cases")

    # Decision agreement
    valid = [r for r in results if r["gate2_error"] is None]
    if valid:
        g1_dec_match = sum(
            1 for r in valid if r["gate1_decision"] == r["human_decision"]
        )
        g2_dec_match = sum(
            1 for r in valid if r["gate2_decision"] == r["human_decision"]
        )
        print("\n  Decision agreement on disagreement cases:")
        print(f"    Gate 1: {g1_dec_match}/{len(valid)} "
              f"({g1_dec_match / len(valid) * 100:.0f}%)")
        print(f"    Gate 2: {g2_dec_match}/{len(valid)} "
              f"({g2_dec_match / len(valid) * 100:.0f}%)")

    # Per-dimension Gate 2 within ±1
    if valid:
        print("\n  Gate 2 within ±1 of human (on disagreement cases only):")
        for dim in dims:
            within1 = sum(
                1 for r in valid
                if abs(r["gate2_scores"][dim] - r["human_scores"][dim]) <= 1
            )
            print(f"    {dim:>12}: {within1}/{len(valid)} "
                  f"({within1 / len(valid) * 100:.0f}%)")

    # Gate 2 explanations
    if valid and valid[0].get("gate2_explanations"):
        print("\n--- GATE 2 EXPLANATIONS (selected) ---")
        for r in valid[:5]:
            print(f"\n  {r['case_id']} (human={r['human_decision']}, "
                  f"gate2={r['gate2_decision']}):")
            expl = r.get("gate2_explanations") or {}
            for dim in dims:
                if dim in expl:
                    print(f"    {dim}: {expl[dim][:80]}")

    print(f"\n{'=' * 80}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 2 Validation — LLM judge on disagreement cases",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to human-scored validation dataset (JSONL)",
    )
    parser.add_argument(
        "--engine",
        default=os.getenv("JUDGE_ENGINE", "gemini"),
        help="LLM engine: gemini, llm (default: gemini)",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run Gate 2 on ALL cases, not just disagreements",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/validation",
        help="Directory for results (default: reports/validation/)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found: {dataset_path}", file=sys.stderr)
        return 1

    # Verify API key
    if args.engine == "gemini" and not os.getenv("GEMINI_API_KEY"):
        print(
            "ERROR: GEMINI_API_KEY not set. Get one from "
            "https://aistudio.google.com/apikey",
            file=sys.stderr,
        )
        return 1
    if args.engine == "llm" and not os.getenv("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY not set.", file=sys.stderr)
        return 1

    # 1. Load dataset
    cases = load_dataset(dataset_path)
    print(f"Loaded {len(cases)} cases from {dataset_path}")

    # 2. Run Gate 1
    print("\nRunning Gate 1 (deterministic)...")
    gate1_results = run_gate1(cases)

    # 3. Find disagreements
    if args.all_cases:
        targets = gate1_results
        print(f"\nRunning Gate 2 on ALL {len(targets)} cases...")
    else:
        targets = [
            r for r in gate1_results
            if r["gate1_decision"] != r["human_decision"]
        ]
        print(f"\nFound {len(targets)} decision disagreements")

    if not targets:
        print("No disagreements — Gate 1 matches human on all cases!")
        return 0

    # 4. Run Gate 2
    print(f"\nRunning Gate 2 ({args.engine})...")
    gate2_results = run_gate2(targets, args.engine)

    # 5. Print comparison
    print_comparison(gate2_results)

    # 6. Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "gate2_results.json"
    # Remove non-serializable fields
    save_results = []
    for r in gate2_results:
        save_r = {k: v for k, v in r.items() if k not in ("conversation",)}
        save_results.append(save_r)

    with results_path.open("w") as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
