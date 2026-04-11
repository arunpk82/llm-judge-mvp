#!/usr/bin/env python3
"""
Gate 2 Validation — Integrated Pipeline (PCT-1).

EPIC 7.5/7.6 CHANGE: Uses IntegratedJudge instead of raw LLMJudge.
Every evaluation now produces:
  - Versioned prompt evidence (prompt_version_used)
  - Hallucination check results (grounding, claims, citations)
  - Property execution evidence (which properties ran, their flags)
  - Detection coverage metric

Usage:
    GEMINI_API_KEY=your-key poetry run python tools/validate_gate2.py \
        --dataset datasets/validation/cs_validation_scored.jsonl \
        --integrated

    Add --raw to use the old LLMJudge without properties (for comparison).
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
from llm_judge.paths import state_root  # noqa: E402
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
            Message(role=m["role"], content=m["content"]) for m in case["conversation"]
        ]
        request = PredictRequest(
            conversation=messages,
            candidate_answer=case["candidate_answer"],
            rubric_id=case["rubric_id"],
        )
        response = judge.evaluate(request)
        results.append(
            {
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
            }
        )
    return results


def run_gate2_integrated(
    disagreements: list[dict],
    engine: str,
) -> list[dict]:
    """Run integrated pipeline on disagreement cases (PCT-1)."""
    from llm_judge.integrated_judge import IntegratedJudge

    judge = IntegratedJudge(engine=engine)
    results = []

    for i, case in enumerate(disagreements):
        messages = [
            Message(role=m["role"], content=m["content"]) for m in case["conversation"]
        ]
        request = PredictRequest(
            conversation=messages,
            candidate_answer=case["candidate_answer"],
            rubric_id=case["rubric_id"],
        )
        print(f"  [{i + 1}/{len(disagreements)}] {case['case_id']}...", end=" ")
        start = time.time()
        try:
            enriched = judge.evaluate_enriched(request, case_id=case["case_id"])
            elapsed = time.time() - start
            print(f"done ({elapsed:.1f}s)")

            case["gate2_scores"] = enriched.scores
            case["gate2_decision"] = enriched.decision
            case["gate2_confidence"] = enriched.confidence
            case["gate2_flags"] = enriched.all_flags()
            case["gate2_explanations"] = enriched.explanations
            case["gate2_error"] = None

            # PCT-1 enrichments
            case["prompt_version"] = enriched.prompt_version
            case["detection_coverage"] = enriched.detection_coverage
            case["pipeline_latency_ms"] = enriched.pipeline_latency_ms

            if enriched.hallucination_result:
                case["hallucination"] = {
                    "risk_score": enriched.hallucination_result.risk_score,
                    "grounding_ratio": enriched.hallucination_result.grounding_ratio,
                    "ungrounded_claims": enriched.hallucination_result.ungrounded_claims,
                    "unverifiable_citations": enriched.hallucination_result.unverifiable_citations,
                    "flags": enriched.hallucination_result.flags,
                }
            else:
                case["hallucination"] = None

            # Property evidence summary
            case["property_evidence"] = {
                name: {
                    "id": ev.property_id,
                    "executed": ev.executed,
                    "gate_mode": ev.gate_mode,
                    "flags": ev.flags,
                }
                for name, ev in enriched.property_evidence.items()
            }

        except Exception as e:
            elapsed = time.time() - start
            print(f"ERROR ({elapsed:.1f}s): {e}")
            case["gate2_scores"] = None
            case["gate2_decision"] = None
            case["gate2_confidence"] = None
            case["gate2_flags"] = []
            case["gate2_explanations"] = None
            case["gate2_error"] = str(e)
            case["prompt_version"] = None
            case["hallucination"] = None
            case["property_evidence"] = None
            case["detection_coverage"] = None
            case["pipeline_latency_ms"] = None

        results.append(case)

        # Rate limit: 1 second between calls for free tier
        if i < len(disagreements) - 1:
            time.sleep(1.0)

    return results


def run_gate2_raw(
    disagreements: list[dict],
    engine: str,
) -> list[dict]:
    """Run raw LLMJudge on disagreement cases (legacy, for comparison)."""
    from llm_judge.llm_judge import LLMJudge

    judge = LLMJudge(engine=engine)

    results = []
    for i, case in enumerate(disagreements):
        messages = [
            Message(role=m["role"], content=m["content"]) for m in case["conversation"]
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
        if i < len(disagreements) - 1:
            time.sleep(1.0)

    return results


def print_comparison(results: list[dict], integrated: bool = False) -> None:
    """Print side-by-side comparison of Gate 1, Gate 2, and human scores."""
    dims = ["relevance", "clarity", "correctness", "tone"]

    print("\n" + "=" * 90)
    mode = "INTEGRATED PIPELINE (PCT-1)" if integrated else "RAW LLM JUDGE"
    print(f"GATE 2 VALIDATION REPORT — {mode}")
    print(f"LLM judge on {len(results)} disagreement cases")
    print("=" * 90)

    # Per-case comparison
    print(
        f"\n{'Case':<10} {'Dim':<12} {'Human':>6} {'Gate1':>6} {'Gate2':>6} "
        f"{'G1Δ':>5} {'G2Δ':>5} {'Winner':<8}"
    )
    print("-" * 80)

    gate1_wins = 0
    gate2_wins = 0
    ties = 0
    gate2_errors = 0

    for r in results:
        if r["gate2_error"]:
            print(
                f"{r['case_id']:<10} {'ERROR':<12} — Gate 2 failed: {r['gate2_error'][:40]}"
            )
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

        # PCT-1: Show hallucination results if available
        if integrated and r.get("hallucination"):
            hal = r["hallucination"]
            print(
                f"{'':<10} {'HALLUC':<12} "
                f"risk={hal['risk_score']:.2f} "
                f"grounding={hal['grounding_ratio']:.2f} "
                f"claims={hal['ungrounded_claims']} "
                f"citations={hal['unverifiable_citations']}"
            )
            if hal.get("flags"):
                print(f"{'':<10} {'FLAGS':<12} {', '.join(hal['flags'])}")

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
        print(
            f"    Gate 1: {g1_dec_match}/{len(valid)} "
            f"({g1_dec_match / len(valid) * 100:.0f}%)"
        )
        print(
            f"    Gate 2: {g2_dec_match}/{len(valid)} "
            f"({g2_dec_match / len(valid) * 100:.0f}%)"
        )

    # PCT-1: Detection coverage and prompt version
    if integrated and valid:
        print(f"\n  Prompt version: {valid[0].get('prompt_version', 'unknown')}")
        print(f"  Detection coverage: {valid[0].get('detection_coverage', 'unknown')}")

        # Hallucination summary
        hal_cases = [r for r in valid if r.get("hallucination")]
        if hal_cases:
            flagged = [r for r in hal_cases if r["hallucination"].get("flags")]
            avg_grounding = sum(
                r["hallucination"]["grounding_ratio"] for r in hal_cases
            ) / len(hal_cases)
            print("\n  Hallucination analysis:")
            print(f"    Cases checked: {len(hal_cases)}")
            print(f"    Cases flagged: {len(flagged)}")
            print(f"    Avg grounding ratio: {avg_grounding:.2f}")

    print(f"\n{'=' * 80}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate 2 Validation — Integrated Pipeline (PCT-1)",
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
        "--integrated",
        action="store_true",
        default=True,
        help="Use integrated pipeline with properties (default: true)",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Use raw LLMJudge without properties (for comparison)",
    )
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run Gate 2 on ALL cases, not just disagreements",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for results (default: reports/validation/)",
    )
    args = parser.parse_args()

    # --raw overrides --integrated
    use_integrated = not args.raw

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
            r for r in gate1_results if r["gate1_decision"] != r["human_decision"]
        ]
        print(f"\nFound {len(targets)} decision disagreements")

    if not targets:
        print("No disagreements — Gate 1 matches human on all cases!")
        return 0

    # 4. Run Gate 2
    mode_name = "integrated pipeline" if use_integrated else "raw LLM judge"
    print(f"\nRunning Gate 2 ({args.engine}, {mode_name})...")

    if use_integrated:
        gate2_results = run_gate2_integrated(targets, args.engine)
    else:
        gate2_results = run_gate2_raw(targets, args.engine)

    # 5. Print comparison
    print_comparison(gate2_results, integrated=use_integrated)

    # 6. Save results
    output_dir = (
        Path(args.output_dir) if args.output_dir else state_root() / "validation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = "_integrated" if use_integrated else "_raw"
    results_path = output_dir / f"gate2_results{suffix}.json"
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
