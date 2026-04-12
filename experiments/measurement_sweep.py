"""
Phase 2: Measurement Sweep — All 28 Properties × All 30 Cases
===============================================================

Runs IntegratedJudge with all 28 properties enabled against the full
validation dataset. Compares judge output against human ground truth.
Produces the detection summary table for each property.

Prerequisites:
    - All 28 properties enabled in property_config.yaml (PCT-5)
    - GEMINI_API_KEY set (or JUDGE_ENGINE configured)
    - Knowledge base loaded at configs/retrieval/knowledge_base.json
    - cs_validation_scored.jsonl in project root or datasets/validation/

Usage:
    JUDGE_ENGINE=gemini poetry run python experiments/measurement_sweep.py

Output:
    experiments/measurement_sweep_results.json  — raw per-case results
    Console: detection summary table with Recall/Precision per property
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_judge.schemas import Message, PredictRequest


@dataclass
class PropertyMetric:
    """Tracks detection performance for one property."""

    property_id: str
    property_name: str
    category: str
    cases_evaluated: int = 0
    detected: int = 0  # correctly flagged (true positive)
    missed: int = 0  # should have flagged but didn't (false negative)
    false_alarm: int = 0  # flagged but shouldn't have (false positive)
    correct_pass: int = 0  # correctly not flagged (true negative)
    errors: int = 0  # property raised an exception
    sample_flags: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float | None:
        total_positive = self.detected + self.missed
        if total_positive == 0:
            return None
        return self.detected / total_positive

    @property
    def precision(self) -> float | None:
        total_flagged = self.detected + self.false_alarm
        if total_flagged == 0:
            return None
        return self.detected / total_flagged

    @property
    def f1(self) -> float | None:
        r, p = self.recall, self.precision
        if r is None or p is None or (r + p) == 0:
            return None
        return 2 * r * p / (r + p)


def _is_problem_case(case: dict, property_name: str, category: str) -> bool | None:
    """Determine if a case SHOULD be flagged by this property, based on human scores.

    Returns True if the case has a known problem this property should catch,
    False if the case is clean, None if we can't determine (no ground truth).
    """
    human = case["human_decision"]
    scores = case["human_scores"]
    corr = scores.get("correctness", 5)
    tone_score = scores.get("tone", 5)
    relevance_score = scores.get("relevance", 5)
    clarity_score = scores.get("clarity", 5)

    if category == "faithfulness":
        # Fabrication: correctness <= 2 indicates factual problems
        if property_name in (
            "groundedness",
            "ungrounded_claims",
            "citation_verification",
            "attribution_accuracy",
            "fabrication_detection",
        ):
            return corr <= 2

    elif category == "semantic_quality":
        # Map property to its human score dimension
        dim_map = {
            "relevance": relevance_score,
            "clarity": clarity_score,
            "correctness": corr,
            "tone": tone_score,
            "completeness": None,  # no human score for this
            "coherence": None,
            "depth_nuance": None,
        }
        score = dim_map.get(property_name)
        if score is not None:
            return score <= 2  # score 1-2 = should be flagged

    elif category == "safety":
        # No safety problems in current dataset — return None (can't evaluate)
        return None

    elif category == "task_fidelity":
        # Very short responses (cs_009) might fail instruction following
        if human == "fail" and corr <= 2:
            return True
        return None  # can't determine from current ground truth

    elif category == "robustness":
        # Calibration diagnostics — not per-response, can't evaluate per-case
        return None

    elif category == "performance":
        # Diagnostics — not per-response
        return None

    return None


def _has_flags(evidence: dict, property_name: str) -> bool:
    """Check if a property produced any flags in the enriched response."""
    prop_ev = evidence.get("property_evidence", {}).get(property_name, {})
    if prop_ev.get("flags"):
        return True

    # Check hallucination-specific results
    if property_name in ("groundedness", "ungrounded_claims", "citation_verification"):
        hall = evidence.get("hallucination", {})
        if property_name == "groundedness":
            return any("low_grounding" in f for f in hall.get("flags", []))
        elif property_name == "ungrounded_claims":
            return any("ungrounded_claims" in f for f in hall.get("flags", []))
        elif property_name == "citation_verification":
            return any("unverifiable" in f for f in hall.get("flags", []))

    # Check semantic quality — score <= 2 means flagged
    prop_ev = evidence.get("property_evidence", {}).get(property_name, {})
    if prop_ev.get("executed") and property_name in (
        "relevance",
        "clarity",
        "correctness",
        "tone",
        "completeness",
        "coherence",
        "depth_nuance",
    ):
        flags = prop_ev.get("flags", [])
        return any(f"low_{property_name}" in f for f in flags)

    return False


def run_sweep():
    """Run the full measurement sweep."""
    repo_root = Path(__file__).parent.parent

    # Load validation cases
    data_path = repo_root / "cs_validation_scored.jsonl"
    if not data_path.exists():
        data_path = repo_root / "datasets" / "validation" / "cs_validation_scored.jsonl"
    if not data_path.exists():
        print("ERROR: validation dataset not found")
        sys.exit(1)

    with open(data_path) as f:
        cases = [json.loads(line) for line in f]

    print("=" * 80)
    print("PHASE 2: MEASUREMENT SWEEP — 28 Properties × 30 Cases")
    print("=" * 80)
    print(f"Dataset: {data_path.name} ({len(cases)} cases)")

    # Initialize IntegratedJudge
    engine = os.getenv("JUDGE_ENGINE", "gemini")
    print(f"Engine: {engine}")

    try:
        from llm_judge.integrated_judge import IntegratedJudge

        judge = IntegratedJudge(engine=engine)
    except Exception as exc:
        print(f"ERROR: Could not initialize IntegratedJudge: {exc}")
        print("Set JUDGE_ENGINE=deterministic for a dry run without LLM calls")
        sys.exit(1)

    # Run all cases
    all_results = []
    print(f"\nEvaluating {len(cases)} cases...\n")

    for i, case in enumerate(cases):
        cid = case["case_id"]

        request = PredictRequest(
            conversation=[
                Message(role=msg["role"], content=msg["content"])
                for msg in case["conversation"]
            ],
            candidate_answer=case["candidate_answer"],
            rubric_id=case["rubric_id"],
        )

        start = time.time()
        try:
            enriched = judge.evaluate_enriched(request, case_id=cid)
            result = enriched.to_dict()
            result["_case_id"] = cid
            result["_human_decision"] = case["human_decision"]
            result["_human_scores"] = case["human_scores"]
            result["_elapsed_ms"] = round((time.time() - start) * 1000, 1)
            all_results.append(result)
            status = "✓" if result["decision"] == case["human_decision"] else "✗"
            print(
                f"  [{i+1:2d}/30] {cid} {status} "
                f"judge={result['decision']:<4} human={case['human_decision']:<4} "
                f"coverage={result.get('detection_coverage', '?')[:30]} "
                f"({result['_elapsed_ms']:.0f}ms)"
            )
        except Exception as exc:
            print(f"  [{i+1:2d}/30] {cid} ERROR: {str(exc)[:60]}")
            all_results.append(
                {
                    "_case_id": cid,
                    "_human_decision": case["human_decision"],
                    "_error": str(exc)[:200],
                }
            )

    # Save raw results
    out_path = Path(__file__).parent / "measurement_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nRaw results → {out_path}")

    # =====================================================================
    # Build detection summary table
    # =====================================================================
    print(f"\n{'='*80}")
    print("DETECTION SUMMARY TABLE")
    print(f"{'='*80}\n")

    # Collect all property names from results
    property_names = set()
    for result in all_results:
        if "property_evidence" in result:
            property_names.update(result["property_evidence"].keys())

    # Build metrics per property
    from llm_judge.property_config import load_property_config

    registry = load_property_config()

    metrics: dict[str, PropertyMetric] = {}
    for prop_name, prop in registry.all_properties.items():
        metrics[prop_name] = PropertyMetric(
            property_id=prop.id,
            property_name=prop_name,
            category=prop.category,
        )

    # Evaluate each case against each property
    for result in all_results:
        if "_error" in result:
            continue

        case_data = None
        for c in cases:
            if c["case_id"] == result["_case_id"]:
                case_data = c
                break
        if case_data is None:
            continue

        for prop_name, metric in metrics.items():
            should_flag = _is_problem_case(case_data, prop_name, metric.category)
            if should_flag is None:
                # Can't evaluate this property against this case
                continue

            metric.cases_evaluated += 1
            did_flag = _has_flags(result, prop_name)

            if should_flag and did_flag:
                metric.detected += 1
            elif should_flag and not did_flag:
                metric.missed += 1
            elif not should_flag and did_flag:
                metric.false_alarm += 1
                if len(metric.sample_flags) < 3:
                    metric.sample_flags.append(result["_case_id"])
            else:
                metric.correct_pass += 1

    # Print summary table
    categories = [
        ("faithfulness", "Category 1 — Faithfulness & Grounding"),
        ("semantic_quality", "Category 2 — Semantic Quality"),
        ("safety", "Category 3 — Safety & Compliance"),
        ("task_fidelity", "Category 4 — Task Fidelity"),
        ("robustness", "Category 5 — Robustness"),
        ("performance", "Category 6 — Performance"),
    ]

    header = (
        f"{'#':<5} {'Property':<25} {'Cases':<7} {'Detect':<8} {'Missed':<8} "
        f"{'False':<8} {'Recall':<10} {'Precis':<10} {'F1':<8} {'Finding'}"
    )
    print(header)
    print("-" * len(header))

    for cat_id, cat_label in categories:
        print(f"\n{cat_label}")
        cat_metrics = [(n, m) for n, m in metrics.items() if m.category == cat_id]
        cat_metrics.sort(key=lambda x: x[1].property_id)

        for prop_name, m in cat_metrics:
            recall_str = f"{m.recall:.0%}" if m.recall is not None else "—"
            prec_str = f"{m.precision:.0%}" if m.precision is not None else "—"
            f1_str = f"{m.f1:.0%}" if m.f1 is not None else "—"

            if m.cases_evaluated == 0:
                finding = "No ground truth for this property in dataset"
            elif (
                m.recall is not None
                and m.recall >= 0.8
                and m.precision is not None
                and m.precision >= 0.8
            ):
                finding = "Ready for calibration"
            elif m.recall is not None and m.recall < 0.5:
                finding = "Low recall — needs method improvement"
            elif m.precision is not None and m.precision < 0.5:
                finding = f"Low precision — FPs: {', '.join(m.sample_flags)}"
            elif m.recall is None and m.precision is None:
                finding = "Calibration diagnostic (batch only)"
            else:
                finding = "Needs more data for calibration"

            print(
                f"{m.property_id:<5} {prop_name:<25} {m.cases_evaluated:<7} "
                f"{m.detected:<8} {m.missed:<8} {m.false_alarm:<8} "
                f"{recall_str:<10} {prec_str:<10} {f1_str:<8} {finding}"
            )

    # =====================================================================
    # Overall summary
    # =====================================================================
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")

    total_cases = len([r for r in all_results if "_error" not in r])
    errors = len([r for r in all_results if "_error" in r])
    decision_matches = sum(
        1
        for r in all_results
        if "_error" not in r and r["decision"] == r["_human_decision"]
    )

    print(f"  Cases evaluated: {total_cases}/{len(cases)}")
    if errors:
        print(f"  Errors: {errors}")
    if total_cases > 0:
        print(
            f"  Decision agreement: {decision_matches}/{total_cases} ({decision_matches/total_cases*100:.0f}%)"
        )
    else:
        print("  Decision agreement: no cases evaluated (check API key)")
    print("  Properties enabled: 28/28 (100%)")
    print(
        f"  Properties with ground truth: {sum(1 for m in metrics.values() if m.cases_evaluated > 0)}"
    )
    print(
        f"  Properties without ground truth: {sum(1 for m in metrics.values() if m.cases_evaluated == 0)}"
    )

    # Properties ready for calibration
    ready = [
        (n, m)
        for n, m in metrics.items()
        if m.recall is not None
        and m.recall >= 0.7
        and m.precision is not None
        and m.precision >= 0.7
    ]
    if ready:
        print("\n  Properties ready for gating (Recall≥70%, Precision≥70%):")
        for n, m in ready:
            print(
                f"    {m.property_id} {n}: R={m.recall:.0%} P={m.precision:.0%} F1={m.f1:.0%}"
            )

    # Properties needing attention
    needs_work = [
        (n, m)
        for n, m in metrics.items()
        if m.cases_evaluated > 0
        and (
            (m.recall is not None and m.recall < 0.5)
            or (m.precision is not None and m.precision < 0.5)
        )
    ]
    if needs_work:
        print("\n  Properties needing improvement:")
        for n, m in needs_work:
            r = f"R={m.recall:.0%}" if m.recall is not None else "R=—"
            p = f"P={m.precision:.0%}" if m.precision is not None else "P=—"
            print(f"    {m.property_id} {n}: {r} {p}")

    # Properties with no ground truth
    no_gt = [(n, m) for n, m in metrics.items() if m.cases_evaluated == 0]
    if no_gt:
        print(f"\n  Properties with no ground truth in dataset ({len(no_gt)}):")
        for n, m in no_gt:
            print(f"    {m.property_id} {n} ({m.category})")

    print("\n  Next steps:")
    print(f"    1. Review per-case results in {out_path.name}")
    print("    2. Properties ready for gating → set gate_mode: auto-gated in config")
    print("    3. Properties needing work → investigate detection method")
    print(
        "    4. Properties with no ground truth → add test cases for those categories"
    )
    print(f"{'='*80}")


if __name__ == "__main__":
    run_sweep()
