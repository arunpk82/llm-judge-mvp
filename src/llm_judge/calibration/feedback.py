"""
Feedback Loop (L4 Step 10).

The dual data flow that makes the system learn:
  LLM evaluations → human adjudication → disagreement analysis →
  calibration improvements → better LLM evaluations.

Components:
  - Disagreement analysis: find systematic patterns in LLM-human disagreements
  - Dimension weakness detection: which dimensions does the LLM misjudge?
  - Calibration recommendations: actionable suggestions for improvement
  - Feedback report: structured output for the engineering team
"""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DimensionFeedback:
    """Feedback for a single scoring dimension."""
    dimension: str
    total_cases: int = 0
    agreements: int = 0
    llm_higher: int = 0  # LLM scored higher than human
    llm_lower: int = 0   # LLM scored lower than human
    avg_deviation: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.agreements / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def bias_direction(self) -> str:
        if self.llm_higher > self.llm_lower * 1.5:
            return "lenient"  # LLM is too generous
        if self.llm_lower > self.llm_higher * 1.5:
            return "strict"   # LLM is too harsh
        return "balanced"


@dataclass
class FeedbackReport:
    """Complete feedback analysis from human adjudication data."""
    timestamp: str
    total_resolved: int = 0
    decision_agreement_rate: float = 0.0
    dimension_feedback: dict[str, DimensionFeedback] = field(default_factory=dict)
    systematic_patterns: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


def analyze_feedback(
    *,
    adjudication_data: list[dict[str, Any]],
    dimensions: list[str] | None = None,
) -> FeedbackReport:
    """
    Analyze human adjudication results to find systematic disagreement patterns.

    Args:
        adjudication_data: resolved adjudication cases (from queue.jsonl)
        dimensions: scoring dimensions to analyze
    """
    if dimensions is None:
        dimensions = ["relevance", "clarity", "correctness", "tone"]

    resolved = [d for d in adjudication_data if d.get("state") == "resolved"]

    report = FeedbackReport(
        timestamp=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        total_resolved=len(resolved),
    )

    if not resolved:
        report.recommendations.append("No resolved adjudication cases — run evaluations and resolve low-confidence cases first")
        return report

    # Decision agreement
    agreements = sum(
        1 for d in resolved
        if d.get("llm_decision") == d.get("human_decision")
    )
    report.decision_agreement_rate = round(agreements / len(resolved), 4) if resolved else 0.0

    # Per-dimension analysis
    for dim in dimensions:
        df = DimensionFeedback(dimension=dim)
        deviations: list[float] = []

        for case in resolved:
            llm_scores = case.get("llm_scores", {})
            human_scores = case.get("human_scores", {})

            if not isinstance(llm_scores, dict) or not isinstance(human_scores, dict):
                continue

            llm_val = llm_scores.get(dim)
            human_val = human_scores.get(dim)

            if llm_val is None or human_val is None:
                continue

            llm_val = int(llm_val)
            human_val = int(human_val)
            df.total_cases += 1
            deviation = llm_val - human_val
            deviations.append(float(deviation))

            if abs(deviation) <= 1:
                df.agreements += 1
            elif deviation > 0:
                df.llm_higher += 1
            else:
                df.llm_lower += 1

        if deviations:
            df.avg_deviation = round(statistics.mean(deviations), 2)

        report.dimension_feedback[dim] = df

    # Detect systematic patterns
    for dim, df in report.dimension_feedback.items():
        if df.total_cases < 3:
            continue

        if df.accuracy < 0.5:
            report.systematic_patterns.append(
                f"Dimension '{dim}' has low agreement ({df.accuracy:.0%}) — "
                f"LLM consistently misjudges this dimension"
            )

        if df.bias_direction == "lenient":
            report.systematic_patterns.append(
                f"Dimension '{dim}' shows leniency bias — LLM scores higher "
                f"than human {df.llm_higher}/{df.total_cases} cases"
            )
        elif df.bias_direction == "strict":
            report.systematic_patterns.append(
                f"Dimension '{dim}' shows strictness bias — LLM scores lower "
                f"than human {df.llm_lower}/{df.total_cases} cases"
            )

    if report.decision_agreement_rate < 0.7:
        report.systematic_patterns.append(
            f"Overall decision agreement is low ({report.decision_agreement_rate:.0%}) — "
            f"consider recalibrating the judge"
        )

    # Generate recommendations
    _generate_recommendations(report)

    return report


def _generate_recommendations(report: FeedbackReport) -> None:
    """Generate actionable recommendations from the feedback analysis."""
    if report.decision_agreement_rate < 0.6:
        report.recommendations.append(
            "CRITICAL: Decision agreement below 60%. Recalibrate the LLM judge "
            "against the golden dataset before further production use."
        )
    elif report.decision_agreement_rate < 0.8:
        report.recommendations.append(
            "Decision agreement at {:.0%} — review the system prompt for "
            "ambiguous scoring criteria.".format(report.decision_agreement_rate)
        )

    # Dimension-specific recommendations
    weakest = None
    weakest_acc = 1.0
    for dim, df in report.dimension_feedback.items():
        if df.total_cases > 0 and df.accuracy < weakest_acc:
            weakest = dim
            weakest_acc = df.accuracy

    if weakest and weakest_acc < 0.6:
        df = report.dimension_feedback[weakest]
        report.recommendations.append(
            f"Weakest dimension: '{weakest}' ({weakest_acc:.0%} accuracy). "
            f"Add more examples of {weakest} scoring to the system prompt."
        )

        if df.bias_direction == "lenient":
            report.recommendations.append(
                f"Dimension '{weakest}' leniency fix: add explicit failure "
                f"examples to the prompt — 'Score {weakest} as 1-2 when...'"
            )
        elif df.bias_direction == "strict":
            report.recommendations.append(
                f"Dimension '{weakest}' strictness fix: add examples of "
                f"acceptable responses — 'Score {weakest} as 4-5 when...'"
            )

    if not report.recommendations:
        report.recommendations.append(
            "No critical issues detected. Continue monitoring via the "
            "adjudication queue and periodic calibration runs."
        )


def save_feedback_report(
    report: FeedbackReport,
    output_dir: Path = Path("reports/feedback"),
) -> Path:
    """Save feedback report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = {
        "schema_version": "1.0",
        "timestamp": report.timestamp,
        "total_resolved": report.total_resolved,
        "decision_agreement_rate": report.decision_agreement_rate,
        "dimensions": {
            dim: {
                "total_cases": df.total_cases,
                "accuracy": round(df.accuracy, 4),
                "llm_higher": df.llm_higher,
                "llm_lower": df.llm_lower,
                "avg_deviation": df.avg_deviation,
                "bias_direction": df.bias_direction,
            }
            for dim, df in report.dimension_feedback.items()
        },
        "systematic_patterns": report.systematic_patterns,
        "recommendations": report.recommendations,
    }

    path = output_dir / f"feedback_{report.timestamp.replace(':', '-')}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    return path
