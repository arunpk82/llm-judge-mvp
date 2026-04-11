"""
Benchmark Metrics (EPIC 7.16).

Computes precision, recall, F1, accuracy per property and overall
response-level metrics from benchmark run results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_judge.benchmarks.runner import BenchmarkRunResult, PropertyResult


@dataclass
class ClassificationMetrics:
    """Standard binary classification metrics."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
            "total": self.total,
        }


@dataclass
class BenchmarkMetricsResult:
    """Complete metrics from a benchmark evaluation."""

    benchmark_name: str
    cases_evaluated: int
    elapsed_seconds: float
    response_level: ClassificationMetrics
    per_property: dict[str, ClassificationMetrics] = field(default_factory=dict)
    per_model: dict[str, ClassificationMetrics] = field(default_factory=dict)
    per_task_type: dict[str, ClassificationMetrics] = field(default_factory=dict)
    per_hallucination_type: dict[str, int] = field(default_factory=dict)
    error_count: int = 0
    fire_rates: dict[str, dict[str, int]] = field(default_factory=dict)
    properties_skipped: dict[str, str] = field(default_factory=dict)
    diagnostic_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "cases_evaluated": self.cases_evaluated,
            "elapsed_seconds": self.elapsed_seconds,
            "error_count": self.error_count,
            "response_level": self.response_level.to_dict(),
            "per_property": {pid: m.to_dict() for pid, m in self.per_property.items()},
            "per_model": {model: m.to_dict() for model, m in self.per_model.items()},
            "per_task_type": {tt: m.to_dict() for tt, m in self.per_task_type.items()},
        }


def _compute_binary_metrics(
    results: list[dict[str, Any]],
    predicted_key: str = "predicted",
    expected_key: str = "expected",
    positive_label: str = "fail",
) -> ClassificationMetrics:
    """Compute binary classification metrics from a list of result dicts."""
    m = ClassificationMetrics()
    for r in results:
        pred = r.get(predicted_key, "pass")
        exp = r.get(expected_key, "pass")
        if pred == positive_label and exp == positive_label:
            m.tp += 1
        elif pred == positive_label and exp != positive_label:
            m.fp += 1
        elif pred != positive_label and exp == positive_label:
            m.fn += 1
        else:
            m.tn += 1
    return m


def _property_metrics(
    results: list[PropertyResult],
    positive_label: str = "fail",
) -> ClassificationMetrics:
    """Compute metrics from PropertyResult list."""
    m = ClassificationMetrics()
    for r in results:
        pred = str(r.predicted)
        exp = str(r.expected)
        if pred == positive_label and exp == positive_label:
            m.tp += 1
        elif pred == positive_label and exp != positive_label:
            m.fp += 1
        elif pred != positive_label and exp == positive_label:
            m.fn += 1
        else:
            m.tn += 1
    return m


def compute_metrics(run_result: BenchmarkRunResult) -> BenchmarkMetricsResult:
    """Compute all metrics from a benchmark run."""

    # Response-level metrics
    response_metrics = _compute_binary_metrics(run_result.response_level_results)

    # Per-property metrics
    per_property: dict[str, ClassificationMetrics] = {}
    for pid, results in run_result.property_results.items():
        if results:
            per_property[pid] = _property_metrics(results)

    # Per-model metrics (response level)
    model_groups: dict[str, list[dict[str, Any]]] = {}
    for r in run_result.response_level_results:
        model = r.get("model", "unknown")
        model_groups.setdefault(model, []).append(r)
    per_model = {
        model: _compute_binary_metrics(results)
        for model, results in model_groups.items()
    }

    # Per-task-type metrics (response level)
    task_groups: dict[str, list[dict[str, Any]]] = {}
    for r in run_result.response_level_results:
        tt = r.get("task_type", "unknown")
        task_groups.setdefault(tt, []).append(r)
    per_task_type = {
        tt: _compute_binary_metrics(results) for tt, results in task_groups.items()
    }

    return BenchmarkMetricsResult(
        benchmark_name=run_result.benchmark_name,
        cases_evaluated=run_result.cases_evaluated,
        elapsed_seconds=run_result.elapsed_seconds,
        response_level=response_metrics,
        per_property=per_property,
        per_model=per_model,
        per_task_type=per_task_type,
        error_count=len(run_result.errors),
        fire_rates=run_result.fire_rates,
        properties_skipped=run_result.properties_skipped,
        diagnostic_results=run_result.diagnostic_results,
    )
