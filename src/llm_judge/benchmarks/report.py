"""
Benchmark Report (EPIC 7.16).

Formats benchmark metrics as a comparison against published baselines.
Outputs text report for console/CI and dict for JSON serialization.
"""

from __future__ import annotations

from typing import Any

from llm_judge.benchmarks import BenchmarkMetadata
from llm_judge.benchmarks.metrics import BenchmarkMetricsResult

# All 28 property names — single source of truth for display
PROPERTY_NAMES: dict[str, str] = {
    "1.1": "Groundedness",
    "1.2": "Ungrounded Claims",
    "1.3": "Citation Verification",
    "1.4": "Attribution Accuracy",
    "1.5": "Fabrication Detection",
    "2.1": "Relevance",
    "2.2": "Clarity",
    "2.3": "Correctness",
    "2.4": "Tone",
    "2.5": "Completeness",
    "2.6": "Coherence",
    "2.7": "Depth/Nuance",
    "3.1": "Toxicity & Bias",
    "3.2": "Instruction Boundary",
    "3.3": "PII & Data Leakage",
    "4.1": "Instruction Following",
    "4.2": "Format & Structure",
    "5.1": "Position Bias",
    "5.2": "Length Bias",
    "5.3": "Self-Preference Bias",
    "5.4": "Consistency",
    "5.5": "Adversarial Resilience",
    "5.6": "Edge Case Handling",
    "5.7": "Reproducibility",
    "6.1": "Latency & Cost",
    "6.2": "Confidence Calibration",
    "6.3": "Explainability",
    "6.4": "Judge Reasoning Fidelity",
}


def _prop_label(pid: str) -> str:
    """Get display label for a property ID."""
    name = PROPERTY_NAMES.get(pid, "")
    return f"{pid} {name}" if name else pid


def generate_report(
    metrics: BenchmarkMetricsResult,
    metadata: BenchmarkMetadata,
) -> dict[str, Any]:
    """Generate benchmark comparison report."""
    # Build baseline comparison
    baselines: list[dict[str, Any]] = []
    for bl in metadata.published_baselines:
        baselines.append(
            {
                "method": bl.method,
                "metric": bl.metric,
                "published_value": round(bl.value, 4),
                "source": bl.source,
            }
        )

    report: dict[str, Any] = {
        "benchmark": metadata.name,
        "version": metadata.version,
        "citation": metadata.citation,
        "cases_evaluated": metrics.cases_evaluated,
        "elapsed_seconds": metrics.elapsed_seconds,
        "error_count": metrics.error_count,
        "response_level": metrics.response_level.to_dict(),
        "per_property": {
            _prop_label(pid): m.to_dict() for pid, m in metrics.per_property.items()
        },
        "per_model": {model: m.to_dict() for model, m in metrics.per_model.items()},
        "per_task_type": {tt: m.to_dict() for tt, m in metrics.per_task_type.items()},
        "published_baselines": baselines,
        "supported_properties": [_prop_label(p) for p in metadata.supported_properties],
    }

    return report


def format_report_text(
    metrics: BenchmarkMetricsResult,
    metadata: BenchmarkMetadata,
) -> str:
    """Format benchmark report as readable text for console/CI output."""
    from llm_judge.benchmarks.runner import ALL_PROPERTY_IDS

    lines: list[str] = []
    lines.append(f"{'=' * 70}")
    lines.append(f"BENCHMARK REPORT: {metadata.name} v{metadata.version}")
    lines.append(f"{'=' * 70}")
    lines.append(f"Citation: {metadata.citation}")
    lines.append(f"Cases evaluated: {metrics.cases_evaluated}")
    lines.append(f"Elapsed: {metrics.elapsed_seconds}s")
    lines.append(f"Errors: {metrics.error_count}")
    lines.append("")

    # Response-level results
    rl = metrics.response_level
    lines.append("RESPONSE-LEVEL RESULTS")
    lines.append(f"  Precision: {rl.precision:.4f}")
    lines.append(f"  Recall:    {rl.recall:.4f}")
    lines.append(f"  F1:        {rl.f1:.4f}")
    lines.append(f"  Accuracy:  {rl.accuracy:.4f}")
    lines.append(f"  (TP={rl.tp} FP={rl.fp} TN={rl.tn} FN={rl.fn})")
    lines.append("")

    # Published baseline comparison
    lines.append("PUBLISHED BASELINES (comparison)")
    for bl in metadata.published_baselines:
        delta = rl.f1 - bl.value
        direction = "+" if delta >= 0 else ""
        lines.append(
            f"  {bl.method}: {bl.metric}={bl.value:.3f} "
            f"[ours: {rl.f1:.3f}, delta: {direction}{delta:.3f}]"
        )
    lines.append("")

    # Per-property results — ALL 28
    lines.append("ALL 28 PROPERTIES")
    lines.append(
        f"  {'Property':<35} {'Status':<18} {'Fired':>6} {'Total':>6} {'Rate':>6}  {'P':>6} {'R':>6} {'F1':>6}  {'TP':>4} {'FP':>4} {'TN':>5} {'FN':>4}"
    )
    lines.append(
        f"  {'-'*35} {'-'*18} {'-'*6} {'-'*6} {'-'*6}  {'-'*6} {'-'*6} {'-'*6}  {'-'*4} {'-'*4} {'-'*5} {'-'*4}"
    )

    diag = metrics.diagnostic_results

    for pid in ALL_PROPERTY_IDS:
        label = _prop_label(pid)
        fr = metrics.fire_rates.get(pid, {})
        fired = fr.get("fail", 0)
        total = fr.get("total", 0)
        rate = f"{fired/total:.1%}" if total > 0 else "—"
        fired_str = str(fired) if total > 0 else "—"
        total_str = str(total) if total > 0 else "—"

        # Cat 5: Show diagnostic metric instead of P/R/F1
        if pid in diag and pid.startswith("5."):
            d = diag[pid]
            decision = d.get("decision", "—")
            detail = ""
            if "consistency_pct" in d:
                detail = f"consistency={d['consistency_pct']}%"
            elif "length_flag_correlation" in d:
                detail = f"corr={d['length_flag_correlation']}"
            elif "resilience_pct" in d:
                detail = f"resilience={d['resilience_pct']}%"
            elif "edge_cases_handled" in d:
                detail = f"{d['edge_cases_handled']}/{d['edge_cases_total']} handled"
            elif "reproducibility_pct" in d:
                detail = f"repro={d['reproducibility_pct']}%"
            elif "note" in d:
                detail = d["note"]
            status = f"DIAG:{decision.upper()}"
            lines.append(
                f"  {label:<35} {status:<18} {detail:>6} {'':>6} {'':>6}"
                f"  {'—':>6} {'—':>6} {'—':>6}"
                f"  {'—':>4} {'—':>4} {'—':>5} {'—':>4}"
            )
            continue

        # Cat 6.1, 6.2: Show metric values
        if pid in diag and pid.startswith("6."):
            d = diag[pid]
            decision = d.get("decision", "—")
            detail = ""
            if "avg_latency_ms" in d:
                detail = f"avg={d['avg_latency_ms']}ms"
            elif "monotonic" in d:
                detail = f"mono={'Y' if d['monotonic'] else 'N'}"
            status = f"DIAG:{decision.upper()}"
            lines.append(
                f"  {label:<35} {status:<18} {detail:>6} {'':>6} {'':>6}"
                f"  {'—':>6} {'—':>6} {'—':>6}"
                f"  {'—':>4} {'—':>4} {'—':>5} {'—':>4}"
            )
            continue

        if pid in metrics.per_property and metrics.per_property[pid].total > 0:
            m = metrics.per_property[pid]
            lines.append(
                f"  {label:<35} {'MEASURED':<18} {fired_str:>6} {total_str:>6} {rate:>6}"
                f"  {m.precision:>6.3f} {m.recall:>6.3f} {m.f1:>6.3f}"
                f"  {m.tp:>4} {m.fp:>4} {m.tn:>5} {m.fn:>4}"
            )
        elif total > 0:
            lines.append(
                f"  {label:<35} {'RAN (no GT)':<18} {fired_str:>6} {total_str:>6} {rate:>6}"
                f"  {'—':>6} {'—':>6} {'—':>6}"
                f"  {'—':>4} {'—':>4} {'—':>5} {'—':>4}"
            )
        else:
            skip_reason = metrics.properties_skipped.get(pid, "")
            if pid.startswith("2."):
                skip_reason = "requires --with-llm"
            elif pid in ("6.3", "6.4"):
                skip_reason = "requires --with-llm"
            lines.append(
                f"  {label:<35} {skip_reason:<18} {'—':>6} {'—':>6} {'—':>6}"
                f"  {'—':>6} {'—':>6} {'—':>6}"
                f"  {'—':>4} {'—':>4} {'—':>5} {'—':>4}"
            )
    lines.append("")

    # Diagnostic detail section
    if diag:
        lines.append("DIAGNOSTIC DETAILS")
        for pid in sorted(diag.keys()):
            d = diag[pid]
            label = _prop_label(pid)
            decision = d.get("decision", "—")
            lines.append(f"  {label}: {decision.upper()}")
            for k, v in d.items():
                if k != "decision":
                    lines.append(f"    {k}: {v}")
        lines.append("")

    # Per-model results
    if metrics.per_model:
        lines.append("PER-MODEL RESULTS")
        for model, m in sorted(metrics.per_model.items()):
            lines.append(
                f"  {model}: F1={m.f1:.3f} P={m.precision:.3f} "
                f"R={m.recall:.3f} (n={m.total})"
            )
        lines.append("")

    # Per-task-type results
    if metrics.per_task_type:
        lines.append("PER-TASK-TYPE RESULTS")
        for tt, m in sorted(metrics.per_task_type.items()):
            lines.append(
                f"  {tt}: F1={m.f1:.3f} P={m.precision:.3f} "
                f"R={m.recall:.3f} (n={m.total})"
            )
        lines.append("")

    lines.append(f"{'=' * 70}")
    return "\n".join(lines)
