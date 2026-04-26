"""Aggregated HTML report for a batch run.

Renders BatchResult + BatchAggregation through a rich Console and
exports to both ``aggregated_report.html`` and ``aggregated_report.md``.
The HTML carries the live ANSI styling thanks to
``Console(record=True, force_terminal=True)``; the Markdown is the
plain-text dump of the same buffer.

Sections (per CP-3 packet specification):
  1. Header — batch_id, source, run timestamp, duration
  2. Aggregate summary — success / failure / error + duration percentiles
  3. Per-capability table — verticals + horizontals with fire counts
     and portal RAG status (D4: horizontals at 0/N is the honest gap)
  4. Per-sub-capability decomposition — CAP-1, CAP-2, CAP-5 only;
     CAP-7 carries an "instrumentation deferred" note (D1)
  5. Per-case table — one row per case with status, duration, manifest
  6. Artifacts — paths the user can follow up on
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_judge.control_plane.batch_aggregation import (
    HORIZONTAL_CAPABILITIES,
    BatchAggregation,
)
from llm_judge.control_plane.batch_runner import BatchResult

# Capability metadata for the report. ``role`` is the portal role
# label; ``rag`` is the portal RAG status today; ``sub_caps`` is the
# documented sub-capability ordering for the decomposition table.
# RAG values: green (instrumented), amber (capability-level only),
# red (not engaged on the Runner path today — D4 honest gap).
_CAPABILITY_META: dict[str, dict[str, object]] = {
    "CAP-1": {
        "name": "Dataset Ingestion",
        "role": "vertical",
        "rag": "green",
        "sub_caps": (
            "reception",
            "validation",
            "registration",
            "hashing",
            "lineage_tracking",
            "discovery",
        ),
    },
    "CAP-2": {
        "name": "Rule Evaluation",
        "role": "vertical",
        "rag": "green",
        "sub_caps": (
            "rule_loading",
            "pattern_compilation",
            "input_matching",
            "evidence_capture",
            "result_emission",
        ),
    },
    "CAP-7": {
        "name": "Hallucination Detection",
        "role": "vertical",
        "rag": "amber",
        "sub_caps": (),
    },
    "CAP-5": {
        "name": "Artifact Governance",
        "role": "horizontal (wired)",
        "rag": "green",
        "sub_caps": (
            "envelope_reception",
            "manifest_composition",
            "persistence",
            "lineage_linking",
            "query_interface",
        ),
    },
    "CAP-6": {
        "name": "Drift Monitoring",
        "role": "horizontal",
        "rag": "red",
        "sub_caps": (),
    },
    "CAP-10": {
        "name": "Calibration",
        "role": "horizontal",
        "rag": "red",
        "sub_caps": (),
    },
}

_RAG_STYLES = {
    "green": "[green]●[/green] green",
    "amber": "[yellow]●[/yellow] amber",
    "red": "[red]●[/red] red",
}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _render_header(
    console: Console, batch_result: BatchResult, output_dir: Path
) -> None:
    body = (
        f"[bold]batch_id[/bold]    {batch_result.batch_id}\n"
        f"[bold]source[/bold]      {batch_result.source}\n"
        f"[bold]total_cases[/bold] {batch_result.total_cases}\n"
        f"[bold]duration[/bold]    {batch_result.duration_ms:.0f}ms\n"
        f"[bold]rendered_at[/bold] {_utc_now_iso()}\n"
        f"[bold]output_dir[/bold]  {output_dir}"
    )
    console.print(Panel(body, title="Batch report", style="cyan"))


def _render_aggregate(
    console: Console,
    batch_result: BatchResult,
    aggregation: BatchAggregation,
) -> None:
    body = (
        f"[bold]successful[/bold] [green]{batch_result.successful_cases}[/green]"
        f"/{batch_result.total_cases}\n"
        f"[bold]failed[/bold]     [yellow]{batch_result.failed_cases}[/yellow]\n"
        f"[bold]errored[/bold]    [red]{batch_result.error_cases}[/red]\n"
        f"\n"
        f"[bold]p50 case duration[/bold] {aggregation.case_duration_p50_ms:.0f}ms\n"
        f"[bold]p95 case duration[/bold] {aggregation.case_duration_p95_ms:.0f}ms\n"
        f"[bold]max case duration[/bold] {aggregation.case_duration_max_ms:.0f}ms"
    )
    console.print(Panel(body, title="Summary", style="green"))


def _render_capability_table(
    console: Console, aggregation: BatchAggregation
) -> None:
    table = Table(
        title="Per-capability rollups",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Capability", style="bold")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Portal RAG")
    table.add_column("Success rate", justify="right")
    table.add_column("Mean ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Engagement", justify="right")

    for cap_id, meta in _CAPABILITY_META.items():
        is_horizontal = cap_id in HORIZONTAL_CAPABILITIES
        if is_horizontal:
            engagement = aggregation.horizontal_engagement.get(cap_id, 0)
            engagement_label = (
                f"{engagement}/{aggregation.total_cases}"
                f" {'[red](not wired)[/red]' if engagement == 0 else ''}"
            ).strip()
            success = "—"
            mean = "—"
            p95 = "—"
        else:
            sr = aggregation.capability_success_rate.get(cap_id, 0.0)
            mean_v = aggregation.capability_mean_duration_ms.get(cap_id, 0.0)
            p95_v = aggregation.capability_p95_duration_ms.get(cap_id, 0.0)
            success = f"{sr * 100:.0f}%"
            mean = f"{mean_v:.1f}"
            p95 = f"{p95_v:.1f}"
            # Vertical engagement = number of cases that recorded the cap.
            engaged = sum(
                1
                for c in aggregation.sub_capability_fire_count.get(cap_id, {})
                .values()
            )
            engagement_label = (
                f"{aggregation.total_cases}/{aggregation.total_cases}"
                if aggregation.total_cases
                else "—"
            )
            if cap_id == "CAP-7":
                engagement_label = (
                    f"{engagement_label} [yellow](capability-level only)[/yellow]"
                )
            del engaged  # not used directly; kept for future per-cap scoping
        table.add_row(
            cap_id,
            str(meta["name"]),
            str(meta["role"]),
            _RAG_STYLES[str(meta["rag"])],
            success,
            mean,
            p95,
            engagement_label,
        )
    console.print(table)


def _render_subcap_decomposition(
    console: Console, aggregation: BatchAggregation
) -> None:
    for cap_id, meta in _CAPABILITY_META.items():
        sub_caps = meta["sub_caps"]
        if not isinstance(sub_caps, tuple):
            continue
        if not sub_caps:
            if cap_id == "CAP-7":
                console.print(
                    Panel(
                        "CAP-7 is rendered at capability level only. "
                        "Sub-capability instrumentation (Input prep, "
                        "Layer cascade, Rubric application, Dimension "
                        "contract, Aggregation, Result emission, "
                        "MiniCheck integration) is [yellow]deferred to "
                        "the CAP-7 completion packet[/yellow] — "
                        "refactoring the L1→L2→L3→L4 cascade is out of "
                        "CP-3 scope (D1).",
                        title="CAP-7 — Hallucination Detection",
                        style="yellow",
                    )
                )
            continue

        table = Table(
            title=f"{cap_id} — {meta['name']} sub-capabilities",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Sub-capability")
        table.add_column("Fire count", justify="right")
        table.add_column("Success rate", justify="right")
        table.add_column("Skipped count", justify="right")
        table.add_column("Note")

        for sub_id in sub_caps:
            fire = aggregation.sub_capability_fire_count.get(cap_id, {}).get(
                sub_id, 0
            )
            sr = aggregation.sub_capability_success_rate.get(cap_id, {}).get(
                sub_id, 0.0
            )
            skipped = aggregation.sub_capability_skipped_count.get(
                cap_id, {}
            ).get(sub_id, 0)
            note = ""
            if cap_id == "CAP-2" and sub_id == "pattern_compilation":
                note = "[dim]SOFT (timing inside rule_loading)[/dim]"
            if skipped > 0 and fire == 0:
                note = "[dim]not engaged on this Runner path[/dim]"
            sr_label = (
                f"{sr * 100:.0f}%"
                if fire > 0
                else "[dim]—[/dim]"
            )
            fire_label = str(fire) if fire > 0 else "[dim]0[/dim]"
            skipped_label = (
                str(skipped) if skipped > 0 else "[dim]0[/dim]"
            )
            table.add_row(sub_id, fire_label, sr_label, skipped_label, note)
        console.print(table)


def _render_per_case_table(
    console: Console, batch_result: BatchResult, output_dir: Path
) -> None:
    if not batch_result.case_results:
        console.print("[dim](no cases)[/dim]")
        return

    table = Table(
        title="Per-case results",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", justify="right")
    table.add_column("case_id")
    table.add_column("Status")
    table.add_column("Duration", justify="right")
    table.add_column("Manifest")

    for i, case in enumerate(batch_result.case_results):
        if case.had_error:
            status = "[red]ERROR[/red]"
        elif case.had_failure:
            status = "[yellow]FAILURE[/yellow]"
        else:
            status = "[green]SUCCESS[/green]"
        try:
            rel = case.manifest_path.relative_to(output_dir)
        except ValueError:
            rel = case.manifest_path
        table.add_row(
            str(i),
            case.case_id,
            status,
            f"{case.duration_ms:.0f}ms",
            str(rel),
        )
    console.print(table)


def _render_artifacts(
    console: Console, output_dir: Path, batch_result: BatchResult
) -> None:
    body = (
        f"[bold]batch_manifest.json[/bold]    {output_dir / 'batch_manifest.json'}\n"
        f"[bold]events.jsonl[/bold]            {output_dir / 'events.jsonl'}\n"
        f"[bold]aggregated_report.html[/bold]  {output_dir / 'aggregated_report.html'}\n"
        f"[bold]aggregated_report.md[/bold]    {output_dir / 'aggregated_report.md'}\n"
        f"[bold]per-case manifests[/bold]      "
        f"{output_dir / 'cases'}/<case_id>/manifest.json   "
        f"({len(batch_result.case_results)} files)\n"
        f"[bold]per-case events[/bold]         "
        f"{output_dir / 'cases'}/<case_id>/events.jsonl"
    )
    console.print(Panel(body, title="Artifacts", style="blue"))


def render_batch_report(
    *,
    batch_result: BatchResult,
    aggregation: BatchAggregation,
    output_dir: Path,
    html_path: Path,
    md_path: Path,
    console: Console | None = None,
) -> None:
    """Render the full report and write HTML + Markdown to disk.

    A pre-warmed ``Console`` (e.g. one already used by the terminal
    renderer) can be passed in via ``console`` so we extend its
    record buffer rather than starting fresh — the HTML then carries
    the live progress text along with the report.
    """
    console = console or Console(record=True, force_terminal=True)

    console.print()
    _render_header(console, batch_result, output_dir)
    _render_aggregate(console, batch_result, aggregation)
    _render_capability_table(console, aggregation)
    _render_subcap_decomposition(console, aggregation)
    _render_per_case_table(console, batch_result, output_dir)
    _render_artifacts(console, output_dir, batch_result)

    # export_text/export_html default to clear=True; pass clear=False
    # on the first call so the record buffer survives for the second.
    md_path.write_text(console.export_text(clear=False), encoding="utf-8")
    html_path.write_text(console.export_html(), encoding="utf-8")
