#!/usr/bin/env python3
"""Control Plane demonstration.

Invokes PlatformRunner.run_single_evaluation end-to-end with a
hardcoded example, renders each capability event as it fires
using rich formatting, writes timestamped Markdown and HTML
artifacts to reports/demos/<ts>/.

Exit 0 if Runner completes (even if a capability failed cleanly
and CAP-5 wrote a degradation manifest).
Exit 1 if the demo itself breaks.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llm_judge.control_plane.event_bus import get_default_bus
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

DEMO_RESPONSE = "Is the sky blue?"
DEMO_SOURCE = "The sky appears blue due to Rayleigh scattering."

EVENT_TYPES = (
    "run_started",
    "run_completed",
    "capability_started",
    "capability_completed",
    "capability_failed",
    "yaml_load_started",
    "yaml_load_completed",
    "yaml_load_failed",
)

STATUS_GLYPHS = {
    "success": ("✓", "green"),
    "failure": ("✗", "red"),
    "skipped_upstream_failure": ("~", "yellow"),
}


def _fmt_ms(duration_ms: float | None) -> str:
    if duration_ms is None:
        return "—"
    if duration_ms >= 1000.0:
        return f"{duration_ms / 1000.0:.2f} s"
    return f"{duration_ms:.1f} ms"


def _stamp_detail(envelope: Any, capability_id: str) -> str:
    """Pick a short, human-readable stamp for the capability row."""
    if capability_id == "CAP-1":
        reg = getattr(envelope, "dataset_registry_id", None)
        input_hash = getattr(envelope, "input_hash", None)
        parts = []
        if reg:
            parts.append(f"dataset={reg}")
        if input_hash:
            parts.append(f"hash={input_hash[:10]}…")
        return ", ".join(parts) if parts else "—"
    if capability_id == "CAP-2":
        rule_set = getattr(envelope, "rule_set_version", None)
        rules_fired = getattr(envelope, "rules_fired", None) or []
        parts = []
        if rule_set:
            parts.append(f"rule_set={rule_set}")
        parts.append(f"rules_fired={len(rules_fired)}")
        return ", ".join(parts)
    if capability_id == "CAP-7":
        return "hallucination pipeline"
    if capability_id == "CAP-5":
        return "manifest attestation"
    return "—"


def _subscribe_all(
    captured: list[tuple[str, dict[str, Any]]],
) -> None:
    bus = get_default_bus()
    for event_type in EVENT_TYPES:
        def _handler(
            event_type: str,
            _captured: list[tuple[str, dict[str, Any]]] = captured,
            **fields: Any,
        ) -> None:
            _captured.append((event_type, fields))

        bus.subscribe(event_type, _handler)


def _render_opening(console: Console, request_id: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    body = (
        f"[bold]response:[/bold] {DEMO_RESPONSE}\n"
        f"[bold]source:[/bold]   {DEMO_SOURCE}\n"
        f"[bold]request:[/bold]  {request_id}\n"
        f"[bold]started:[/bold]  {ts}"
    )
    console.print(Panel(body, title="LLM-Judge Control Plane — demo", style="cyan"))


def _render_capability_table(
    console: Console,
    envelope: Any,
) -> None:
    table = Table(title="Capability invocations", show_lines=False)
    table.add_column("duration", justify="right")
    table.add_column("capability")
    table.add_column("status")
    table.add_column("detail", overflow="fold")

    for record in envelope.integrity:
        glyph, colour = STATUS_GLYPHS.get(record.status, ("?", "white"))
        status_cell = f"[{colour}]{glyph} {record.status}[/{colour}]"
        detail = _stamp_detail(envelope, record.capability_id)
        if record.status == "failure" and record.error_type:
            detail = (
                f"[red]{record.error_type}: "
                f"{(record.error_message or '')[:80]}[/red]"
            )
        table.add_row(
            _fmt_ms(record.duration_ms),
            record.capability_id,
            status_cell,
            detail,
        )
    console.print(table)


def _render_envelope(console: Console, envelope: Any) -> None:
    chain = " → ".join(envelope.capability_chain) or "(empty)"
    try:
        signature_state = "valid" if envelope.verify_signature() else "INVALID"
    except Exception:
        signature_state = "present"

    total = len(envelope.integrity)
    failures = sum(1 for r in envelope.integrity if r.status == "failure")
    if failures == 0:
        integrity_line = f"[green]all {total} capabilities succeeded[/green]"
    else:
        integrity_line = f"[red]{failures} of {total} capabilities failed[/red]"

    body = (
        f"[bold]capability_chain:[/bold] {chain}\n"
        f"[bold]schema_version:[/bold]   {envelope.schema_version}\n"
        f"[bold]signature:[/bold]        {signature_state}\n"
        f"[bold]integrity:[/bold]        {integrity_line}"
    )
    console.print(Panel(body, title="Provenance envelope", style="magenta"))


def _render_outputs(
    console: Console,
    manifest_id: str,
    md_path: Path,
    html_path: Path,
) -> None:
    body = (
        f"[bold]manifest_id:[/bold]  {manifest_id}\n"
        f"[bold]demo_report.md:[/bold]   {md_path}\n"
        f"[bold]demo_report.html:[/bold] {html_path}"
    )
    console.print(Panel(body, title="Outputs", style="blue"))


def _render_closing(
    console: Console,
    duration_ms: float | None,
    integrity_complete: bool,
) -> None:
    status = (
        "[green]SUCCESS[/green]" if integrity_complete else "[yellow]PARTIAL[/yellow]"
    )
    body = (
        f"[bold]status:[/bold]   {status}\n"
        f"[bold]duration:[/bold] {_fmt_ms(duration_ms)}"
    )
    console.print(Panel(body, title="Run complete", style="cyan"))


def main() -> int:
    # ``force_terminal=True`` keeps styled segments in the record buffer
    # even when stdout is piped (e.g. ``make demo | tee``). Without it,
    # export_html() produces an empty <code> block.
    console = Console(record=True, force_terminal=True)
    captured: list[tuple[str, dict[str, Any]]] = []
    _subscribe_all(captured)

    request_id = "demo-" + datetime.now().strftime("%Y%m%dT%H%M%S")
    payload = SingleEvaluationRequest(
        response=DEMO_RESPONSE,
        source=DEMO_SOURCE,
        request_id=request_id,
        caller_id="demo_platform",
    )

    _render_opening(console, request_id)

    runner = PlatformRunner()
    try:
        result = runner.run_single_evaluation(payload)
    except Exception as exc:
        console.print(
            f"[red]Runner raised before completing: "
            f"{type(exc).__name__}: {exc}[/red]"
        )
        console.print(f"[dim]captured {len(captured)} events before failure[/dim]")
        console.print_exception()
        return 1

    run_completed = next(
        (fields for et, fields in captured if et == "run_completed"),
        {},
    )
    total_duration = run_completed.get("duration_ms")

    _render_capability_table(console, result.envelope)
    _render_envelope(console, result.envelope)

    ts = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    demo_dir = Path("reports/demos") / ts
    demo_dir.mkdir(parents=True, exist_ok=True)
    md_path = demo_dir / "demo_report.md"
    html_path = demo_dir / "demo_report.html"

    _render_outputs(console, result.manifest_id, md_path, html_path)
    _render_closing(console, total_duration, result.integrity.complete)

    # export_text/export_html default to clear=True; pass clear=False on
    # the first call so the record buffer survives for the second.
    md_path.write_text(console.export_text(clear=False), encoding="utf-8")
    html_path.write_text(console.export_html(), encoding="utf-8")

    console.print()
    console.print(f"[dim]artifacts → {demo_dir}[/dim]")

    return 0


if __name__ == "__main__":
    console_for_error = Console()
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover - demo safety net
        console_for_error.print(f"[red]Demo failed: {exc}[/red]")
        console_for_error.print_exception()
        sys.exit(1)
