"""Rich-formatted live progress + event capture for batch demos.

Subscribes to the Control Plane event bus and renders a header
panel, a live progress bar, per-case status lines, and a final
summary panel. While the renderer is attached, every event is
captured in memory (and flushed to a per-case ``events.jsonl``
when the case completes), giving Commit 11's HTML report a stable
event log without re-running the batch.

Reuses the CP-2 rich gotcha fixes:
  * ``Console(record=True, force_terminal=True)`` keeps styled
    segments in the record buffer when stdout is piped.
  * ``console.export_text(clear=False)`` before ``export_html()``
    lets both forms share the same buffer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from llm_judge.control_plane.event_bus import get_default_bus


class BatchTerminalRenderer:
    """Live-progress renderer + event capturer for a single batch run.

    Construction takes the ``output_dir`` so per-case events.jsonl
    files can land alongside the matching manifest.json. ``attach``
    subscribes to the default event bus; ``detach`` unsubscribes.
    Tests should call ``detach`` (or ``get_default_bus().clear()``)
    to keep handlers from leaking across cases.
    """

    def __init__(
        self,
        output_dir: Path,
        *,
        console: Console | None = None,
    ) -> None:
        self._output_dir = output_dir
        self._cases_root = output_dir / "cases"
        # ``record=True`` lets Commit 11 export the same console to HTML.
        # ``force_terminal=True`` keeps styling alive when stdout is piped.
        self.console = console or Console(record=True, force_terminal=True)

        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._batch_id: str | None = None

        # Captured event log, scoped per-case + globally.
        self._captured_events: list[dict[str, Any]] = []
        self._current_case_id: str | None = None
        self._current_case_events: list[dict[str, Any]] = []

        self._handlers = {
            "batch_started": self._on_batch_started,
            "batch_case_started": self._on_case_started,
            "batch_case_completed": self._on_case_completed,
            "batch_completed": self._on_batch_completed,
        }
        self._wildcard = self._on_any_event

    # --- public surface --------------------------------------------------

    def attach(self) -> None:
        """Subscribe handlers to the default event bus."""
        bus = get_default_bus()
        for event_type, handler in self._handlers.items():
            bus.subscribe(event_type, handler)
        bus.subscribe("*", self._wildcard)

    def detach(self) -> None:
        """Reverse :meth:`attach` — used by tests to keep handlers
        from leaking between cases.
        """
        bus = get_default_bus()
        for event_type, handler in self._handlers.items():
            bus.unsubscribe(event_type, handler)
        bus.unsubscribe("*", self._wildcard)

    @property
    def captured_events(self) -> list[dict[str, Any]]:
        """All events seen since :meth:`attach`. Read after the batch
        completes; the list is mutated in place during the run.
        """
        return self._captured_events

    # --- event handlers --------------------------------------------------

    def _on_any_event(self, event_type: str, **fields: Any) -> None:
        record: dict[str, Any] = {"event": event_type, **fields}
        self._captured_events.append(record)
        if self._current_case_id is not None:
            self._current_case_events.append(record)

    def _on_batch_started(self, event_type: str, **fields: Any) -> None:
        self._batch_id = fields.get("batch_id")
        total = int(fields.get("total_cases", 0) or 0)
        source = str(fields.get("source", ""))

        body = (
            f"[bold]batch_id[/bold]    {self._batch_id}\n"
            f"[bold]source[/bold]      {source}\n"
            f"[bold]total_cases[/bold] {total}"
        )
        self.console.print(
            Panel(body, title="Batch starting", style="cyan", expand=False)
        )
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
            TimeElapsedColumn(),
            console=self.console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task("Running batch", total=total)

    def _on_case_started(self, event_type: str, **fields: Any) -> None:
        self._current_case_id = str(fields.get("case_id"))
        self._current_case_events = []

    def _on_case_completed(self, event_type: str, **fields: Any) -> None:
        case_id = str(fields.get("case_id"))
        case_index = int(fields.get("case_index", 0) or 0)
        status = str(fields.get("status", "unknown"))
        duration_ms = float(fields.get("duration_ms", 0.0) or 0.0)

        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, advance=1)

        icon = {"success": "✓", "failure": "✗", "error": "!"}.get(
            status, "?"
        )
        colour = {
            "success": "green",
            "failure": "yellow",
            "error": "red",
        }.get(status, "white")
        line = (
            f"[ case {case_index:3d} ] [{colour}]{icon} "
            f"{status.upper():<7}[/{colour}]  "
            f"{duration_ms:>7.0f}ms  {case_id}"
        )
        self.console.print(line)

        # Flush this case's events to disk so the HTML report can
        # find them later even if the renderer is torn down early.
        case_dir = self._cases_root / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        events_path = case_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as f:
            for record in self._current_case_events:
                f.write(json.dumps(record, default=str) + "\n")

        self._current_case_id = None
        self._current_case_events = []

    def _on_batch_completed(self, event_type: str, **fields: Any) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

        total = int(fields.get("total_cases", 0) or 0)
        successful = int(fields.get("successful", 0) or 0)
        failed = int(fields.get("failed", 0) or 0)
        errored = int(fields.get("error", 0) or 0)
        duration_ms = float(fields.get("duration_ms", 0.0) or 0.0)

        body = (
            f"[bold]successful[/bold] [green]{successful}[/green]/{total}\n"
            f"[bold]failed[/bold]     [yellow]{failed}[/yellow]\n"
            f"[bold]errored[/bold]    [red]{errored}[/red]\n"
            f"[bold]duration[/bold]   {duration_ms:.0f}ms"
        )
        self.console.print(
            Panel(body, title="Batch complete", style="cyan", expand=False)
        )

        # Flush the global events log too, for cross-case tooling.
        events_path = self._output_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as f:
            for record in self._captured_events:
                f.write(json.dumps(record, default=str) + "\n")
