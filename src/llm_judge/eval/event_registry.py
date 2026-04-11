"""
Cross-capability Event Registry (EPIC 5.1).

Master artifact registry spanning ALL capabilities — not just eval runs.
Extends the existing run_registry pattern with typed events, relationship
links, and retry-on-failure.

Event types:
  eval_run              — evaluation completed (from eval/run.py)
  baseline_promotion    — baseline promoted (from eval/baseline.py)
  rule_change           — rule manifest changed (from rules/lifecycle.py)
  dataset_registration  — dataset registered/updated (from datasets/registry.py)
  drift_alert           — drift detected (from eval/drift.py)
  drift_response        — response action taken on drift issue

Design principles:
  - Append-only JSONL (immutable, auditable)
  - Every event has: event_type, timestamp, related_ids
  - Retry with exponential backoff on write failure
  - Backward compatible — existing run_registry.jsonl untouched
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from llm_judge.paths import state_root

logger = logging.getLogger(__name__)

EVENT_REGISTRY_SCHEMA_VERSION = "1.0"
DEFAULT_EVENT_REGISTRY_PATH = state_root() / "event_registry.jsonl"

# Valid event types — enforced at append time
VALID_EVENT_TYPES = frozenset(
    {
        "eval_run",
        "baseline_promotion",
        "rule_change",
        "dataset_registration",
        "drift_alert",
        "drift_response",
    }
)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class GovernanceEvent:
    """A single governance event in the cross-capability registry."""

    schema_version: str
    event_type: str
    timestamp: str

    # Who/what produced the event
    source: str  # e.g. "eval/run.py", "eval/baseline.py", "rules/lifecycle.py"
    actor: str  # e.g. "ci", "operator", "system"

    # Cross-capability relationship links
    related_ids: dict[str, str] = field(default_factory=dict)
    # Example: {"run_id": "pr-gate-...", "dataset_id": "math_basic",
    #           "rubric_id": "math_basic", "baseline_id": "20260305-..."}

    # Event-specific payload (varies by event_type)
    payload: dict[str, Any] = field(default_factory=dict)

    # Optional content hash of this event (for immutability verification)
    content_hash: str | None = None


def _append_with_retry(
    path: Path,
    line: str,
    *,
    max_retries: int = 3,
    base_delay: float = 0.5,
) -> bool:
    """
    Append a line to a file with exponential backoff retry.

    Returns True on success, False on all retries exhausted.
    P05: write failure is a governance risk — retry before alerting.
    """
    for attempt in range(max_retries):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "event_registry.write_retry",
                    extra={"attempt": attempt + 1, "delay": delay, "error": str(e)},
                )
                time.sleep(delay)
            else:
                logger.error(
                    "event_registry.write_failed",
                    extra={"path": str(path), "error": str(e), "attempts": max_retries},
                )
    return False


def append_event(
    *,
    event_type: str,
    source: str,
    actor: str = "system",
    related_ids: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
    registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH,
) -> GovernanceEvent | None:
    """
    Append a typed governance event to the cross-capability registry.

    Returns the event if written successfully, None if write failed.
    """
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Invalid event_type: '{event_type}'. "
            f"Valid types: {sorted(VALID_EVENT_TYPES)}"
        )

    event = GovernanceEvent(
        schema_version=EVENT_REGISTRY_SCHEMA_VERSION,
        event_type=event_type,
        timestamp=_utc_now_iso(),
        source=source,
        actor=actor,
        related_ids=related_ids or {},
        payload=payload or {},
    )

    line = json.dumps(asdict(event), sort_keys=True)
    ok = _append_with_retry(registry_path, line)

    if not ok:
        logger.error(
            "event_registry.event_lost",
            extra={"event_type": event_type, "source": source},
        )
        return None

    return event


# =====================================================================
# Query helpers
# =====================================================================


def iter_events(
    registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH,
) -> Iterable[dict[str, Any]]:
    """Iterate all events from the registry (append-order)."""
    if not registry_path.exists():
        return
    with registry_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def query_events(
    *,
    event_type: str | None = None,
    since: str | None = None,
    until: str | None = None,
    related_id_key: str | None = None,
    related_id_value: str | None = None,
    registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    """
    Query events with optional filters.

    Args:
        event_type: filter by event type (exact match)
        since: ISO timestamp — events at or after this time
        until: ISO timestamp — events at or before this time
        related_id_key: filter by presence of a specific related_id key
        related_id_value: combined with related_id_key, match exact value
    """
    results: list[dict[str, Any]] = []

    for event in iter_events(registry_path):
        # Type filter
        if event_type and event.get("event_type") != event_type:
            continue

        # Time range filter (ISO strings sort correctly)
        ts = event.get("timestamp", "")
        if since and ts < since:
            continue
        if until and ts > until:
            continue

        # Related ID filter
        if related_id_key:
            related = event.get("related_ids", {})
            if related_id_key not in related:
                continue
            if related_id_value and related.get(related_id_key) != related_id_value:
                continue

        results.append(event)

    return results


def query_events_in_window(
    *,
    center_timestamp: str,
    window_hours: float = 24.0,
    event_types: list[str] | None = None,
    registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    """
    Query events within a time window around a center timestamp.

    Used by causation analysis (EPIC 6.2) to find correlated events
    when drift is detected.
    """
    from datetime import timedelta

    center = datetime.fromisoformat(center_timestamp.replace("Z", "+00:00"))
    window_start = (
        (center - timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z")
    )
    window_end = (
        (center + timedelta(hours=window_hours)).isoformat().replace("+00:00", "Z")
    )

    results = query_events(
        since=window_start,
        until=window_end,
        registry_path=registry_path,
    )

    # Optional type filter
    if event_types:
        type_set = set(event_types)
        results = [e for e in results if e.get("event_type") in type_set]

    return results


# =====================================================================
# EPIC 5.2: Trace queries
# =====================================================================


def trace_by_run_id(
    *,
    run_id: str,
    registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH,
) -> list[dict[str, Any]]:
    """
    Find all events related to a specific run_id.

    Traces the complete event chain: dataset → rules → eval → baseline.
    """
    return query_events(
        related_id_key="run_id",
        related_id_value=run_id,
        registry_path=registry_path,
    )


# =====================================================================
# EPIC 8.2: Structured alerts
# =====================================================================

DEFAULT_ALERTS_DIR = state_root() / "alerts"


def write_structured_alert(
    *,
    alert_type: str,
    severity: str,
    persona: str,
    title: str,
    recommended_action: str,
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    alerts_dir: Path = DEFAULT_ALERTS_DIR,
) -> Path:
    """
    Write a structured alert to reports/alerts/ directory.

    Personas: engineer, qa_lead, product_owner
    Severities: critical, high, medium, low
    """
    import hashlib

    ts = _utc_now_iso()
    alert_id = (
        "alert-" + hashlib.sha256(f"{ts}{alert_type}{title}".encode()).hexdigest()[:12]
    )

    alert = {
        "schema_version": "1.0",
        "alert_id": alert_id,
        "alert_type": alert_type,
        "severity": severity,
        "persona": persona,
        "title": title,
        "recommended_action": recommended_action,
        "details": details or {},
        "correlation_id": correlation_id,
        "created_at": ts,
    }

    alerts_dir.mkdir(parents=True, exist_ok=True)
    alert_path = alerts_dir / f"{alert_id}.json"
    with alert_path.open("w", encoding="utf-8") as f:
        json.dump(alert, f, indent=2, sort_keys=True)

    return alert_path


# =====================================================================
# CLI
# =====================================================================


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Event registry queries for LLM-Judge."
    )
    sub = parser.add_subparsers(dest="cmd")

    p_query = sub.add_parser("query", help="Query events with filters")
    p_query.add_argument(
        "--since", default=None, help="ISO timestamp (events at or after)"
    )
    p_query.add_argument(
        "--until", default=None, help="ISO timestamp (events at or before)"
    )
    p_query.add_argument("--event-type", default=None, help="Filter by event type")
    p_query.add_argument("--registry", default=str(DEFAULT_EVENT_REGISTRY_PATH))
    p_query.add_argument("--limit", type=int, default=50)

    p_trace = sub.add_parser("trace", help="Trace all events for a run_id")
    p_trace.add_argument("run_id", help="Run ID to trace")
    p_trace.add_argument("--registry", default=str(DEFAULT_EVENT_REGISTRY_PATH))

    p_stats = sub.add_parser("stats", help="Show event registry statistics")
    p_stats.add_argument("--registry", default=str(DEFAULT_EVENT_REGISTRY_PATH))

    args = parser.parse_args()

    if args.cmd == "query":
        events = query_events(
            event_type=args.event_type,
            since=args.since,
            until=args.until,
            registry_path=Path(args.registry),
        )
        events = events[-args.limit :]
        print(f"\nEvents ({len(events)} results):")
        print("-" * 80)
        for e in events:
            print(
                f"  {e.get('timestamp', '?'):<22} "
                f"{e.get('event_type', '?'):<22} "
                f"{e.get('source', '?'):<24} "
                f"{e.get('actor', '?')}"
            )
        return 0

    if args.cmd == "trace":
        events = trace_by_run_id(
            run_id=args.run_id,
            registry_path=Path(args.registry),
        )
        print(f"\nTrace for run_id={args.run_id} ({len(events)} events):")
        print("-" * 80)
        for e in events:
            print(
                f"  {e.get('timestamp', '?'):<22} "
                f"{e.get('event_type', '?'):<22} "
                f"{e.get('source', '?')}"
            )
            related = e.get("related_ids", {})
            if related:
                print(f"    related: {related}")
        return 0

    if args.cmd == "stats":
        all_events = list(iter_events(Path(args.registry)))
        type_counts: dict[str, int] = {}
        for e in all_events:
            t = e.get("event_type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        print(f"\nEvent Registry Stats ({len(all_events)} total events):")
        print("-" * 40)
        for t, c in sorted(type_counts.items()):
            print(f"  {t:<24} {c}")
        if all_events:
            print(f"\n  First: {all_events[0].get('timestamp', '?')}")
            print(f"  Last:  {all_events[-1].get('timestamp', '?')}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
