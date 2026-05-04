"""Batch rollups: capability + sub-capability statistics.

Computes a :class:`BatchAggregation` from a :class:`BatchResult` and
an optional event log captured during the batch run. The HTML report
renders the aggregation; the CLI driver passes both to it.

Honest-gap surfacing (D4): the horizontal capabilities (CAP-6, CAP-10)
report engagement counts derived from emitted events. They are
expected to be ``0`` today — the wiring is out of scope for CP-3 —
and the report renders that 0/N as the truth, not as a missing field.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_judge.control_plane.batch_runner import BatchResult
from llm_judge.control_plane.capability_registry import CAPABILITY_REGISTRY

# Vertical capabilities derived from CAPABILITY_REGISTRY (CP-F9
# closure, L1-Pkt-B). Adding a capability requires a registry
# change; this rollup picks it up automatically rather than via a
# parallel hard-coded enumeration. CAP-7 stays capability-level
# (D1: no layer cascade refactor); the other three carry
# sub-capability instrumentation.
VERTICAL_CAPABILITIES: tuple[str, ...] = tuple(
    spec.capability_id for spec in CAPABILITY_REGISTRY
)

# Horizontal capabilities reported as 0/N until they are wired (D4).
HORIZONTAL_CAPABILITIES: tuple[str, ...] = ("CAP-6", "CAP-10")


class BatchAggregation(BaseModel):
    """Computed rollups for a batch run."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    total_cases: int

    capability_success_rate: dict[str, float] = Field(default_factory=dict)
    capability_mean_duration_ms: dict[str, float] = Field(default_factory=dict)
    capability_p95_duration_ms: dict[str, float] = Field(default_factory=dict)

    sub_capability_fire_count: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )
    sub_capability_success_rate: dict[str, dict[str, float]] = Field(
        default_factory=dict
    )
    sub_capability_skipped_count: dict[str, dict[str, int]] = Field(
        default_factory=dict
    )

    horizontal_engagement: dict[str, int] = Field(default_factory=dict)

    total_duration_ms: float = 0.0
    case_duration_p50_ms: float = 0.0
    case_duration_p95_ms: float = 0.0
    case_duration_max_ms: float = 0.0


def _percentile(values: list[float], p: float) -> float:
    """Crude percentile: sort + index. ``0.0`` on empty lists.

    Uses linear interpolation between sorted samples to keep small
    batches from collapsing every quantile to the same value.
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _capability_rollups(
    batch_result: BatchResult,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute success rate + mean + p95 duration per vertical capability."""
    success_rate: dict[str, float] = {}
    mean_ms: dict[str, float] = {}
    p95_ms: dict[str, float] = {}

    for cap_id in VERTICAL_CAPABILITIES:
        statuses: list[str] = []
        durations: list[float] = []
        for case in batch_result.case_results:
            for record in case.integrity:
                if record.capability_id != cap_id:
                    continue
                statuses.append(record.status)
                if record.duration_ms is not None:
                    durations.append(record.duration_ms)

        if statuses:
            success_rate[cap_id] = sum(
                1 for s in statuses if s == "success"
            ) / len(statuses)
        else:
            success_rate[cap_id] = 0.0

        if durations:
            mean_ms[cap_id] = statistics.fmean(durations)
            p95_ms[cap_id] = _percentile(durations, 0.95)
        else:
            mean_ms[cap_id] = 0.0
            p95_ms[cap_id] = 0.0

    return success_rate, mean_ms, p95_ms


def _sub_capability_rollups(
    events: Iterable[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, dict[str, float]],
    dict[str, dict[str, int]],
]:
    """Walk the event log and tally sub-capability fire rates.

    Returns ``(fire_count, success_rate, skipped_count)`` keyed by
    ``capability_id`` then ``sub_capability_id``.
    """
    fire: dict[str, dict[str, int]] = {}
    started: dict[str, dict[str, int]] = {}
    completed: dict[str, dict[str, int]] = {}
    skipped: dict[str, dict[str, int]] = {}

    for event in events:
        ev_name = event.get("event")
        cap_id = event.get("capability_id")
        sub_id = event.get("sub_capability_id")
        if not (
            isinstance(ev_name, str)
            and isinstance(cap_id, str)
            and isinstance(sub_id, str)
        ):
            continue

        if ev_name == "sub_capability_started":
            fire.setdefault(cap_id, {}).setdefault(sub_id, 0)
            fire[cap_id][sub_id] += 1
            started.setdefault(cap_id, {}).setdefault(sub_id, 0)
            started[cap_id][sub_id] += 1
        elif ev_name == "sub_capability_completed":
            status = event.get("status")
            if status == "success":
                completed.setdefault(cap_id, {}).setdefault(sub_id, 0)
                completed[cap_id][sub_id] += 1
        elif ev_name == "sub_capability_skipped":
            skipped.setdefault(cap_id, {}).setdefault(sub_id, 0)
            skipped[cap_id][sub_id] += 1

    success_rate: dict[str, dict[str, float]] = {}
    for cap_id, sub_counts in started.items():
        success_rate[cap_id] = {}
        for sub_id, n_started in sub_counts.items():
            n_completed = completed.get(cap_id, {}).get(sub_id, 0)
            success_rate[cap_id][sub_id] = (
                n_completed / n_started if n_started else 0.0
            )

    return fire, success_rate, skipped


def _horizontal_engagement(
    events: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Count any event tagged with a horizontal capability_id.

    Today this is expected to be ``0`` for every horizontal — that's
    the honest 0/N the report surfaces (D4: wiring out of scope).
    """
    counts: dict[str, int] = dict.fromkeys(HORIZONTAL_CAPABILITIES, 0)
    for event in events:
        cap_id = event.get("capability_id")
        if cap_id in counts:
            counts[cap_id] += 1
    return counts


def aggregate_batch(
    batch_result: BatchResult,
    events: Iterable[dict[str, Any]] | None = None,
) -> BatchAggregation:
    """Compute :class:`BatchAggregation` from a :class:`BatchResult`.

    ``events`` is an iterable of structlog event dicts captured during
    the batch run (each dict has at minimum ``event``, plus the
    fields the emitter passed). When omitted, sub-capability and
    horizontal-engagement rollups are empty/zero — the report will
    show that as 'no event log captured' rather than fake zeros.
    """
    events_list = list(events) if events is not None else []

    success_rate, mean_ms, p95_ms = _capability_rollups(batch_result)
    fire, sub_success, skipped = _sub_capability_rollups(events_list)
    horizontal = _horizontal_engagement(events_list)

    case_durations = [c.duration_ms for c in batch_result.case_results]

    return BatchAggregation(
        batch_id=batch_result.batch_id,
        total_cases=batch_result.total_cases,
        capability_success_rate=success_rate,
        capability_mean_duration_ms=mean_ms,
        capability_p95_duration_ms=p95_ms,
        sub_capability_fire_count=fire,
        sub_capability_success_rate=sub_success,
        sub_capability_skipped_count=skipped,
        horizontal_engagement=horizontal,
        total_duration_ms=batch_result.duration_ms,
        case_duration_p50_ms=_percentile(case_durations, 0.50),
        case_duration_p95_ms=_percentile(case_durations, 0.95),
        case_duration_max_ms=max(case_durations) if case_durations else 0.0,
    )
