"""Tests for batch_aggregation.aggregate_batch.

Synthetic BatchResult + event-log fixtures exercise the rollup
arithmetic without going through the real Runner. The horizontal
0/N invariant (CAP-6, CAP-10) and the empty-batch edge case are
specifically asserted because those are the surprising cases users
will see in the HTML report.
"""

from __future__ import annotations

from pathlib import Path

from llm_judge.control_plane.batch_aggregation import (
    HORIZONTAL_CAPABILITIES,
    VERTICAL_CAPABILITIES,
    BatchAggregation,
    aggregate_batch,
)
from llm_judge.control_plane.batch_runner import BatchResult, CaseResult
from llm_judge.control_plane.envelope import CapabilityIntegrityRecord


def _success_record(cap: str, duration_ms: float) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=cap, status="success", duration_ms=duration_ms
    )


def _failure_record(
    cap: str, duration_ms: float, error: str = "Boom"
) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=cap,
        status="failure",
        error_type=error,
        error_message=f"{cap} failed",
        duration_ms=duration_ms,
    )


def _case(
    case_id: str,
    integrity: list[CapabilityIntegrityRecord],
    *,
    duration_ms: float = 100.0,
    had_failure: bool = False,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        request_id=case_id,
        manifest_path=Path("/tmp/manifest.json"),
        integrity=integrity,
        duration_ms=duration_ms,
        had_failure=had_failure,
        had_error=False,
    )


def _batch_result(case_results: list[CaseResult], **overrides) -> BatchResult:
    base = dict(
        batch_id="batch-test",
        source="benchmark:synthetic",
        total_cases=len(case_results),
        successful_cases=sum(
            1 for c in case_results if not c.had_failure and not c.had_error
        ),
        failed_cases=sum(1 for c in case_results if c.had_failure),
        error_cases=sum(1 for c in case_results if c.had_error),
        duration_ms=sum(c.duration_ms for c in case_results),
        case_results=case_results,
    )
    base.update(overrides)
    return BatchResult(**base)


def test_returns_batch_aggregation() -> None:
    br = _batch_result([])
    agg = aggregate_batch(br)
    assert isinstance(agg, BatchAggregation)
    assert agg.batch_id == "batch-test"
    assert agg.total_cases == 0


def test_zero_case_batch_yields_zero_durations() -> None:
    agg = aggregate_batch(_batch_result([]))
    assert agg.case_duration_p50_ms == 0.0
    assert agg.case_duration_p95_ms == 0.0
    assert agg.case_duration_max_ms == 0.0
    for cap in VERTICAL_CAPABILITIES:
        assert agg.capability_success_rate[cap] == 0.0
        assert agg.capability_mean_duration_ms[cap] == 0.0
        assert agg.capability_p95_duration_ms[cap] == 0.0


def test_capability_success_rate_with_mixed_outcomes() -> None:
    cases = [
        _case(
            "c1",
            [_success_record("CAP-1", 10.0), _success_record("CAP-2", 5.0)],
        ),
        _case(
            "c2",
            [_success_record("CAP-1", 12.0), _failure_record("CAP-2", 3.0)],
            had_failure=True,
        ),
        _case(
            "c3",
            [_success_record("CAP-1", 11.0), _success_record("CAP-2", 6.0)],
        ),
    ]
    agg = aggregate_batch(_batch_result(cases))
    assert agg.capability_success_rate["CAP-1"] == 1.0  # 3/3
    assert abs(agg.capability_success_rate["CAP-2"] - (2 / 3)) < 1e-9


def test_capability_durations_use_records_only() -> None:
    cases = [
        _case("c1", [_success_record("CAP-1", 10.0)]),
        _case("c2", [_success_record("CAP-1", 20.0)]),
        _case("c3", [_success_record("CAP-1", 30.0)]),
    ]
    agg = aggregate_batch(_batch_result(cases))
    assert abs(agg.capability_mean_duration_ms["CAP-1"] - 20.0) < 1e-9
    # p95 of [10, 20, 30]: k = 2 * 0.95 = 1.9, lo=1, hi=2, frac=0.9
    # → s[1] + 0.9 * (s[2] - s[1]) = 20 + 0.9 * 10 = 29.0
    assert abs(agg.capability_p95_duration_ms["CAP-1"] - 29.0) < 1e-9


def test_sub_capability_fire_count_and_success_rate() -> None:
    events = [
        {
            "event": "sub_capability_started",
            "capability_id": "CAP-1",
            "sub_capability_id": "reception",
        },
        {
            "event": "sub_capability_completed",
            "capability_id": "CAP-1",
            "sub_capability_id": "reception",
            "status": "success",
        },
        {
            "event": "sub_capability_started",
            "capability_id": "CAP-1",
            "sub_capability_id": "reception",
        },
        {
            "event": "sub_capability_completed",
            "capability_id": "CAP-1",
            "sub_capability_id": "reception",
            "status": "success",
        },
        # one failure that should not contribute to success rate
        {
            "event": "sub_capability_started",
            "capability_id": "CAP-1",
            "sub_capability_id": "validation",
        },
        {
            "event": "sub_capability_failed",
            "capability_id": "CAP-1",
            "sub_capability_id": "validation",
            "status": "failure",
        },
    ]
    agg = aggregate_batch(_batch_result([]), events=events)
    assert agg.sub_capability_fire_count["CAP-1"]["reception"] == 2
    assert agg.sub_capability_fire_count["CAP-1"]["validation"] == 1
    assert agg.sub_capability_success_rate["CAP-1"]["reception"] == 1.0
    assert agg.sub_capability_success_rate["CAP-1"]["validation"] == 0.0


def test_sub_capability_skipped_counted_separately() -> None:
    events = [
        {
            "event": "sub_capability_skipped",
            "capability_id": "CAP-1",
            "sub_capability_id": "discovery",
            "reason": "single_eval_does_not_query_registry",
        },
        {
            "event": "sub_capability_skipped",
            "capability_id": "CAP-1",
            "sub_capability_id": "discovery",
            "reason": "single_eval_does_not_query_registry",
        },
    ]
    agg = aggregate_batch(_batch_result([]), events=events)
    assert agg.sub_capability_skipped_count["CAP-1"]["discovery"] == 2
    # Skipped sub-caps must NOT appear in fire_count.
    assert (
        "discovery"
        not in agg.sub_capability_fire_count.get("CAP-1", {})
    )


def test_horizontal_engagement_reports_zero_when_no_events() -> None:
    """Honest 0/N — CAP-6 and CAP-10 are not wired today (D4)."""
    agg = aggregate_batch(_batch_result([]))
    for horiz in HORIZONTAL_CAPABILITIES:
        assert agg.horizontal_engagement[horiz] == 0


def test_horizontal_engagement_counts_when_events_seen() -> None:
    """If we ever wire CAP-10, the aggregator must surface it."""
    events = [
        {
            "event": "capability_started",
            "capability_id": "CAP-10",
            "request_id": "r1",
        },
        {
            "event": "capability_completed",
            "capability_id": "CAP-10",
            "request_id": "r1",
        },
    ]
    agg = aggregate_batch(_batch_result([]), events=events)
    assert agg.horizontal_engagement["CAP-10"] == 2
    assert agg.horizontal_engagement["CAP-6"] == 0


def test_case_duration_percentiles() -> None:
    cases = [
        _case("c1", [], duration_ms=100.0),
        _case("c2", [], duration_ms=200.0),
        _case("c3", [], duration_ms=300.0),
        _case("c4", [], duration_ms=400.0),
        _case("c5", [], duration_ms=500.0),
    ]
    agg = aggregate_batch(_batch_result(cases))
    assert agg.case_duration_max_ms == 500.0
    # p50 of [100..500] with len=5, k = 4*0.5 = 2 → s[2] = 300.
    assert abs(agg.case_duration_p50_ms - 300.0) < 1e-9
    # p95 of len=5: k = 4*0.95 = 3.8, s[3] + 0.8*(s[4]-s[3]) = 400+0.8*100 = 480
    assert abs(agg.case_duration_p95_ms - 480.0) < 1e-9


def test_aggregation_ignores_malformed_events() -> None:
    """Events without the expected fields are skipped, not crashed."""
    events = [
        {"event": "sub_capability_started"},  # missing capability_id
        {"event": "sub_capability_started", "capability_id": "CAP-1"},  # missing sub_id
        {"capability_id": "CAP-1", "sub_capability_id": "x"},  # missing event name
        # one valid event so the aggregation has something to find
        {
            "event": "sub_capability_started",
            "capability_id": "CAP-2",
            "sub_capability_id": "rule_loading",
        },
    ]
    agg = aggregate_batch(_batch_result([]), events=events)
    assert agg.sub_capability_fire_count == {"CAP-2": {"rule_loading": 1}}
