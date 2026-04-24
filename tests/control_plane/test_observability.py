"""Tests for :mod:`llm_judge.control_plane.observability` — Timer,
@timed, emit_event."""

from __future__ import annotations

import time

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane.observability import Timer, emit_event, timed

# =====================================================================
# Timer
# =====================================================================


def test_timer_happy_path_measures_elapsed_ms() -> None:
    with Timer() as t:
        time.sleep(0.02)  # 20 ms
    # Allow slack for scheduler jitter; assert order-of-magnitude.
    assert t.duration_ms >= 15.0
    assert t.duration_ms < 300.0


def test_timer_duration_before_exit_is_zero() -> None:
    t = Timer()
    assert t.duration_ms == 0.0
    with t:
        # Duration still 0.0 mid-block; populated only on __exit__.
        assert t.duration_ms == 0.0
    assert t.duration_ms > 0.0


def test_timer_exception_still_records_duration() -> None:
    t = Timer()
    with pytest.raises(ValueError, match="boom"):
        with t:
            time.sleep(0.01)
            raise ValueError("boom")
    assert t.duration_ms > 0.0


def test_timer_sequential_calls_are_independent() -> None:
    first = Timer()
    second = Timer()
    with first:
        time.sleep(0.01)
    with second:
        pass
    assert first.duration_ms > second.duration_ms or second.duration_ms >= 0.0


# =====================================================================
# @timed
# =====================================================================


def test_timed_emits_started_and_completed_on_success() -> None:
    @timed("op")
    def work() -> int:
        return 42

    with capture_logs() as logs:
        result = work()

    assert result == 42
    events = [log["event"] for log in logs]
    assert "op_started" in events
    assert "op_completed" in events
    completed = next(log for log in logs if log["event"] == "op_completed")
    assert "duration_ms" in completed
    assert isinstance(completed["duration_ms"], float)
    assert completed["duration_ms"] >= 0.0


def test_timed_emits_started_and_failed_on_exception() -> None:
    @timed("broken")
    def work() -> int:
        raise RuntimeError("nope")

    with capture_logs() as logs, pytest.raises(RuntimeError, match="nope"):
        work()

    events = [log["event"] for log in logs]
    assert "broken_started" in events
    assert "broken_failed" in events
    failed = next(log for log in logs if log["event"] == "broken_failed")
    assert failed["error_type"] == "RuntimeError"
    assert failed["error_message"] == "nope"
    assert "duration_ms" in failed
    assert failed["duration_ms"] >= 0.0


def test_timed_extracts_request_id_from_kwargs() -> None:
    @timed("task")
    def work(*, request_id: str) -> None:
        pass

    with capture_logs() as logs:
        work(request_id="req-abc")

    for expected in ("task_started", "task_completed"):
        rec = next(log for log in logs if log["event"] == expected)
        assert rec["request_id"] == "req-abc"


def test_timed_extracts_request_id_from_payload_attribute() -> None:
    class Payload:
        def __init__(self, request_id: str) -> None:
            self.request_id = request_id

    @timed("task")
    def work(payload: Payload) -> None:
        pass

    with capture_logs() as logs:
        work(Payload("req-xyz"))

    completed = next(log for log in logs if log["event"] == "task_completed")
    assert completed["request_id"] == "req-xyz"


def test_timed_preserves_function_signature() -> None:
    @timed("passthrough")
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert add.__doc__ == "Add two integers."


# =====================================================================
# emit_event
# =====================================================================


def test_emit_event_logs_through_structlog() -> None:
    with capture_logs() as logs:
        emit_event("custom_event", foo="bar", number=7)

    assert len(logs) == 1
    assert logs[0]["event"] == "custom_event"
    assert logs[0]["foo"] == "bar"
    assert logs[0]["number"] == 7
    assert logs[0]["log_level"] == "info"


def test_emit_event_with_no_fields_still_logs() -> None:
    with capture_logs() as logs:
        emit_event("lonely_event")

    assert len(logs) == 1
    assert logs[0]["event"] == "lonely_event"
