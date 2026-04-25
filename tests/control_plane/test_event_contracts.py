"""Contract tests for the 8 Control Plane event types.

Each event has a documented contract:
  * required fields and their types
  * no extra fields beyond the contract

These tests capture events through ``structlog.testing.capture_logs``
(every event flows through :func:`emit_event`, which logs via
structlog before dispatching to the bus). Field shapes are asserted
per event type.

Capturing through structlog rather than the bus gives us a stable
snapshot regardless of subscriber state between tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
from structlog.testing import capture_logs

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

# =====================================================================
# Event-field contracts — a single source of truth for the tests.
# Required fields must be present on every matching emission. Extra
# fields beyond these (plus the inherent "event", "log_level",
# "timestamp" structlog adds) trip an assertion.
# =====================================================================

_STRUCTLOG_META = {"event", "log_level", "timestamp"}

RUN_STARTED_REQUIRED = {
    "request_id": str,
    "timestamp": str,
    "caller_id": str,
    "platform_version": str,
}
RUN_COMPLETED_REQUIRED = {
    "request_id": str,
    "duration_ms": float,
    "status": str,
}
CAPABILITY_STARTED_REQUIRED = {
    "capability_id": str,
    "request_id": str,
    "timestamp": str,
}
CAPABILITY_COMPLETED_REQUIRED = {
    "capability_id": str,
    "request_id": str,
    "duration_ms": float,
    "status": str,
}
CAPABILITY_FAILED_REQUIRED = {
    "capability_id": str,
    "request_id": str,
    "duration_ms": float,
    "status": str,
    "error_type": str,
    "error_message": str,
}
YAML_LOAD_STARTED_REQUIRED = {
    "file_path": str,
}
YAML_LOAD_COMPLETED_REQUIRED = {
    "file_path": str,
    "duration_ms": float,
    "parser": str,
}
YAML_LOAD_FAILED_REQUIRED = {
    "file_path": str,
    "duration_ms": float,
    "parser": str,
    "error_type": str,
    "error_message": str,
}


def _first_event(
    logs: Sequence[Mapping[str, Any]], event_name: str
) -> Mapping[str, Any]:
    matches = [log for log in logs if log.get("event") == event_name]
    assert matches, f"no {event_name!r} event in captured logs"
    return matches[0]


def _assert_contract(
    log: Mapping[str, Any],
    required: Mapping[str, type[Any]],
    event_name: str,
) -> None:
    """Every required field present + correctly typed; no extras
    beyond structlog's meta keys."""
    for field, expected_type in required.items():
        assert field in log, (
            f"{event_name}: required field {field!r} missing. "
            f"Got: {sorted(log.keys())}"
        )
        assert isinstance(log[field], expected_type), (
            f"{event_name}.{field}: expected {expected_type.__name__}, "
            f"got {type(log[field]).__name__}"
        )

    # No extras beyond required + structlog meta. ``timestamp`` and
    # ``event`` may both be part of the required contract AND structlog
    # metadata — they are allowed in either case.
    allowed = set(required) | _STRUCTLOG_META
    extras = set(log) - allowed
    assert not extras, (
        f"{event_name}: unexpected extra fields {sorted(extras)}. "
        f"Expected ⊆ {sorted(allowed)}"
    )


# =====================================================================
# Runner-driven events (1..5 of 8): happy path + CAP-2 failure inject
# =====================================================================


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="contract-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _happy_payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        caller_id="contract-test",
    )


def test_run_started_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    event = _first_event(logs, "run_started")
    _assert_contract(event, RUN_STARTED_REQUIRED, "run_started")
    assert event["caller_id"] == "contract-test"
    assert event["platform_version"] == "contract-sha"


def test_run_completed_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    event = _first_event(logs, "run_completed")
    _assert_contract(event, RUN_COMPLETED_REQUIRED, "run_completed")
    assert event["duration_ms"] >= 0.0
    assert event["status"] in ("success", "partial")


def test_capability_started_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    cap1 = next(
        log
        for log in logs
        if log.get("event") == "capability_started"
        and log.get("capability_id") == "CAP-1"
    )
    _assert_contract(cap1, CAPABILITY_STARTED_REQUIRED, "capability_started")


def test_capability_completed_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    cap1 = next(
        log
        for log in logs
        if log.get("event") == "capability_completed"
        and log.get("capability_id") == "CAP-1"
    )
    _assert_contract(cap1, CAPABILITY_COMPLETED_REQUIRED, "capability_completed")
    assert cap1["status"] == "success"
    assert cap1["duration_ms"] >= 0.0


def test_capability_failed_contract(
    runner: PlatformRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inject a CAP-2 failure; assert capability_failed shape."""

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("cap2 injected failure")

    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap2", _boom)

    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    cap2_fail = next(
        log
        for log in logs
        if log.get("event") == "capability_failed"
        and log.get("capability_id") == "CAP-2"
    )
    _assert_contract(cap2_fail, CAPABILITY_FAILED_REQUIRED, "capability_failed")
    assert cap2_fail["status"] == "failure"
    assert cap2_fail["error_type"] == "RuntimeError"
    assert "cap2 injected failure" in cap2_fail["error_message"]
    assert cap2_fail["duration_ms"] >= 0.0


# =====================================================================
# YAML-load events (6..8 of 8): happy path on real rule plan + malformed
# =====================================================================


def test_yaml_load_started_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    event = _first_event(logs, "yaml_load_started")
    _assert_contract(event, YAML_LOAD_STARTED_REQUIRED, "yaml_load_started")


def test_yaml_load_completed_contract(runner: PlatformRunner) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())

    event = _first_event(logs, "yaml_load_completed")
    _assert_contract(event, YAML_LOAD_COMPLETED_REQUIRED, "yaml_load_completed")
    assert event["parser"] in ("pyyaml", "fallback")
    assert event["duration_ms"] >= 0.0


def test_yaml_load_failed_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed YAML file at the rule-plan path surfaces
    yaml_load_failed with the contract fields."""
    from llm_judge.rules.engine import load_plan_for_rubric

    monkeypatch.chdir(tmp_path)
    plan_dir = tmp_path / "configs" / "rules" / "fixture_rubric"
    plan_dir.mkdir(parents=True)
    # Malformed YAML: unclosed brace.
    (plan_dir / "v1.yaml").write_text("{unclosed: ", encoding="utf-8")

    with capture_logs() as logs:
        with pytest.raises(Exception):
            load_plan_for_rubric("fixture_rubric", "v1")

    event = _first_event(logs, "yaml_load_failed")
    _assert_contract(event, YAML_LOAD_FAILED_REQUIRED, "yaml_load_failed")
    assert event["parser"] in ("pyyaml", "fallback")
    assert event["error_type"]  # non-empty


# =====================================================================
# Cross-event coverage sanity: all 8 event types fire on a happy run
# =====================================================================


def test_happy_run_emits_every_expected_event_type(
    runner: PlatformRunner,
) -> None:
    """A single successful end-to-end run must emit at least one of
    each non-failure event type: run_started, run_completed,
    capability_started, capability_completed, yaml_load_started,
    yaml_load_completed. (Failure events tested separately.)"""
    expected = {
        "run_started",
        "run_completed",
        "capability_started",
        "capability_completed",
        "yaml_load_started",
        "yaml_load_completed",
    }
    with capture_logs() as logs:
        runner.run_single_evaluation(_happy_payload())
    seen = {log.get("event") for log in logs}
    missing = expected - seen
    assert not missing, f"happy run did not emit: {missing}"


def test_yaml_roundtrip_sanity(tmp_path: Path) -> None:
    """PyYAML is importable and loads valid YAML — baseline sanity
    so yaml_load_completed's parser=='pyyaml' branch really runs on
    this host (caught one bug in earlier work where pyyaml was
    absent from a test environment)."""
    path = tmp_path / "ok.yaml"
    path.write_text(yaml.safe_dump({"k": 1}), encoding="utf-8")
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == {"k": 1}
