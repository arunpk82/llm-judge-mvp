"""CAP-2 sub-capability event coverage.

Five portal sub-capabilities (Rule loading, Pattern compilation,
Input matching, Evidence capture, Result emission) must each fire
on a happy CAP-2 invocation. Pattern compilation is SOFT — its
duration_ms is ``0.0`` because the work happens inside
``load_plan_for_rubric``; only the fire-rate signal is observable
at this layer (D1 forbids refactoring rules/engine.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

CAP2_TIMED_SUB_CAPS = (
    "rule_loading",
    "input_matching",
    "evidence_capture",
    "result_emission",
)
CAP2_SOFT_SUB_CAPS = ("pattern_compilation",)


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="cap2-subcap-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        caller_id="cap2-subcap-test",
    )


def _matching(
    logs: Sequence[Mapping[str, Any]],
    event: str,
    capability_id: str,
    sub_capability_id: str,
) -> list[Mapping[str, Any]]:
    return [
        log
        for log in logs
        if log.get("event") == event
        and log.get("capability_id") == capability_id
        and log.get("sub_capability_id") == sub_capability_id
    ]


@pytest.mark.parametrize("sub_cap", CAP2_TIMED_SUB_CAPS)
def test_cap2_subcap_started_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_started", "CAP-2", sub_cap)
    assert len(matches) == 1, (
        f"CAP-2.{sub_cap}: expected one sub_capability_started, got {len(matches)}"
    )


@pytest.mark.parametrize("sub_cap", CAP2_TIMED_SUB_CAPS)
def test_cap2_subcap_completed_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_completed", "CAP-2", sub_cap)
    assert len(matches) == 1, (
        f"CAP-2.{sub_cap}: expected one sub_capability_completed, got {len(matches)}"
    )
    assert matches[0]["status"] == "success"
    assert matches[0]["duration_ms"] >= 0.0


@pytest.mark.parametrize("sub_cap", CAP2_SOFT_SUB_CAPS)
def test_cap2_soft_subcap_fires_with_zero_duration(
    runner: PlatformRunner, sub_cap: str
) -> None:
    """SOFT boundaries (e.g., pattern_compilation) emit fire-rate
    signals with duration_ms=0.0 — the underlying work is timed
    under a sibling sub-cap, not separately."""
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    started = _matching(logs, "sub_capability_started", "CAP-2", sub_cap)
    completed = _matching(logs, "sub_capability_completed", "CAP-2", sub_cap)
    assert len(started) == 1
    assert len(completed) == 1
    assert completed[0]["duration_ms"] == 0.0
    assert completed[0]["status"] == "success"


def test_cap2_emits_no_subcap_failed_on_happy_path(
    runner: PlatformRunner,
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    failures = [
        log
        for log in logs
        if log.get("event") == "sub_capability_failed"
        and log.get("capability_id") == "CAP-2"
    ]
    assert not failures, f"unexpected CAP-2 sub-cap failures: {failures}"
