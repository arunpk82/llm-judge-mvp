"""CAP-5 sub-capability event coverage.

Five portal sub-capabilities (Envelope reception, Manifest
composition, Persistence, Lineage linking, Query interface) must
report on every CAP-5 invocation. The first four are CLEAN
function-boundary timers; Query interface is the read-side surface
and never engages on the write path, so it reports via
``sub_capability_skipped`` with reason
``"manifest_write_does_not_query"``.

Envelope reception runs in invoke_cap5 (wrappers.py); the other
three CLEAN sub-caps run inside record_evaluation_manifest
(eval/cap5_entry.py).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

CAP5_TIMED_SUB_CAPS = (
    "envelope_reception",
    "manifest_composition",
    "persistence",
    "lineage_linking",
)
CAP5_SKIPPED_SUB_CAPS = ("query_interface",)


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="cap5-subcap-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        caller_id="cap5-subcap-test",
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


@pytest.mark.parametrize("sub_cap", CAP5_TIMED_SUB_CAPS)
def test_cap5_subcap_started_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_started", "CAP-5", sub_cap)
    assert len(matches) == 1, (
        f"CAP-5.{sub_cap}: expected one sub_capability_started, got {len(matches)}"
    )


@pytest.mark.parametrize("sub_cap", CAP5_TIMED_SUB_CAPS)
def test_cap5_subcap_completed_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_completed", "CAP-5", sub_cap)
    assert len(matches) == 1, (
        f"CAP-5.{sub_cap}: expected one sub_capability_completed, got {len(matches)}"
    )
    assert matches[0]["status"] == "success"
    assert matches[0]["duration_ms"] >= 0.0


@pytest.mark.parametrize("sub_cap", CAP5_SKIPPED_SUB_CAPS)
def test_cap5_subcap_skipped_fires_with_reason(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_skipped", "CAP-5", sub_cap)
    assert len(matches) == 1, (
        f"CAP-5.{sub_cap}: expected one sub_capability_skipped, got {len(matches)}"
    )
    assert matches[0]["reason"] == "manifest_write_does_not_query"


def test_cap5_emits_no_subcap_failed_on_happy_path(
    runner: PlatformRunner,
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    failures = [
        log
        for log in logs
        if log.get("event") == "sub_capability_failed"
        and log.get("capability_id") == "CAP-5"
    ]
    assert not failures, f"unexpected CAP-5 sub-cap failures: {failures}"
