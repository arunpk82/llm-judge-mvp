"""CAP-1 sub-capability event coverage.

For every CAP-1 sub-capability the portal documents (Reception,
Validation, Registration, Hashing, Lineage tracking, Discovery), a
single happy-path Runner invocation must produce the matching event:

* Reception, Validation, Registration, Hashing, Lineage tracking →
  one ``sub_capability_started`` + one ``sub_capability_completed``
  pair, both tagged ``capability_id="CAP-1"``.
* Discovery → ``sub_capability_skipped`` with the documented reason
  ``"single_eval_does_not_query_registry"`` (it never engages on the
  single-eval Runner path).

These tests pin the sub-cap surface so future refactors can't
silently drop emissions. Field-shape contracts live in
``test_event_contracts.py``; this module only checks fire-rate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

CAP1_TIMED_SUB_CAPS = (
    "reception",
    "hashing",
    "validation",
    "registration",
    "lineage_tracking",
)
CAP1_SKIPPED_SUB_CAPS = ("discovery",)


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="cap1-subcap-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        caller_id="cap1-subcap-test",
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


@pytest.mark.parametrize("sub_cap", CAP1_TIMED_SUB_CAPS)
def test_cap1_subcap_started_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_started", "CAP-1", sub_cap)
    assert len(matches) == 1, (
        f"CAP-1.{sub_cap}: expected exactly one sub_capability_started, "
        f"got {len(matches)}"
    )


@pytest.mark.parametrize("sub_cap", CAP1_TIMED_SUB_CAPS)
def test_cap1_subcap_completed_fires(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_completed", "CAP-1", sub_cap)
    assert len(matches) == 1, (
        f"CAP-1.{sub_cap}: expected exactly one sub_capability_completed, "
        f"got {len(matches)}"
    )
    assert matches[0]["status"] == "success"
    assert matches[0]["duration_ms"] >= 0.0


@pytest.mark.parametrize("sub_cap", CAP1_SKIPPED_SUB_CAPS)
def test_cap1_subcap_skipped_fires_with_reason(
    runner: PlatformRunner, sub_cap: str
) -> None:
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    matches = _matching(logs, "sub_capability_skipped", "CAP-1", sub_cap)
    assert len(matches) == 1, (
        f"CAP-1.{sub_cap}: expected exactly one sub_capability_skipped, "
        f"got {len(matches)}"
    )
    assert matches[0]["reason"] == "single_eval_does_not_query_registry"


def test_cap1_emits_no_subcap_failed_on_happy_path(
    runner: PlatformRunner,
) -> None:
    """A successful run must not emit any CAP-1 sub_capability_failed."""
    with capture_logs() as logs:
        runner.run_single_evaluation(_payload())

    failures = [
        log
        for log in logs
        if log.get("event") == "sub_capability_failed"
        and log.get("capability_id") == "CAP-1"
    ]
    assert not failures, f"unexpected CAP-1 sub-cap failures: {failures}"
