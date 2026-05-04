"""CP-F8 substrate gap-absence test.

Per Brief Template v1.3: registers a mock guardrail that always
denies CAP-1 and verifies the denial actually prevents the
capability from running. If L1-Pkt-B Commit 4's substrate wrap
around the orchestrator iteration were reverted, the mock
guardrail's pre_call decision would have no effect, ``invoke_cap1``
would still run, and this test would fail.

Evidence D1.1 (operational guardrails substrate) mechanism class
transition is verified.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_judge.control_plane.guardrails import (
    Guardrail,
    GuardrailContext,
    GuardrailDecision,
    register_guardrail,
)
from llm_judge.control_plane.guardrails import substrate as _substrate
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture(autouse=True)
def isolated_guardrail_registry() -> Iterator[None]:
    saved = list(_substrate._REGISTERED_GUARDRAILS)
    _substrate._REGISTERED_GUARDRAILS.clear()
    try:
        yield
    finally:
        _substrate._REGISTERED_GUARDRAILS.clear()
        _substrate._REGISTERED_GUARDRAILS.extend(saved)


class _DenyCap1Always(Guardrail):
    """Pre-call denial for CAP-1; everything else passes through."""

    def __init__(self) -> None:
        self.cap1_invocations_observed_pre = 0
        self.cap1_invocations_observed_post = 0

    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        if ctx.spec.capability_id == "CAP-1":
            self.cap1_invocations_observed_pre += 1
            return GuardrailDecision.deny(
                "test-only Deny on CAP-1", capability_id="CAP-1"
            )
        return GuardrailDecision.allow()

    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        if ctx.spec.capability_id == "CAP-1":
            self.cap1_invocations_observed_post += 1
        return GuardrailDecision.allow()


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        rubric_id="chat_quality",
        caller_id="cap-f8-gap-absence",
    )


def test_pre_call_deny_blocks_capability_invocation(tmp_path: Path) -> None:
    """The mock guardrail's pre_call returns Deny for CAP-1; the
    runner must not invoke ``invoke_cap1`` and must record CAP-1 as
    failed in the integrity list. CAP-2 and CAP-7 must be skipped
    via the standard cap1_failed path; CAP-5 must still run."""
    deny = _DenyCap1Always()
    register_guardrail(deny)

    runner = PlatformRunner(
        platform_version="cap-f8-gap-absence-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )
    result = runner.run_single_evaluation(_payload())

    # Pre_call observed CAP-1 exactly once.
    assert deny.cap1_invocations_observed_pre == 1
    # Pre_call denied before the substrate yielded; no body ran and
    # post_call did not fire for CAP-1.
    assert deny.cap1_invocations_observed_post == 0

    # CAP-1 failed (denial recorded as a capability failure).
    cap1_records = [
        r for r in result.envelope.integrity if r.capability_id == "CAP-1"
    ]
    assert len(cap1_records) == 1
    assert cap1_records[0].status == "failure"
    assert cap1_records[0].error_type == "GuardrailDeniedError"

    # CAP-2 and CAP-7 skipped via the existing sibling-skip path.
    cap2_records = [
        r for r in result.envelope.integrity if r.capability_id == "CAP-2"
    ]
    cap7_records = [
        r for r in result.envelope.integrity if r.capability_id == "CAP-7"
    ]
    assert cap2_records[0].status == "skipped_upstream_failure"
    assert cap7_records[0].status == "skipped_upstream_failure"

    # CAP-5 ran (manifest written).
    cap5_records = [
        r for r in result.envelope.integrity if r.capability_id == "CAP-5"
    ]
    assert cap5_records[0].status == "success"
    assert result.manifest_id

    # Run-level integrity reflects the denial.
    assert result.integrity.complete is False
    assert "CAP-1" in result.integrity.missing_capabilities
    assert result.integrity.reason is not None
    assert "siblings skipped" in result.integrity.reason
