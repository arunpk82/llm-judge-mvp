"""Substrate-wired runner integration test (CP-F8 closure, L1-Pkt-B Commit 4).

Verifies that ``PlatformRunner.run_single_evaluation`` actually
invokes registered guardrails' pre_call and post_call hooks for
every capability in :data:`CAPABILITY_REGISTRY`. The full mock-
guardrail-Deny gap-absence test lives in Commit 7 (separate file);
this file covers the wiring assertion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.control_plane.capability_registry import CAPABILITY_REGISTRY
from llm_judge.control_plane.guardrails import (
    Guardrail,
    GuardrailContext,
    GuardrailDecision,
    _reset_for_tests,
    register_guardrail,
)
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture(autouse=True)
def reset_guardrail_registry() -> None:
    _reset_for_tests()


class _RecordingGuardrail(Guardrail):
    """Records every pre_call and post_call invocation by capability_id."""

    def __init__(self) -> None:
        self.pre_calls: list[str] = []
        self.post_calls: list[str] = []

    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        self.pre_calls.append(ctx.spec.capability_id)
        return GuardrailDecision.allow()

    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        self.post_calls.append(ctx.spec.capability_id)
        return GuardrailDecision.allow()


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        rubric_id="chat_quality",
        caller_id="substrate-wiring-test",
    )


def test_substrate_fires_pre_and_post_hooks_for_every_capability(
    tmp_path: Path,
) -> None:
    """Registered guardrails see one pre_call and one post_call per
    capability in registry order. If Commit 4's substrate wrap were
    reverted, the recorder lists would be empty for some or all
    capabilities."""
    spy = _RecordingGuardrail()
    register_guardrail(spy)

    runner = PlatformRunner(
        platform_version="substrate-wiring-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )
    runner.run_single_evaluation(_payload())

    expected = [spec.capability_id for spec in CAPABILITY_REGISTRY]
    assert spy.pre_calls == expected
    assert spy.post_calls == expected
