"""Guardrail substrate primitive tests (CP-F8 scaffold, L1-Pkt-B Commit 3).

Pure scaffold coverage: decision shapes, context lifecycle, hook
ordering, registry isolation. Orchestrator wiring is exercised by
Commit 4's tests; concrete guardrails (timeout) by Commit 5.
"""

from __future__ import annotations

import pytest

from llm_judge.control_plane.capability_registry import CapabilitySpec
from llm_judge.control_plane.guardrails import (
    Guardrail,
    GuardrailContext,
    GuardrailDecision,
    _reset_for_tests,
    guardrail_context,
    register_guardrail,
)
from llm_judge.control_plane.types import GuardrailDeniedError


def _spec() -> CapabilitySpec:
    return CapabilitySpec(
        capability_id="CAP-1", sequence_position=0, timeout_seconds=5.0
    )


@pytest.fixture(autouse=True)
def reset_guardrail_registry() -> None:
    """Each test runs against an empty module-level registry."""
    _reset_for_tests()


# ---------------------------------------------------------------------
# GuardrailDecision shape
# ---------------------------------------------------------------------


def test_decision_allow_factory_constructs_allowed_decision() -> None:
    decision = GuardrailDecision.allow()
    assert decision.allowed is True
    assert decision.reason is None
    assert decision.context == {}


def test_decision_deny_factory_requires_reason_and_carries_context() -> None:
    decision = GuardrailDecision.deny("budget exceeded", capability_id="CAP-7", limit=30.0)
    assert decision.allowed is False
    assert decision.reason == "budget exceeded"
    assert decision.context == {"capability_id": "CAP-7", "limit": 30.0}


def test_decision_is_frozen() -> None:
    decision = GuardrailDecision.allow()
    with pytest.raises(Exception):
        decision.allowed = False


# ---------------------------------------------------------------------
# GuardrailContext lifecycle
# ---------------------------------------------------------------------


def test_context_carries_spec_request_id_and_mutable_state() -> None:
    ctx = GuardrailContext(spec=_spec(), request_id="req-1")
    assert ctx.spec.capability_id == "CAP-1"
    assert ctx.request_id == "req-1"
    assert ctx.guardrail_state == {}

    ctx.guardrail_state["timeout_start"] = 1234.5
    assert ctx.guardrail_state["timeout_start"] == 1234.5


# ---------------------------------------------------------------------
# guardrail_context with no guardrails registered
# ---------------------------------------------------------------------


def test_empty_registry_yields_context_without_raising() -> None:
    spec = _spec()
    with guardrail_context(spec, "req-empty") as ctx:
        assert ctx.spec is spec
        assert ctx.request_id == "req-empty"


# ---------------------------------------------------------------------
# Pre-call hook semantics
# ---------------------------------------------------------------------


class _AllowSpy(Guardrail):
    def __init__(self) -> None:
        self.pre_called = 0
        self.post_called = 0

    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        self.pre_called += 1
        return GuardrailDecision.allow()

    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        self.post_called += 1
        return GuardrailDecision.allow()


class _DenyPreCall(Guardrail):
    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        return GuardrailDecision.deny(
            "denied for test", capability_id=ctx.spec.capability_id
        )


class _DenyPostCall(Guardrail):
    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        return GuardrailDecision.deny("post denied", elapsed=99.9)


def test_pre_call_allow_runs_body_and_post_call() -> None:
    spy = _AllowSpy()
    body_ran = False
    with guardrail_context(_spec(), "req-allow", guardrails=[spy]):
        body_ran = True
    assert body_ran
    assert spy.pre_called == 1
    assert spy.post_called == 1


def test_pre_call_deny_raises_and_skips_body() -> None:
    body_ran = False
    with pytest.raises(GuardrailDeniedError, match="denied for test"):
        with guardrail_context(_spec(), "req-deny", guardrails=[_DenyPreCall()]):
            body_ran = True
    assert body_ran is False


def test_post_call_deny_raises_after_body() -> None:
    body_ran = False
    with pytest.raises(GuardrailDeniedError, match="post denied"):
        with guardrail_context(_spec(), "req-post-deny", guardrails=[_DenyPostCall()]):
            body_ran = True
    assert body_ran is True


def test_post_call_runs_even_when_body_raises() -> None:
    spy = _AllowSpy()
    with pytest.raises(RuntimeError, match="body boom"):
        with guardrail_context(_spec(), "req-body-raise", guardrails=[spy]):
            raise RuntimeError("body boom")
    # Pre and post both fired despite the body raising.
    assert spy.pre_called == 1
    assert spy.post_called == 1


# ---------------------------------------------------------------------
# Module-level registry consumed when no override is passed
# ---------------------------------------------------------------------


def test_register_guardrail_adds_to_module_registry() -> None:
    spy = _AllowSpy()
    register_guardrail(spy)
    with guardrail_context(_spec(), "req-registry"):
        pass
    assert spy.pre_called == 1
    assert spy.post_called == 1


def test_reset_for_tests_clears_registry() -> None:
    register_guardrail(_AllowSpy())
    _reset_for_tests()
    spy = _AllowSpy()
    with guardrail_context(_spec(), "req-after-reset"):
        pass
    # First spy is gone; the second one was never registered, so it
    # never fired either.
    assert spy.pre_called == 0
    assert spy.post_called == 0
