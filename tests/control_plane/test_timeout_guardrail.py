"""TimeoutGuardrail unit tests (CP-F12 closure, L1-Pkt-B Commit 5).

Pure guardrail behaviour: pre_call records a start; post_call denies
on overshoot, allows on undershoot. The CP-F12 gap-absence test
that exercises the runner end-to-end with a slow mock capability
ships in Commit 8.
"""

from __future__ import annotations

import time

from llm_judge.control_plane.capability_registry import CapabilitySpec
from llm_judge.control_plane.guardrails import (
    GuardrailContext,
    TimeoutGuardrail,
)


def _ctx(timeout_seconds: float = 5.0) -> GuardrailContext:
    return GuardrailContext(
        spec=CapabilitySpec(
            capability_id="CAP-1",
            sequence_position=0,
            timeout_seconds=timeout_seconds,
        ),
        request_id="req-timeout",
    )


def test_pre_call_records_start_perf_counter() -> None:
    g = TimeoutGuardrail()
    ctx = _ctx()
    decision = g.pre_call(ctx)
    assert decision.allowed is True
    assert any(k.startswith("timeout_guardrail.") for k in ctx.guardrail_state)


def test_post_call_allows_when_elapsed_under_budget() -> None:
    g = TimeoutGuardrail()
    ctx = _ctx(timeout_seconds=5.0)
    g.pre_call(ctx)
    decision = g.post_call(ctx)
    assert decision.allowed is True
    assert decision.reason is None


def test_post_call_denies_when_elapsed_exceeds_budget() -> None:
    g = TimeoutGuardrail()
    ctx = _ctx(timeout_seconds=0.05)
    g.pre_call(ctx)
    time.sleep(0.1)
    decision = g.post_call(ctx)
    assert decision.allowed is False
    assert decision.reason is not None
    assert "post-completion denial" in decision.reason
    assert decision.context["capability_id"] == "CAP-1"
    assert decision.context["configured_timeout_seconds"] == 0.05
    assert decision.context["observed_duration_seconds"] > 0.05


def test_post_call_without_pre_call_allows() -> None:
    """Defensive guard: if pre_call never ran (substrate misuse),
    post_call has nothing to measure against and returns Allow
    rather than denying without evidence."""
    g = TimeoutGuardrail()
    ctx = _ctx()
    decision = g.post_call(ctx)
    assert decision.allowed is True
