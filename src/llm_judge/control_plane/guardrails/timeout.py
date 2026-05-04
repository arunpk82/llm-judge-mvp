"""TimeoutGuardrail — first concrete guardrail (CP-F12 closure, L1-Pkt-B Commit 5).

Per-capability timeout enforcement on the substrate from Commit 3,
wired around every invocation by Commit 4. Implementation strategy
is **Option β (post-completion denial)** per Layer 1 chat checkpoint
of L1-Pkt-B.

Why post-completion denial:
  Pre-flight 4 of L1-Pkt-B established that the entire Layer 1 stack
  is synchronous — no asyncio anywhere, no threading, batch_runner
  iterates cases in the main thread. The two viable sync timeout
  strategies were:

  * Option α — :func:`signal.alarm`: Unix-only, main-thread only,
    nesting is fragile, library code can swallow signals. Brittle.
  * Option β — post-call elapsed-time check: capability runs to
    completion; pre_call records a start timestamp; post_call
    computes elapsed and denies if the configured budget was
    exceeded. Portable, no platform coupling.

  The Layer 1 chat picked Option β. Trade-off recorded in the
  L1-Pkt-B completion signal: this closes CP-F12 in the
  observability sense (timeout breach is a materialised
  :class:`GuardrailDeniedError` with structured context plus a
  ``guardrail.timeout_exceeded`` telemetry event) but does NOT
  bound resource consumption — a runaway capability continues to
  hold the worker until it finishes naturally; only its outcome is
  rejected.

The unmet "actually interrupt a runaway capability" surface is
filed as **CP-F23 candidate** for v1.5 gap analysis. Future packets
may add bounded execution via subprocess sandboxing, async
migration, or signal-based fallback for narrow cases.

Pre_call sets ``ctx.guardrail_state[_TIMEOUT_START_KEY]`` to the
current ``time.perf_counter()`` reading and returns Allow.
Post_call reads that key, computes ``elapsed = perf_counter() -
start``, compares against ``ctx.spec.timeout_seconds``, emits
``guardrail.timeout_exceeded`` on breach, and returns Deny — the
substrate translates that into a :class:`GuardrailDeniedError`
that surfaces through the runner's per-arm exception handler
(captured to integrity for CAP-1/CAP-2/CAP-7; propagated for
CAP-5 per the D5 contract).
"""

from __future__ import annotations

import time

from llm_judge.control_plane.guardrails.substrate import (
    Guardrail,
    GuardrailContext,
    GuardrailDecision,
)
from llm_judge.control_plane.observability import emit_event

__all__ = ["TimeoutGuardrail"]


_TIMEOUT_START_KEY = "timeout_guardrail.start_perf_counter"


class TimeoutGuardrail(Guardrail):
    """Per-request timeout enforcement (CP-F12 closure, Option β)."""

    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        ctx.guardrail_state[_TIMEOUT_START_KEY] = time.perf_counter()
        return GuardrailDecision.allow()

    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        start = ctx.guardrail_state.get(_TIMEOUT_START_KEY)
        if start is None:
            # pre_call did not run for this invocation; nothing to
            # measure against. Substrate guarantees pre_call before
            # post_call when the same guardrail instance is used,
            # so this branch is defensive against future misuse.
            return GuardrailDecision.allow()
        elapsed = time.perf_counter() - start
        configured = ctx.spec.timeout_seconds
        if elapsed <= configured:
            return GuardrailDecision.allow()
        emit_event(
            "guardrail.timeout_exceeded",
            capability_id=ctx.spec.capability_id,
            request_id=ctx.request_id,
            configured_timeout_seconds=configured,
            observed_duration_seconds=elapsed,
        )
        return GuardrailDecision.deny(
            "timeout exceeded (post-completion denial)",
            capability_id=ctx.spec.capability_id,
            configured_timeout_seconds=configured,
            observed_duration_seconds=elapsed,
        )
