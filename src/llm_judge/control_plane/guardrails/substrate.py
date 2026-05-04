"""Guardrail substrate primitives (CP-F8 scaffold, L1-Pkt-B Commit 3).

Provides the framework that concrete guardrails plug into:

* :class:`Guardrail` — base class. Subclasses override ``pre_call``
  and/or ``post_call``; both default to Allow.
* :class:`GuardrailDecision` — Allow/Deny return shape with a
  reason string and structured context for telemetry. Modify is
  reserved for a future packet (Decision 3c, L1-Pkt-B).
* :class:`GuardrailContext` — per-invocation state passed to hooks.
  Mutable so guardrails can carry handles between ``pre_call`` and
  ``post_call`` (e.g. the timeout guardrail records a start time).
* :func:`guardrail_context` — context manager. Iterates registered
  guardrails: pre-call before yielding, post-call after. Any Deny
  decision raises :class:`GuardrailDeniedError`. Telemetry events
  ``guardrail.pre_call`` and ``guardrail.post_call`` are emitted
  for every hook invocation. Events ship through the existing
  structlog/event-bus path (ungoverned vocabulary in this packet;
  CP-F7 + CP-F13 closure absorbs into the platform-wide event
  vocabulary in a future packet).
* :func:`register_guardrail` — adds a guardrail to the
  module-level registry consumed by ``guardrail_context``.

This commit ships the substrate as a pure scaffold — no guardrails
registered, orchestrator does not yet wrap iteration. Commit 4
wires the substrate into the orchestrator; Commit 5 registers the
first concrete guardrail (TimeoutGuardrail).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_judge.control_plane.capability_registry import CapabilitySpec
from llm_judge.control_plane.observability import emit_event
from llm_judge.control_plane.types import GuardrailDeniedError

__all__ = [
    "Guardrail",
    "GuardrailContext",
    "GuardrailDecision",
    "_reset_for_tests",
    "guardrail_context",
    "register_guardrail",
]


class GuardrailDecision(BaseModel):
    """Return type of guardrail ``pre_call`` and ``post_call`` hooks.

    Allow/Deny only — Modify-on-return semantics are reserved for a
    future packet (per L1-Pkt-B Decision 3c). When ``allowed`` is
    False, ``reason`` is required and the substrate raises
    :class:`GuardrailDeniedError` with the structured context
    attached for downstream observers.
    """

    model_config = ConfigDict(frozen=True)

    allowed: bool
    reason: str | None = None
    context: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def allow(cls) -> GuardrailDecision:
        """Construct an Allow decision."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str, **context: Any) -> GuardrailDecision:
        """Construct a Deny decision. ``reason`` is required;
        keyword arguments populate ``context`` for telemetry and the
        :class:`GuardrailDeniedError` message."""
        return cls(allowed=False, reason=reason, context=dict(context))


class GuardrailContext(BaseModel):
    """Per-invocation state passed to every guardrail hook.

    Mutable: guardrails write to ``guardrail_state`` to carry handles
    between ``pre_call`` and ``post_call`` (the timeout guardrail
    stores a start timestamp, for example). Per-request scope only —
    the dict does not persist across capability invocations or
    across runs (Decision 3b, L1-Pkt-B; per-process / per-tenant
    state deferred to a future packet).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spec: CapabilitySpec
    request_id: str
    guardrail_state: dict[str, Any] = Field(default_factory=dict)


class Guardrail:
    """Base class for guardrails.

    Subclasses override ``pre_call``, ``post_call``, or both. Both
    methods default to Allow so a subclass that only governs the
    pre- or post- side of the call is not forced to write a no-op
    on the other side.

    Implementations should keep hook bodies fast and side-effect-
    bounded; the substrate executes hooks synchronously around every
    capability invocation and any cost is added to the per-request
    latency budget.
    """

    def pre_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        """Run before the capability is invoked. Return Allow to
        permit invocation; return Deny to short-circuit the call."""
        del ctx
        return GuardrailDecision.allow()

    def post_call(self, ctx: GuardrailContext) -> GuardrailDecision:
        """Run after the capability has executed (or attempted to
        execute and raised — the substrate uses ``finally`` semantics
        so post_call still fires on capability error). Return Allow
        for a clean exit; return Deny to mark the invocation as
        denied after-the-fact (used by the timeout guardrail's
        Option β post-completion denial)."""
        del ctx
        return GuardrailDecision.allow()


# Module-level registry of guardrails consumed by guardrail_context()
# when no explicit override is passed. Populated by Commit 5 (timeout)
# and future guardrail packets via register_guardrail.
_REGISTERED_GUARDRAILS: list[Guardrail] = []


def register_guardrail(guardrail: Guardrail) -> None:
    """Add a guardrail to the global registry. All capability
    invocations through :func:`guardrail_context` (without an
    explicit ``guardrails`` override) will run this guardrail's
    hooks."""
    _REGISTERED_GUARDRAILS.append(guardrail)


def _reset_for_tests() -> None:
    """Clear the module-level guardrail registry. Tests that mutate
    registered guardrails use this helper for isolation, matching
    the :func:`llm_judge.control_plane.configuration._reset_for_tests`
    pattern from L1-Pkt-1."""
    _REGISTERED_GUARDRAILS.clear()


def _raise_denied(
    *,
    guardrail_class: str,
    spec: CapabilitySpec,
    decision: GuardrailDecision,
    phase: str,
) -> None:
    """Translate a Deny decision into a :class:`GuardrailDeniedError`
    raise. Centralised so the message format and exception args
    stay consistent across pre- and post- phases."""
    reason = decision.reason or "(no reason supplied)"
    raise GuardrailDeniedError(
        f"Guardrail {guardrail_class} denied {spec.capability_id} "
        f"({phase}): {reason}"
    )


@contextmanager
def guardrail_context(
    spec: CapabilitySpec,
    request_id: str,
    *,
    guardrails: Sequence[Guardrail] | None = None,
) -> Iterator[GuardrailContext]:
    """Wrap a capability invocation with the guardrail substrate.

    Pre-call hooks run before the with-block executes; if any
    returns Deny, :class:`GuardrailDeniedError` is raised and the
    capability is not invoked. Post-call hooks run after the
    with-block exits (using ``finally``-like semantics so they run
    even if the capability raised); a Deny here also raises
    :class:`GuardrailDeniedError`.

    Telemetry: each hook invocation emits a structlog/event-bus
    event (``guardrail.pre_call`` or ``guardrail.post_call``) with
    capability_id, request_id, guardrail class, allowed flag, and
    reason. Concrete guardrails may emit additional events (e.g.
    the timeout guardrail emits ``guardrail.timeout_exceeded``
    when its post_call denies).

    ``guardrails`` defaults to the module-level registry populated
    by :func:`register_guardrail`; tests pass an explicit list to
    isolate from registry state.
    """
    active = list(guardrails) if guardrails is not None else list(_REGISTERED_GUARDRAILS)
    ctx = GuardrailContext(spec=spec, request_id=request_id)

    for guard in active:
        decision = guard.pre_call(ctx)
        emit_event(
            "guardrail.pre_call",
            capability_id=spec.capability_id,
            request_id=request_id,
            guardrail=type(guard).__name__,
            allowed=decision.allowed,
            reason=decision.reason,
        )
        if not decision.allowed:
            _raise_denied(
                guardrail_class=type(guard).__name__,
                spec=spec,
                decision=decision,
                phase="pre_call",
            )

    post_call_error: BaseException | None = None
    try:
        yield ctx
    finally:
        for guard in active:
            try:
                decision = guard.post_call(ctx)
            except Exception as exc:
                # Hook itself raised — record so we can re-raise once
                # all hooks have had a chance to observe.
                emit_event(
                    "guardrail.post_call",
                    capability_id=spec.capability_id,
                    request_id=request_id,
                    guardrail=type(guard).__name__,
                    allowed=False,
                    reason=f"hook raised: {type(exc).__name__}",
                )
                if post_call_error is None:
                    post_call_error = exc
                continue
            emit_event(
                "guardrail.post_call",
                capability_id=spec.capability_id,
                request_id=request_id,
                guardrail=type(guard).__name__,
                allowed=decision.allowed,
                reason=decision.reason,
            )
            if not decision.allowed and post_call_error is None:
                # Capture the raise; let any remaining hooks observe
                # the invocation before propagating.
                try:
                    _raise_denied(
                        guardrail_class=type(guard).__name__,
                        spec=spec,
                        decision=decision,
                        phase="post_call",
                    )
                except GuardrailDeniedError as exc:
                    post_call_error = exc
        if post_call_error is not None:
            raise post_call_error
