"""Guardrails — operational substrate around capability invocations (CP-F8 closure).

Layer 1's orchestrator wraps each capability invocation with the
substrate provided here. Concrete guardrails (timeout in L1-Pkt-B;
rate limit / circuit breaker / kill switch in future packets)
register against the substrate and have their ``pre_call`` and
``post_call`` hooks fired around every invocation.

Public API:

* :class:`Guardrail` — base class for concrete guardrails.
* :class:`GuardrailDecision` — Allow/Deny return shape.
* :class:`GuardrailContext` — per-invocation state passed to hooks.
* :func:`guardrail_context` — context manager that runs hooks.
* :func:`register_guardrail` — adds a guardrail to the global
  registry consumed by ``guardrail_context``.
* :func:`_reset_for_tests` — clears the registry; tests use it to
  isolate module-level state per the L1-Pkt-1 pattern.
"""

from __future__ import annotations

from llm_judge.control_plane.guardrails.substrate import (
    Guardrail,
    GuardrailContext,
    GuardrailDecision,
    _reset_for_tests,
    guardrail_context,
    register_guardrail,
)

__all__ = [
    "Guardrail",
    "GuardrailContext",
    "GuardrailDecision",
    "_reset_for_tests",
    "guardrail_context",
    "register_guardrail",
]
