"""Capability registry — runtime catalog of orchestrated capabilities (CP-F9 closure).

Layer 1's per-case orchestration runs CAP-1 → [CAP-2, CAP-7] → CAP-5.
Prior to L1-Pkt-B that sequence was hard-coded inside
:meth:`PlatformRunner.run_single_evaluation` with no module-level
record. This module records the sequence as ``CAPABILITY_REGISTRY``:
a frozen tuple of :class:`CapabilitySpec` entries that the
orchestrator iterates, dispatching to the right wrapper based on
``capability_id``.

Each spec carries metadata only — capability id, sequence position,
and the operational timeout in seconds. Callable references stay
in :mod:`llm_judge.control_plane.runner` because the four capabilities
have heterogeneous signatures and conditional dispatch on CAP-1
failure / aggregation between CAP-7 and CAP-5; uniform-signature
refactor is out of scope for L1-Pkt-B.

The registry is strict-from-day-one: it lists exactly the four
capabilities currently orchestrated. Adding a fifth capability is
a registry change plus a new dispatch arm in the orchestrator.

Initial timeouts are conservative starting points derived from a
50-case ``make verify-l1`` run on master @ 85c7e5c (Pre-flight 3
of L1-Pkt-B):

* CAP-1, CAP-2, CAP-5: P99 < 50 ms; timeout set to 5 s (>100x P99).
* CAP-7: P99 ≈ 3925 ms (driven by case-0 LLM warm-up); timeout
  set to 30 s — extra headroom over the strict 5x rule for
  warm-up volatility.

Sample is small (test runs only, not production traffic). Values
are operational starting points; tune post-deployment as production
latency data accumulates.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CAPABILITY_REGISTRY",
    "CapabilitySpec",
    "get_spec",
]


class CapabilitySpec(BaseModel):
    """One capability's runtime metadata.

    Frozen Pydantic; treat instances as values. The orchestrator
    looks up the matching wrapper by ``capability_id`` and uses
    ``timeout_seconds`` to configure the TimeoutGuardrail for
    each invocation.

    ``sequence_position`` is redundant with the registry tuple's
    index but preserved as an explicit field so the contract is
    self-describing: a spec viewed in isolation still records where
    it sits in the chain.
    """

    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(..., min_length=1)
    sequence_position: int = Field(..., ge=0)
    timeout_seconds: float = Field(..., gt=0)


CAPABILITY_REGISTRY: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(capability_id="CAP-1", sequence_position=0, timeout_seconds=5.0),
    CapabilitySpec(capability_id="CAP-2", sequence_position=1, timeout_seconds=5.0),
    CapabilitySpec(capability_id="CAP-7", sequence_position=2, timeout_seconds=30.0),
    CapabilitySpec(capability_id="CAP-5", sequence_position=3, timeout_seconds=5.0),
)


def get_spec(capability_id: str) -> CapabilitySpec:
    """Return the spec for ``capability_id``.

    Raises :class:`KeyError` if no entry matches. Linear scan — the
    registry is small (single-digit) and fixed at import time, so
    the cost is negligible and a parallel dict-by-id index would be
    duplicated state.
    """
    for spec in CAPABILITY_REGISTRY:
        if spec.capability_id == capability_id:
            return spec
    raise KeyError(f"No CapabilitySpec for capability_id={capability_id!r}")
