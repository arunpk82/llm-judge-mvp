"""Capability-registry shape and lookup tests (CP-F9 mechanism).

Verifies the runtime catalogue produced in Commit 1 of L1-Pkt-B.
Orchestrator iteration (Commit 2) and substrate wiring (Commits 3-5)
are covered separately.
"""

from __future__ import annotations

import pytest

from llm_judge.control_plane.capability_registry import (
    CAPABILITY_REGISTRY,
    CapabilitySpec,
    get_spec,
)

# ---------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------


def test_registry_is_a_tuple_of_capability_specs() -> None:
    assert isinstance(CAPABILITY_REGISTRY, tuple)
    assert all(isinstance(spec, CapabilitySpec) for spec in CAPABILITY_REGISTRY)


def test_registry_lists_recon_confirmed_capabilities() -> None:
    """The four capabilities currently orchestrated all appear; new
    capabilities require an explicit registry update + dispatch arm."""
    ids = [spec.capability_id for spec in CAPABILITY_REGISTRY]
    assert ids == ["CAP-1", "CAP-2", "CAP-7", "CAP-5"]


def test_sequence_position_matches_tuple_index() -> None:
    """``sequence_position`` is redundant-but-explicit; assert the
    self-describing invariant holds at module load."""
    for index, spec in enumerate(CAPABILITY_REGISTRY):
        assert spec.sequence_position == index


# ---------------------------------------------------------------------
# Initial timeout values (Pre-flight 3 latency catalog)
# ---------------------------------------------------------------------


def test_cap1_cap2_cap5_timeouts_are_five_seconds() -> None:
    """CAP-1, CAP-2, CAP-5 P99 latency is sub-50 ms in test runs;
    a 5 s timeout is >100x P99 and absorbs CI-environment jitter."""
    assert get_spec("CAP-1").timeout_seconds == 5.0
    assert get_spec("CAP-2").timeout_seconds == 5.0
    assert get_spec("CAP-5").timeout_seconds == 5.0


def test_cap7_timeout_is_thirty_seconds() -> None:
    """CAP-7 P99 ≈ 3925 ms in test runs (case-0 LLM warm-up); 30 s
    gives extra headroom over the strict 5x rule for warm-up
    volatility. Tune post-deployment from production traffic."""
    assert get_spec("CAP-7").timeout_seconds == 30.0


def test_all_timeouts_are_positive() -> None:
    """Pydantic ``Field(..., gt=0)`` enforces this at construction;
    re-asserting documents the invariant for readers and catches a
    future spec that drifts past validator weakening."""
    for spec in CAPABILITY_REGISTRY:
        assert spec.timeout_seconds > 0


# ---------------------------------------------------------------------
# Frozen Pydantic — instances are values, not handles
# ---------------------------------------------------------------------


def test_capability_spec_is_frozen() -> None:
    spec = CAPABILITY_REGISTRY[0]
    with pytest.raises(Exception):
        spec.timeout_seconds = 99.0


# ---------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------


def test_get_spec_returns_matching_entry() -> None:
    spec = get_spec("CAP-7")
    assert spec.capability_id == "CAP-7"
    assert spec.sequence_position == 2
    assert spec.timeout_seconds == 30.0


def test_get_spec_raises_key_error_for_unknown_id() -> None:
    with pytest.raises(KeyError, match="CAP-99"):
        get_spec("CAP-99")
