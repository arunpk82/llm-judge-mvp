"""CP-F3 gap-absence test.

Per Brief Template v1.3: a gap-absence test exercises the closure
surface and would fail if the runtime gate introduced by Commit 1
were reverted. End-State property A3.3 transitions
"Capability cannot stamp fields outside its declared territory"
from convention to runtime gate; this test asserts the runtime gate
fails closed when crossed.

Three vectors are covered:

1. CAP-1 stamping a CAP-2 field (rule_set_version)
2. CAP-2 stamping a CAP-1 field (dataset_registry_id)
3. CAP-7 (chain-only allowlist) stamping any field at all

In every case the offending field name appears in the message so the
bypass attempt fails loudly at the wrapper boundary.

If the validation block in :meth:`ProvenanceEnvelope.stamped` were
removed, all four assertions below would fail — that is the
gap-absence property.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from llm_judge.control_plane.envelope import (
    ProvenanceEnvelope,
    new_envelope,
)
from llm_judge.control_plane.types import FieldOwnershipViolationError


def _make() -> ProvenanceEnvelope:
    return new_envelope(
        request_id="req-fo-gap",
        caller_id="test",
        arrived_at=datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        platform_version="deadbeef",
    )


def test_cap1_stamping_cap2_field_raises() -> None:
    env = _make()
    with pytest.raises(FieldOwnershipViolationError) as ei:
        env.stamped(capability="CAP-1", rule_set_version="v1")
    assert "rule_set_version" in str(ei.value)
    assert "CAP-1" in str(ei.value)


def test_cap2_stamping_cap1_field_raises() -> None:
    env = _make()
    with pytest.raises(FieldOwnershipViolationError) as ei:
        env.stamped(capability="CAP-2", dataset_registry_id="transient_abc")
    assert "dataset_registry_id" in str(ei.value)
    assert "CAP-2" in str(ei.value)


def test_cap7_chain_only_capability_cannot_stamp_any_field() -> None:
    env = _make()
    with pytest.raises(FieldOwnershipViolationError) as ei:
        env.stamped(capability="CAP-7", input_hash="sha256:beef")
    assert "input_hash" in str(ei.value)
    assert "CAP-7" in str(ei.value)


def test_unregistered_capability_raises_with_registration_hint() -> None:
    """A capability id not in FIELD_OWNERSHIP at all must fail
    closed. The message points at the registration site so the
    fix is obvious."""
    env = _make()
    with pytest.raises(FieldOwnershipViolationError) as ei:
        env.stamped(capability="CAP-99")
    assert "CAP-99" in str(ei.value)
    assert "FIELD_OWNERSHIP" in str(ei.value)


def test_field_ownership_violation_is_value_error() -> None:
    """Subclass relationship: existing ``except ValueError`` callers
    around envelope construction continue to catch the new class."""
    env = _make()
    with pytest.raises(ValueError):
        env.stamped(capability="CAP-1", rule_set_version="v1")
