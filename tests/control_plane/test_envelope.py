"""Unit tests for ProvenanceEnvelope immutability and signing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llm_judge.control_plane.envelope import (
    ProvenanceEnvelope,
    compute_signature,
    new_envelope,
)


def _make() -> ProvenanceEnvelope:
    return new_envelope(
        request_id="req-001",
        caller_id="test",
        arrived_at=datetime(2026, 4, 22, 12, 0, tzinfo=timezone.utc),
        platform_version="deadbeef",
    )


def test_construction_signs_initial_envelope() -> None:
    env = _make()
    assert env.signature
    assert env.capability_chain == []
    assert env.verify_signature()


def test_stamped_returns_new_instance_with_chain_and_stamps() -> None:
    env = _make()
    stamped = env.stamped(
        capability="CAP-1",
        dataset_registry_id="transient_abc",
        input_hash="sha256:feed",
    )

    assert stamped is not env
    assert stamped.capability_chain == ["CAP-1"]
    assert stamped.dataset_registry_id == "transient_abc"
    assert stamped.input_hash == "sha256:feed"
    # Original is unchanged
    assert env.capability_chain == []
    assert env.dataset_registry_id is None


def test_stamped_signature_changes_and_verifies() -> None:
    env = _make()
    stamped = env.stamped(capability="CAP-1", dataset_registry_id="transient_abc")

    assert stamped.signature != env.signature
    assert stamped.verify_signature()


def test_original_is_immutable() -> None:
    env = _make()
    with pytest.raises(ValidationError):
        env.signature = "tampered"  # type: ignore[misc]


def test_tampered_field_invalidates_signature() -> None:
    env = _make()
    stamped = env.stamped(capability="CAP-1", dataset_registry_id="transient_abc")

    # Simulate tampering: recreate envelope with a field changed but the
    # stamped signature carried over.
    tampered = ProvenanceEnvelope(
        **{**stamped.model_dump(), "dataset_registry_id": "transient_OTHER"}
    )
    assert not tampered.verify_signature()


def test_compute_signature_is_stable_for_same_payload() -> None:
    payload = {
        "request_id": "r1",
        "caller_id": "c",
        "arrived_at": datetime(2026, 4, 22, tzinfo=timezone.utc).isoformat(),
        "capability_chain": ["CAP-1"],
        "platform_version": "abc",
    }
    assert compute_signature(payload) == compute_signature(payload)


def test_stamped_requires_capability() -> None:
    env = _make()
    with pytest.raises(ValueError, match="capability"):
        env.stamped(capability="")


def test_stamped_chains_multiple_capabilities() -> None:
    env = _make()
    env = env.stamped(capability="CAP-1", dataset_registry_id="transient_a")
    env = env.stamped(capability="CAP-2", rule_set_version="v1", rules_fired=[])
    env = env.stamped(capability="CAP-7")
    env = env.stamped(capability="CAP-5")

    assert env.capability_chain == ["CAP-1", "CAP-2", "CAP-7", "CAP-5"]
    assert env.verify_signature()
