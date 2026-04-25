"""Unit tests for ProvenanceEnvelope immutability and signing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
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


# -----------------------------------------------------------------
# CP-1b: schema_version + integrity
# -----------------------------------------------------------------


def test_schema_version_defaults_to_3() -> None:
    """CP-2 bumps the schema to 3 (CapabilityIntegrityRecord gains
    duration_ms). Old envelopes without a schema_version backfill to 1;
    explicit v2 envelopes still parse (see
    test_v2_envelope_dict_parses_with_duration_none)."""
    env = _make()
    assert env.schema_version == 3


def test_integrity_defaults_to_empty() -> None:
    env = _make()
    assert env.integrity == []


def test_old_cp1_shape_envelope_parses() -> None:
    """An envelope dict from CP-1 (no schema_version, no integrity)
    must round-trip; the model backfills schema_version=1 and
    integrity=[]."""
    cp1_dict = {
        "request_id": "old-req",
        "caller_id": "old-caller",
        "arrived_at": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
        "parent_attestation_id": None,
        "dataset_registry_id": "transient_x",
        "input_hash": "sha256:abc",
        "rule_set_version": "v1",
        "rules_fired": [],
        "platform_version": "old-sha",
        "capability_chain": ["CAP-1", "CAP-2", "CAP-7"],
        "signature": "ignored-for-this-test",
    }
    parsed = ProvenanceEnvelope.model_validate(cp1_dict)
    assert parsed.schema_version == 1
    assert parsed.integrity == []
    assert parsed.capability_chain == ["CAP-1", "CAP-2", "CAP-7"]


def test_integrity_record_frozen() -> None:
    rec = CapabilityIntegrityRecord(capability_id="CAP-1", status="success")
    with pytest.raises(ValidationError):
        rec.capability_id = "CAP-7"  # type: ignore[misc]


def test_stamping_with_integrity_record_updates_signature() -> None:
    env = _make()
    rec = CapabilityIntegrityRecord(capability_id="CAP-1", status="success")
    stamped = env.with_integrity(rec)

    assert stamped is not env
    assert len(stamped.integrity) == 1
    assert stamped.integrity[0].capability_id == "CAP-1"
    assert stamped.integrity[0].status == "success"
    assert stamped.signature != env.signature
    assert stamped.verify_signature()
    # Original unchanged
    assert env.integrity == []


def test_with_integrity_appends_preserves_chain() -> None:
    env = _make().stamped(capability="CAP-1", dataset_registry_id="t_x")
    rec = CapabilityIntegrityRecord(
        capability_id="CAP-2",
        status="failure",
        error_type="RuntimeError",
        error_message="boom",
    )
    stamped = env.with_integrity(rec)
    # chain is not touched by with_integrity
    assert stamped.capability_chain == env.capability_chain
    # record recorded
    assert stamped.integrity[-1].status == "failure"
    assert stamped.integrity[-1].error_type == "RuntimeError"
    assert stamped.verify_signature()


def test_integrity_included_in_canonical_json_for_signing() -> None:
    env = _make().with_integrity(
        CapabilityIntegrityRecord(capability_id="CAP-1", status="success")
    )

    # Tamper: swap in a different integrity list but keep the signature.
    fake_record = CapabilityIntegrityRecord(
        capability_id="CAP-1", status="failure"
    )
    tampered = ProvenanceEnvelope(
        **{**env.model_dump(), "integrity": [fake_record.model_dump()]}
    )
    assert not tampered.verify_signature()


def test_schema_version_included_in_canonical_json_for_signing() -> None:
    env = _make()
    # Force schema_version to a different value without recomputing signature.
    tampered = ProvenanceEnvelope(
        **{**env.model_dump(), "schema_version": 1}
    )
    assert not tampered.verify_signature()


# -----------------------------------------------------------------
# CP-2: schema v3 — CapabilityIntegrityRecord.duration_ms
# -----------------------------------------------------------------


def test_integrity_record_duration_ms_defaults_to_none() -> None:
    """Older CP-1b-style constructions that omit duration_ms still
    produce a valid record."""
    rec = CapabilityIntegrityRecord(capability_id="CAP-1", status="success")
    assert rec.duration_ms is None


def test_integrity_record_accepts_duration_ms() -> None:
    rec = CapabilityIntegrityRecord(
        capability_id="CAP-1", status="success", duration_ms=12.34
    )
    assert rec.duration_ms == 12.34


def test_v2_envelope_dict_parses_with_duration_ms_none_on_records() -> None:
    """A v2-shape envelope (schema_version=2, integrity records without
    duration_ms) continues to parse; duration_ms defaults to None on
    each record."""
    v2_dict = {
        "request_id": "v2-req",
        "caller_id": "test",
        "arrived_at": datetime(2026, 3, 1, tzinfo=timezone.utc).isoformat(),
        "parent_attestation_id": None,
        "platform_version": "v2-sha",
        "capability_chain": ["CAP-1"],
        "schema_version": 2,
        "integrity": [
            {"capability_id": "CAP-1", "status": "success"},
            {"capability_id": "CAP-2", "status": "success"},
        ],
        "signature": "ignored-for-this-test",
    }
    parsed = ProvenanceEnvelope.model_validate(v2_dict)
    assert parsed.schema_version == 2
    assert len(parsed.integrity) == 2
    for rec in parsed.integrity:
        assert rec.duration_ms is None


def test_v3_envelope_with_duration_ms_parses_and_signs() -> None:
    """End-to-end: construct a fresh envelope, stamp with an
    integrity record carrying duration_ms, verify signature."""
    env = _make().with_integrity(
        CapabilityIntegrityRecord(
            capability_id="CAP-1", status="success", duration_ms=7.5
        )
    )
    assert env.schema_version == 3
    assert env.integrity[0].duration_ms == 7.5
    assert env.verify_signature()


def test_duration_ms_included_in_canonical_json_for_signing() -> None:
    """Tampering with a record's duration_ms after signing must
    invalidate the envelope signature."""
    env = _make().with_integrity(
        CapabilityIntegrityRecord(
            capability_id="CAP-1", status="success", duration_ms=5.0
        )
    )

    tampered_records = list(env.model_dump()["integrity"])
    tampered_records[0]["duration_ms"] = 5000.0
    tampered = ProvenanceEnvelope(
        **{**env.model_dump(), "integrity": tampered_records}
    )
    assert not tampered.verify_signature()
