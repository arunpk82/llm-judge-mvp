"""Field-ownership runtime gate tests (CP-F3 closure).

Mechanism + backward-compat coverage. The cross-capability
gap-absence test lives in test_field_ownership_gap_absence.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from llm_judge.control_plane.envelope import (
    ProvenanceEnvelope,
    new_envelope,
)
from llm_judge.control_plane.field_ownership import FIELD_OWNERSHIP


def _make() -> ProvenanceEnvelope:
    return new_envelope(
        request_id="req-fo",
        caller_id="test",
        arrived_at=datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        platform_version="deadbeef",
    )


# ---------------------------------------------------------------------
# Allowlist shape (recon-confirmed — Pre-flight 2)
# ---------------------------------------------------------------------


def test_field_ownership_lists_recon_confirmed_capabilities() -> None:
    """The four capabilities recon catalogued at packet drafting all
    have entries; missing entries would let stamped() fail closed for
    a wired capability."""
    assert set(FIELD_OWNERSHIP.keys()) >= {"CAP-1", "CAP-2", "CAP-7", "CAP-5"}


def test_cap1_owns_dataset_and_benchmark_provenance_fields() -> None:
    assert FIELD_OWNERSHIP["CAP-1"] == frozenset(
        {
            "dataset_registry_id",
            "input_hash",
            "benchmark_id",
            "benchmark_version",
            "benchmark_content_hash",
            "benchmark_registration_timestamp",
        }
    )


def test_cap2_owns_rule_engine_outcome_fields() -> None:
    assert FIELD_OWNERSHIP["CAP-2"] == frozenset(
        {"rule_set_version", "rules_fired"}
    )


def test_cap7_and_cap5_chain_stamping_only() -> None:
    assert FIELD_OWNERSHIP["CAP-7"] == frozenset()
    assert FIELD_OWNERSHIP["CAP-5"] == frozenset()


# ---------------------------------------------------------------------
# Mechanism — valid stamping passes
# ---------------------------------------------------------------------


def test_cap1_stamping_dataset_fields_passes() -> None:
    env = _make()
    stamped = env.stamped(
        capability="CAP-1",
        dataset_registry_id="transient_abc",
        input_hash="sha256:feed",
    )
    assert stamped.capability_chain == ["CAP-1"]
    assert stamped.dataset_registry_id == "transient_abc"
    assert stamped.verify_signature()


def test_cap1_stamping_benchmark_provenance_passes() -> None:
    env = _make()
    ts = datetime(2026, 4, 15, tzinfo=timezone.utc)
    stamped = env.stamped(
        capability="CAP-1",
        dataset_registry_id="transient_abc",
        input_hash="sha256:feed",
        benchmark_id="ragtruth_50",
        benchmark_version="1.0",
        benchmark_content_hash="sha256:beef",
        benchmark_registration_timestamp=ts,
    )
    assert stamped.benchmark_id == "ragtruth_50"
    assert stamped.benchmark_version == "1.0"
    assert stamped.benchmark_content_hash == "sha256:beef"
    assert stamped.benchmark_registration_timestamp == ts
    assert stamped.verify_signature()


def test_cap2_stamping_rule_engine_fields_passes() -> None:
    env = _make()
    stamped = env.stamped(
        capability="CAP-2",
        rule_set_version="v1",
        rules_fired=["R1", "R2"],
    )
    assert stamped.capability_chain == ["CAP-2"]
    assert stamped.rule_set_version == "v1"
    assert stamped.verify_signature()


def test_cap7_chain_only_stamping_passes() -> None:
    env = _make()
    stamped = env.stamped(capability="CAP-7")
    assert stamped.capability_chain == ["CAP-7"]
    assert stamped.verify_signature()


def test_cap5_chain_only_stamping_passes() -> None:
    env = _make()
    stamped = env.stamped(capability="CAP-5")
    assert stamped.capability_chain == ["CAP-5"]
    assert stamped.verify_signature()


# ---------------------------------------------------------------------
# Backward compat — existing call sites' shapes flow unchanged
# ---------------------------------------------------------------------


def test_full_chain_replays_existing_wrapper_shapes() -> None:
    """The four production stamped() call sites at packet drafting
    are CAP-1 → CAP-2 → CAP-7 → CAP-5. Replay the chain verbatim;
    the strict-from-day-one allowlist must accept it without raising."""
    env = _make()
    env = env.stamped(
        capability="CAP-1",
        dataset_registry_id="transient_abc",
        input_hash="sha256:feed",
    )
    env = env.stamped(
        capability="CAP-2",
        rule_set_version="v1",
        rules_fired=["R1"],
    )
    env = env.stamped(capability="CAP-7")
    env = env.stamped(capability="CAP-5")
    assert env.capability_chain == ["CAP-1", "CAP-2", "CAP-7", "CAP-5"]
    assert env.verify_signature()


def test_provenance_envelope_benchmark_fields_default_to_none() -> None:
    env = _make()
    assert env.benchmark_id is None
    assert env.benchmark_version is None
    assert env.benchmark_content_hash is None
    assert env.benchmark_registration_timestamp is None
