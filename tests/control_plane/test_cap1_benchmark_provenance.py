"""CAP-1 wrapper benchmark-provenance stamping (CP-F1 closure, Decision 9).

When ``SingleEvaluationRequest.benchmark_reference`` is populated,
``_cap1_lineage_tracking`` stamps the four benchmark provenance
fields onto the envelope under CAP-1's allowlist. When the field
is None (default), only ``dataset_registry_id`` + ``input_hash``
are stamped — backward compat for single-eval and other per-case
flows.

The field-ownership runtime gate (CP-F3) is what enforces that the
six-field shape stays inside CAP-1's territory; this test exercises
the wiring end-to-end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from llm_judge.control_plane.envelope import new_envelope
from llm_judge.control_plane.types import (
    BenchmarkReference,
    SingleEvaluationRequest,
)
from llm_judge.control_plane.wrappers import invoke_cap1


def _make_envelope():
    return new_envelope(
        request_id="req-cap1-bench",
        caller_id="test",
        arrived_at=datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
        platform_version="deadbeef",
    )


def _make_request(*, with_benchmark: bool) -> SingleEvaluationRequest:
    ref = (
        BenchmarkReference(
            benchmark_id="ragtruth_50",
            benchmark_version="1.0",
            benchmark_content_hash="sha256:beef",
            benchmark_registration_timestamp=datetime(
                2026, 4, 15, tzinfo=timezone.utc
            ),
        )
        if with_benchmark
        else None
    )
    return SingleEvaluationRequest(
        response="r",
        source="s",
        rubric_id="chat_quality",
        request_id="req-cap1-bench",
        benchmark_reference=ref,
    )


def test_envelope_lacks_benchmark_fields_when_request_has_no_reference(
    tmp_path: Path,
) -> None:
    """Backward compat: per-case flows leave benchmark provenance fields None."""
    env = _make_envelope()
    request = _make_request(with_benchmark=False)
    stamped, _ = invoke_cap1(env, request, transient_root=tmp_path)
    assert "CAP-1" in stamped.capability_chain
    assert stamped.dataset_registry_id is not None
    assert stamped.input_hash is not None
    assert stamped.benchmark_id is None
    assert stamped.benchmark_version is None
    assert stamped.benchmark_content_hash is None
    assert stamped.benchmark_registration_timestamp is None
    assert stamped.verify_signature()


def test_envelope_carries_benchmark_provenance_when_request_has_reference(
    tmp_path: Path,
) -> None:
    """CP-F1 wiring: benchmark provenance flows from request →
    BenchmarkReference → envelope stamp under CAP-1."""
    env = _make_envelope()
    request = _make_request(with_benchmark=True)
    stamped, _ = invoke_cap1(env, request, transient_root=tmp_path)
    assert stamped.benchmark_id == "ragtruth_50"
    assert stamped.benchmark_version == "1.0"
    assert stamped.benchmark_content_hash == "sha256:beef"
    assert stamped.benchmark_registration_timestamp == datetime(
        2026, 4, 15, tzinfo=timezone.utc
    )
    # Per-case fields stamped alongside benchmark fields.
    assert stamped.dataset_registry_id is not None
    assert stamped.input_hash is not None
    assert stamped.verify_signature()


def test_benchmark_stamping_does_not_violate_field_ownership(
    tmp_path: Path,
) -> None:
    """The strict-from-day-one allowlist must accept the six-field
    CAP-1 stamp by construction. If the wiring were stamping a field
    outside CAP-1's territory, the FIELD_OWNERSHIP runtime gate
    (Commit 1) would raise FieldOwnershipViolationError here."""
    env = _make_envelope()
    request = _make_request(with_benchmark=True)
    stamped, _ = invoke_cap1(env, request, transient_root=tmp_path)
    assert stamped.verify_signature()
