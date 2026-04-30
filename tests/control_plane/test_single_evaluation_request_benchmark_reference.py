"""Tests for the ``benchmark_reference`` extension on
:class:`SingleEvaluationRequest` (CP-F1 closure, Decision 8).

Field is optional + default-None so existing per-case callers
(single-eval, demo, manually-constructed requests) are unaffected.
The CAP-1 wrapper update (Commit 5) is what consumes the field;
this commit only verifies the schema shape and the backward-compat
property of the new field.
"""

from __future__ import annotations

from datetime import datetime, timezone

from llm_judge.control_plane.types import (
    BenchmarkReference,
    SingleEvaluationRequest,
)


def _make_reference() -> BenchmarkReference:
    return BenchmarkReference(
        benchmark_id="ragtruth_50",
        benchmark_version="1.0",
        benchmark_content_hash="sha256:beef",
        benchmark_registration_timestamp=datetime(
            2026, 4, 15, tzinfo=timezone.utc
        ),
    )


def test_benchmark_reference_field_defaults_to_none() -> None:
    req = SingleEvaluationRequest(
        response="r",
        source="s",
        rubric_id="chat_quality",
    )
    assert req.benchmark_reference is None


def test_existing_per_case_construction_unchanged() -> None:
    """Backward compat: single-eval callers that constructed
    SingleEvaluationRequest before this commit must keep working
    with no kwargs change."""
    req = SingleEvaluationRequest(
        response="r",
        source="s",
        rubric_id="chat_quality",
        caller_id="single-eval-driver",
        request_id="case-001",
    )
    assert req.caller_id == "single-eval-driver"
    assert req.request_id == "case-001"
    assert req.benchmark_reference is None


def test_benchmark_reference_round_trip() -> None:
    ref = _make_reference()
    req = SingleEvaluationRequest(
        response="r",
        source="s",
        rubric_id="chat_quality",
        benchmark_reference=ref,
    )
    assert req.benchmark_reference == ref
    assert req.benchmark_reference.benchmark_id == "ragtruth_50"


def test_request_remains_frozen() -> None:
    """Schema extension does not loosen the frozen contract."""
    import pytest
    from pydantic import ValidationError

    req = SingleEvaluationRequest(
        response="r",
        source="s",
        rubric_id="chat_quality",
    )
    with pytest.raises(ValidationError):
        req.benchmark_reference = _make_reference()  # type: ignore[misc]
