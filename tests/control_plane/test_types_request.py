"""Tests for SingleEvaluationRequest rubric_id binding (CP-1c-b.1)."""
import pytest
from pydantic import ValidationError
from llm_judge.control_plane.types import SingleEvaluationRequest


def test_request_requires_rubric_id():
    """Pydantic rejects requests without rubric_id."""
    with pytest.raises(ValidationError):
        SingleEvaluationRequest(
            response="The sky is blue.",
            source="Sky color: blue.",
        )


def test_request_rejects_empty_rubric_id():
    """Empty string fails min_length=1."""
    with pytest.raises(ValidationError):
        SingleEvaluationRequest(
            response="The sky is blue.",
            source="Sky color: blue.",
            rubric_id="",
        )


def test_request_default_version_is_latest():
    """rubric_version defaults to 'latest'."""
    req = SingleEvaluationRequest(
        response="The sky is blue.",
        source="Sky color: blue.",
        rubric_id="chat_quality",
    )
    assert req.rubric_version == "latest"


def test_request_accepts_explicit_version():
    """rubric_version can be pinned."""
    req = SingleEvaluationRequest(
        response="...",
        source="...",
        rubric_id="chat_quality",
        rubric_version="v1",
    )
    assert req.rubric_version == "v1"
