"""Bypass-rejection tests: wrappers must raise on absent upstream stamps."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
    new_envelope,
)
from llm_judge.control_plane.types import (
    MissingProvenanceError,
    SingleEvaluationRequest,
)
from llm_judge.control_plane.wrappers import (
    invoke_cap2,
    invoke_cap5,
    invoke_cap7,
)


def _fresh_envelope() -> Any:
    return new_envelope(
        request_id="bypass-test",
        caller_id="test",
        arrived_at=datetime(2026, 4, 22, tzinfo=timezone.utc),
        platform_version="test-sha",
    )


def _request() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(response="r", source="s", rubric_id="chat_quality")


def test_invoke_cap2_rejects_missing_dataset_stamp() -> None:
    env = _fresh_envelope()
    assert env.dataset_registry_id is None
    with pytest.raises(MissingProvenanceError, match="dataset_registry_id"):
        invoke_cap2(env, _request(), MagicMock())


def test_invoke_cap7_rejects_missing_dataset_stamp() -> None:
    env = _fresh_envelope()
    assert env.dataset_registry_id is None
    with pytest.raises(MissingProvenanceError, match="dataset_registry_id"):
        invoke_cap7(env, _request(), MagicMock())


def test_invoke_cap5_rejects_empty_integrity(tmp_path: Path) -> None:
    """CP-1b pre-check: empty integrity is rejected even if other
    fields are present. The integrity trail is what makes the manifest
    auditable; without it CAP-5 refuses to write."""
    env = _fresh_envelope()
    assert env.integrity == []
    with pytest.raises(MissingProvenanceError, match="integrity is empty"):
        invoke_cap5(
            env,
            {},
            {"complete": True},
            rubric_id="chat_quality",
            rubric_version="v1",
            runs_root=tmp_path,
        )


def test_invoke_cap5_accepts_one_integrity_record(tmp_path: Path) -> None:
    """A single integrity record is enough — even a CAP-1 failure is a
    legitimate manifest to write under the horizontal CAP-5 contract."""
    env = _fresh_envelope().with_integrity(
        CapabilityIntegrityRecord(
            capability_id="CAP-1",
            status="failure",
            error_type="DatasetValidationError",
            error_message="simulated",
        )
    )
    env_out, manifest_id = invoke_cap5(
        env,
        {},
        {"complete": False},
        rubric_id="chat_quality",
        rubric_version="v1",
        runs_root=tmp_path,
    )
    assert manifest_id == "bypass-test"
    assert env_out.capability_chain[-1] == "CAP-5"


def test_invoke_cap5_accepts_success_chain(tmp_path: Path) -> None:
    """The happy-path composition still works: CAP-1 stamp + integrity
    record → CAP-5 writes."""
    env = (
        _fresh_envelope()
        .stamped(
            capability="CAP-1",
            dataset_registry_id="transient_x",
            input_hash="sha256:abc",
        )
        .with_integrity(
            CapabilityIntegrityRecord(capability_id="CAP-1", status="success")
        )
    )
    env_out, manifest_id = invoke_cap5(
        env,
        {"risk_score": 0.1},
        {"complete": False},
        rubric_id="chat_quality",
        rubric_version="v1",
        runs_root=tmp_path,
    )
    assert manifest_id == "bypass-test"
    assert env_out.capability_chain[-1] == "CAP-5"
