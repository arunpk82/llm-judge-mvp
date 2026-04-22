"""Bypass-rejection tests: wrappers must raise on absent upstream stamps."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_judge.control_plane.envelope import new_envelope
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
    return SingleEvaluationRequest(response="r", source="s")


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


def test_invoke_cap5_rejects_empty_chain(tmp_path: Path) -> None:
    env = _fresh_envelope()
    with pytest.raises(MissingProvenanceError, match="CAP-1"):
        invoke_cap5(env, {}, {"complete": True}, runs_root=tmp_path)


def test_invoke_cap5_rejects_cap1_only_chain(tmp_path: Path) -> None:
    env = _fresh_envelope().stamped(
        capability="CAP-1",
        dataset_registry_id="transient_x",
        input_hash="sha256:abc",
    )
    # CAP-1 present, but no sibling (CAP-2 / CAP-7) ran.
    with pytest.raises(MissingProvenanceError, match="CAP-2 nor CAP-7"):
        invoke_cap5(env, {}, {"complete": False}, runs_root=tmp_path)


def test_invoke_cap5_accepts_cap1_plus_cap2(tmp_path: Path) -> None:
    env = _fresh_envelope().stamped(
        capability="CAP-1",
        dataset_registry_id="transient_x",
        input_hash="sha256:abc",
    ).stamped(
        capability="CAP-2",
        rule_set_version="v1",
        rules_fired=[],
    )
    env_out, manifest_id = invoke_cap5(
        env, {"risk_score": 0.1}, {"complete": False}, runs_root=tmp_path
    )
    assert manifest_id == "bypass-test"
    assert env_out.capability_chain[-1] == "CAP-5"
