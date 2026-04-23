"""Sibling-failure tests: CAP-2 or CAP-7 raising must not abort the run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import (
    MissingProvenanceError,
    SingleEvaluationRequest,
)


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="test-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _request() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(response="ans", source="ctx")


def test_cap2_failure_returns_partial_verdict(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated rule-plan load failure")

    # Patch the symbol the Runner imports, not the wrappers module.
    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap2", _boom
    )

    result = runner.run_single_evaluation(_request())

    assert result.integrity.complete is False
    assert result.integrity.missing_capabilities == ["CAP-2"]
    assert result.integrity.reason is not None
    assert "CAP-2" in result.integrity.reason
    assert "simulated rule-plan load failure" in result.integrity.reason

    # CAP-7 populated the verdict.
    assert "risk_score" in result.verdict
    assert result.verdict["rule_evidence"] == []

    # Manifest still written on partial verdict.
    manifest_path = tmp_path / "runs" / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["integrity"]["complete"] is False
    assert manifest["integrity"]["missing_capabilities"] == ["CAP-2"]

    # CAP-2 is absent from the chain; CAP-1, CAP-7, CAP-5 are present.
    chain = result.envelope.capability_chain
    assert "CAP-1" in chain
    assert "CAP-7" in chain
    assert "CAP-5" in chain
    assert "CAP-2" not in chain


def test_cap7_failure_returns_minimal_verdict_with_rule_evidence(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated hallucination pipeline crash")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap7", _boom
    )

    result = runner.run_single_evaluation(_request())

    assert result.integrity.complete is False
    assert result.integrity.missing_capabilities == ["CAP-7"]
    assert result.integrity.reason is not None
    assert "CAP-7" in result.integrity.reason

    # Verdict is a minimal stub — no CAP-7 fields — but rule_evidence
    # from CAP-2 is attached.
    assert "rule_evidence" in result.verdict
    assert isinstance(result.verdict["rule_evidence"], list)
    assert "risk_score" not in result.verdict

    # CAP-7 absent from chain; CAP-1, CAP-2, CAP-5 present.
    chain = result.envelope.capability_chain
    assert "CAP-1" in chain
    assert "CAP-2" in chain
    assert "CAP-5" in chain
    assert "CAP-7" not in chain

    manifest_path = tmp_path / "runs" / result.manifest_id / "manifest.json"
    assert manifest_path.exists()


def test_cap1_failure_propagates(
    runner: PlatformRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated dataset-registry failure")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap1", _boom
    )

    with pytest.raises(RuntimeError, match="simulated dataset-registry failure"):
        runner.run_single_evaluation(_request())


def test_cap5_failure_propagates(
    runner: PlatformRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated manifest-write failure")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap5", _boom
    )

    with pytest.raises(RuntimeError, match="simulated manifest-write failure"):
        runner.run_single_evaluation(_request())


def test_both_siblings_fail_propagates_cap5_pre_check(
    runner: PlatformRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When both siblings fail, CAP-5's pre-check refuses to write the
    manifest (its guard requires at least one of CAP-2 / CAP-7). The
    Runner does not catch CAP-5 failures, so the error propagates —
    this is the intended loud-failure shape for a fully-degraded run.
    """

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("both siblings down")

    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap2", _boom)
    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap7", _boom)

    with pytest.raises(MissingProvenanceError, match="CAP-2 nor CAP-7"):
        runner.run_single_evaluation(_request())
