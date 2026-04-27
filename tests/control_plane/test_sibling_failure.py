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
    return SingleEvaluationRequest(response="ans", source="ctx", rubric_id="chat_quality")


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
    # CP-1b: rule_evidence is attached only when CAP-2 ran; CAP-2
    # failed here, so the key is absent rather than an empty list.
    assert "rule_evidence" not in result.verdict

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


def test_cap1_failure_is_captured_not_propagated(
    runner: PlatformRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CP-1b: CAP-1 is now wrapped by the Runner (D5). Its failure is
    recorded in the envelope's integrity list and CAP-2 / CAP-7 are
    marked skipped_upstream_failure. The Runner does not raise."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("simulated dataset-registry failure")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap1", _boom
    )

    result = runner.run_single_evaluation(_request())

    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-1"] == "failure"
    assert statuses["CAP-2"] == "skipped_upstream_failure"
    assert statuses["CAP-7"] == "skipped_upstream_failure"
    assert statuses["CAP-5"] == "success"
    assert result.integrity.complete is False
    assert result.integrity.missing_capabilities == ["CAP-1", "CAP-2", "CAP-7"]


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


def test_both_siblings_fail_writes_manifest(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CP-1b: when both siblings fail but CAP-1 succeeded, CAP-5 still
    writes a manifest. The integrity trail records all three failures;
    verdict is empty. The MissingProvenanceError escape hatch in CP-1
    is gone — the horizontal CAP-5 contract replaces it."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("both siblings down")

    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap2", _boom)
    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap7", _boom)

    # MissingProvenanceError remains a valid import but is not raised
    # on this path anymore.
    _ = MissingProvenanceError  # keep import resolved

    result = runner.run_single_evaluation(_request())

    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-1"] == "success"
    assert statuses["CAP-2"] == "failure"
    assert statuses["CAP-7"] == "failure"
    assert statuses["CAP-5"] == "success"
    assert result.integrity.complete is False
    assert result.verdict == {}
    manifest_path = tmp_path / "runs" / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
