"""Happy-path integration test for PlatformRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="test-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def test_runner_full_chain(runner: PlatformRunner, tmp_path: Path) -> None:
    req = SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        caller_id="test-caller",
    )
    result = runner.run_single_evaluation(req)

    # Verdict is non-empty (CAP-7 populated it).
    assert result.verdict
    assert "risk_score" in result.verdict
    assert "rule_evidence" in result.verdict
    assert isinstance(result.verdict["rule_evidence"], list)

    # Chain contains all four capabilities (CAP-2/CAP-7 are siblings,
    # but in this Runner they run sequentially so order is stable).
    chain = result.envelope.capability_chain
    assert "CAP-1" in chain
    assert "CAP-2" in chain
    assert "CAP-7" in chain
    assert "CAP-5" in chain
    assert chain[0] == "CAP-1"
    assert chain[-1] == "CAP-5"

    # Stamps set by CAP-1 and CAP-2.
    assert result.envelope.dataset_registry_id is not None
    assert result.envelope.dataset_registry_id.startswith("transient_")
    assert result.envelope.input_hash
    assert result.envelope.input_hash.startswith("sha256:")
    assert result.envelope.rule_set_version == "v1"
    assert isinstance(result.envelope.rules_fired, list)
    assert result.envelope.platform_version == "test-sha"

    # Integrity is complete on the happy path.
    assert result.integrity.complete is True
    assert result.integrity.missing_capabilities == []
    assert result.integrity.reason is None

    # Manifest resolves on disk.
    manifest_path = tmp_path / "runs" / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_id"] == result.manifest_id
    assert manifest["envelope"]["capability_chain"][-1] in ("CAP-7", "CAP-2")
    # The CAP-5 append happens after manifest write; the envelope
    # stored in the manifest reflects the state as seen by CAP-5's
    # entry (post CAP-7, pre CAP-5 stamp).
    assert manifest["verdict"] == result.verdict


def test_runner_signature_verifies_end_to_end(runner: PlatformRunner) -> None:
    req = SingleEvaluationRequest(
        response="A short answer.",
        source="Some reference context for the short answer.",
    )
    result = runner.run_single_evaluation(req)
    assert result.envelope.verify_signature()


def test_runner_assigns_request_id_when_absent(runner: PlatformRunner) -> None:
    req = SingleEvaluationRequest(
        response="Text.",
        source="Context.",
    )
    assert req.request_id is None
    result = runner.run_single_evaluation(req)
    assert result.envelope.request_id
    assert result.envelope.request_id.startswith("se-")
