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
        rubric_id="chat_quality",
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

    # Integrity is complete on the happy path (legacy run-level shape).
    assert result.integrity.complete is True
    assert result.integrity.missing_capabilities == []
    assert result.integrity.reason is None

    # CP-2: envelope carries per-capability integrity records (v3).
    assert result.envelope.schema_version == 3
    assert len(result.envelope.integrity) == 4
    # Every successful capability carries a non-negative duration_ms.
    for rec in result.envelope.integrity:
        assert rec.duration_ms is not None, rec
        assert rec.duration_ms >= 0.0, rec
    assert all(r.status == "success" for r in result.envelope.integrity)
    assert {r.capability_id for r in result.envelope.integrity} == {
        "CAP-1",
        "CAP-2",
        "CAP-7",
        "CAP-5",
    }

    # Manifest resolves on disk.
    manifest_path = tmp_path / "runs" / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["manifest_id"] == result.manifest_id
    assert manifest["schema_version"] == 2
    assert manifest["envelope"]["capability_chain"][-1] in ("CAP-7", "CAP-2")
    # The CAP-5 append happens after manifest write; the envelope
    # stored in the manifest reflects the state as seen by CAP-5's
    # entry (post CAP-7, pre CAP-5 stamp).
    assert manifest["verdict"] == result.verdict


def test_runner_signature_verifies_end_to_end(runner: PlatformRunner) -> None:
    req = SingleEvaluationRequest(
        response="A short answer.",
        source="Some reference context for the short answer.",
        rubric_id="chat_quality",
    )
    result = runner.run_single_evaluation(req)
    assert result.envelope.verify_signature()


def test_runner_assigns_request_id_when_absent(runner: PlatformRunner) -> None:
    req = SingleEvaluationRequest(
        response="Text.",
        source="Context.",
        rubric_id="chat_quality",
    )
    assert req.request_id is None
    result = runner.run_single_evaluation(req)
    assert result.envelope.request_id
    assert result.envelope.request_id.startswith("se-")


def test_runner_layers_explicit_opt_in(
    runner: PlatformRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runner forwards ``layers`` to CAP-7; default is ``["L1"]``."""
    captured: dict[str, object] = {}

    from llm_judge.control_plane import wrappers as real_wrappers

    real_invoke_cap7 = real_wrappers.invoke_cap7

    def _spy(envelope, request, dataset_handle, *, layers=None):
        captured["layers"] = layers
        return real_invoke_cap7(
            envelope, request, dataset_handle, layers=layers
        )

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap7", _spy
    )

    req = SingleEvaluationRequest(response="ans", source="ctx", rubric_id="chat_quality")

    # Default: Runner normalizes None → ["L1"] before dispatching.
    runner.run_single_evaluation(req)
    assert captured["layers"] == ["L1"]

    # Explicit opt-in: list propagates unchanged.
    runner.run_single_evaluation(req, layers=["L1"])
    assert captured["layers"] == ["L1"]


def test_runner_default_verdict_reports_l1_only(
    runner: PlatformRunner,
) -> None:
    req = SingleEvaluationRequest(response="ans", source="ctx", rubric_id="chat_quality")
    result = runner.run_single_evaluation(req)
    # CP-1b: verdict advertises which layers ran (default is L1 only).
    assert result.verdict["layers_requested"] == ["L1"]
