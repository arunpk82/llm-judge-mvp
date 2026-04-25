"""Degradation scenarios (CP-1b A1 + A2).

These tests cover what CP-1 could not: scenarios in which one or
more capabilities fail and CAP-5 still writes a durable manifest
that records the outcome. All manifests land under ``tmp_path`` so
no artifact survives the test run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="test-sha",
        transient_root=tmp_path / "transient",
    )


def _request() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="answer candidate",
        source="reference context",
        caller_id="degradation-tests",
    )


def _manifest_statuses(manifest: dict[str, Any]) -> dict[str, str]:
    return {
        r["capability_id"]: r["status"]
        for r in manifest.get("envelope_integrity", [])
    }


def test_cap2_failure_cap7_success(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAP-2 fails but CAP-7 succeeds → partial verdict with CAP-7's
    output and a failure record for CAP-2."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("RuleSetLoadError: simulated plan failure")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap2", _boom
    )

    output_dir = tmp_path / "runs"
    result = runner.run_single_evaluation(_request(), output_dir=output_dir)

    # Runner did not raise.
    assert result is not None

    # Verdict contains CAP-7 output.
    assert "risk_score" in result.verdict

    # Envelope integrity reflects the per-capability outcomes.
    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-1"] == "success"
    assert statuses["CAP-2"] == "failure"
    assert statuses["CAP-7"] == "success"
    assert statuses["CAP-5"] == "success"

    # CP-2: every record (success OR failure) carries duration_ms.
    by_id = {r.capability_id: r for r in result.envelope.integrity}
    for cid in ("CAP-1", "CAP-2", "CAP-7", "CAP-5"):
        assert by_id[cid].duration_ms is not None, cid
        assert by_id[cid].duration_ms >= 0.0, cid

    # Manifest on disk under tmp_path; schema_version=2.
    manifest_path = output_dir / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2

    # Manifest's envelope_integrity mirrors the result's envelope.integrity
    # (as of CAP-5 entry — CAP-5's own record is appended post-write).
    man_statuses = _manifest_statuses(manifest)
    assert man_statuses["CAP-1"] == "success"
    assert man_statuses["CAP-2"] == "failure"
    assert man_statuses["CAP-7"] == "success"
    # CAP-2's failure record includes error_type/message.
    cap2_rec = next(
        r for r in manifest["envelope_integrity"] if r["capability_id"] == "CAP-2"
    )
    assert cap2_rec["error_type"] == "RuntimeError"
    assert "RuleSetLoadError" in (cap2_rec["error_message"] or "")
    # CP-2: CAP-2's failure record still carries duration_ms.
    assert cap2_rec["duration_ms"] is not None
    assert cap2_rec["duration_ms"] >= 0.0


def test_cap1_total_failure(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAP-1 fails → siblings skipped → manifest still written with a
    full integrity trail and no verdict. This is the degradation case
    CP-1 could not record."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("DatasetValidationError: simulated CAP-1 failure")

    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap1", _boom
    )

    output_dir = tmp_path / "runs"
    result = runner.run_single_evaluation(_request(), output_dir=output_dir)

    # Runner did not raise.
    assert result is not None

    # Verdict is empty (no evaluation ran).
    assert result.verdict == {}

    # Integrity records: CAP-1 failure, CAP-2/CAP-7 skipped, CAP-5 success.
    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-1"] == "failure"
    assert statuses["CAP-2"] == "skipped_upstream_failure"
    assert statuses["CAP-7"] == "skipped_upstream_failure"
    assert statuses["CAP-5"] == "success"

    # Runner-level fields still populated on the envelope.
    assert result.envelope.request_id
    assert result.envelope.caller_id == "degradation-tests"
    assert result.envelope.arrived_at is not None
    assert result.envelope.platform_version == "test-sha"

    # CAP-1 never stamped the envelope.
    assert result.envelope.dataset_registry_id is None
    assert result.envelope.input_hash is None

    # Manifest exists under tmp_path with schema_version=2.
    manifest_path = output_dir / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2

    # Manifest's envelope echoes the missing CAP-1 stamps.
    assert manifest["envelope"]["dataset_registry_id"] is None
    assert manifest["envelope"]["input_hash"] is None

    # Manifest integrity trail matches the envelope integrity (CAP-5's
    # own success record is appended after the manifest is written).
    man_statuses = _manifest_statuses(manifest)
    assert man_statuses["CAP-1"] == "failure"
    assert man_statuses["CAP-2"] == "skipped_upstream_failure"
    assert man_statuses["CAP-7"] == "skipped_upstream_failure"


def test_both_siblings_fail_cap5_still_writes(
    runner: PlatformRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both siblings fail after CAP-1 success → CAP-5 writes a manifest
    capturing two failure records. This is the specific degradation
    CP-1 could not record because CAP-5's old pre-check refused
    envelopes without sibling stamps."""

    def _boom(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("both siblings down")

    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap2", _boom)
    monkeypatch.setattr("llm_judge.control_plane.runner.invoke_cap7", _boom)

    output_dir = tmp_path / "runs"
    result = runner.run_single_evaluation(_request(), output_dir=output_dir)

    # Runner did not raise.
    assert result is not None

    # Verdict is empty (no CAP-7 output, CAP-2's rule_evidence not attached
    # because CAP-2 failed).
    assert result.verdict == {}

    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-1"] == "success"
    assert statuses["CAP-2"] == "failure"
    assert statuses["CAP-7"] == "failure"
    assert statuses["CAP-5"] == "success"

    manifest_path = output_dir / result.manifest_id / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    man_statuses = _manifest_statuses(manifest)
    assert man_statuses["CAP-1"] == "success"
    assert man_statuses["CAP-2"] == "failure"
    assert man_statuses["CAP-7"] == "failure"
