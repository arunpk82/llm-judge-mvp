"""Math_basic end-to-end integration test for CP-1c-b.1.

This is the load-bearing verification: a non-chat_quality rubric
must flow correctly through the Runner from request to manifest.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture
def runner(tmp_path: Path) -> PlatformRunner:
    return PlatformRunner(
        platform_version="cp1cb1-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _read_manifest(tmp_path: Path, manifest_id: str) -> dict:
    manifest_path = tmp_path / "runs" / manifest_id / "manifest.json"
    assert manifest_path.exists(), f"manifest missing at {manifest_path}"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_runner_math_basic_end_to_end(
    runner: PlatformRunner, tmp_path: Path
) -> None:
    """CP-1c-b.1 verification: non-chat_quality rubric flows through Runner.

    Constructs a SingleEvaluationRequest with rubric_id="math_basic",
    invokes PlatformRunner.run_single_evaluation, asserts:
    - The Runner accepts the request (no validation failure)
    - The capability chain completes successfully
    - The manifest's rubric_id reflects "math_basic", not "chat_quality"
    - rubric_version is resolved from the "latest" sentinel (default)
    """
    request = SingleEvaluationRequest(
        response="2 + 2 = 4",
        source="Basic arithmetic.",
        rubric_id="math_basic",
    )

    result = runner.run_single_evaluation(request)

    assert result is not None
    assert result.envelope.capability_chain[-1] == "CAP-5"
    assert all(r.status == "success" for r in result.envelope.integrity)

    manifest = _read_manifest(tmp_path, result.manifest_id)
    assert manifest["rubric_id"] == "math_basic"
    # "latest" resolves against rubrics/registry.yaml — math_basic@v1
    assert manifest["rubric_version"] == "v1"


def test_runner_chat_quality_still_works(
    runner: PlatformRunner, tmp_path: Path
) -> None:
    """Regression: chat_quality continues to flow correctly."""
    request = SingleEvaluationRequest(
        response="The sky is blue.",
        source="Sky color: blue.",
        rubric_id="chat_quality",
    )

    result = runner.run_single_evaluation(request)

    assert result is not None
    manifest = _read_manifest(tmp_path, result.manifest_id)
    assert manifest["rubric_id"] == "chat_quality"
    assert manifest["rubric_version"] == "v1"


def test_runner_explicit_version_pins(
    runner: PlatformRunner, tmp_path: Path
) -> None:
    """Explicit rubric_version is honored end-to-end."""
    request = SingleEvaluationRequest(
        response="The sky is blue.",
        source="Sky color: blue.",
        rubric_id="chat_quality",
        rubric_version="v1",
    )

    result = runner.run_single_evaluation(request)

    manifest = _read_manifest(tmp_path, result.manifest_id)
    assert manifest["rubric_id"] == "chat_quality"
    assert manifest["rubric_version"] == "v1"
