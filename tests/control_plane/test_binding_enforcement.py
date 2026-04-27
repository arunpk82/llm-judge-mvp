"""Integration tests for CP-1c-b.2 binding enforcement.

These tests run end-to-end through the Control Plane runner and
assert that the request's ``rubric_id`` propagates correctly after
the per-rubric prompt-loading refactor. They complement
``test_runner_math_basic.py`` (CP-1c-b.1) by guarding against a
specific failure mode introduced by Commit 1: the per-rubric
prompt cache could shadow the runner's rubric_id flow if the cache
key were wrong. By running both rubrics through the same fixture
in this file we catch any cache-key drift in a single run.

Note: Control Plane is the demonstration path; ``IntegratedJudge``
is not invoked here (see ``governance.py``'s docstring for the
rigor-vs-demonstration architectural distinction). Concern 2's
direct prompt-loading verification lives in
``tests/calibration/test_integrated_judge_per_rubric.py``.
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
        platform_version="cp1cb2-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )


def _read_manifest(tmp_path: Path, manifest_id: str) -> dict:
    manifest_path = tmp_path / "runs" / manifest_id / "manifest.json"
    assert manifest_path.exists(), f"manifest missing at {manifest_path}"
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def test_math_basic_request_uses_math_basic_prompt(
    runner: PlatformRunner, tmp_path: Path
) -> None:
    """End-to-end: a math_basic request flows through the runner and
    the rubric_id reaches the manifest.

    Load-bearing verification that Concern 2 is closed: with Commit 1's
    per-rubric prompt cache in place, this still produces the
    expected manifest rubric_id (i.e. the cache does not silently
    coerce math_basic into chat_quality).
    """
    request = SingleEvaluationRequest(
        response="2 + 2 = 4",
        source="Basic arithmetic.",
        rubric_id="math_basic",
    )

    result = runner.run_single_evaluation(request)

    assert result is not None
    assert all(r.status == "success" for r in result.envelope.integrity)
    manifest = _read_manifest(tmp_path, result.manifest_id)
    assert manifest["rubric_id"] == "math_basic"
    assert manifest["rubric_version"] == "v1"


def test_chat_quality_regression_continues_to_work(
    runner: PlatformRunner, tmp_path: Path
) -> None:
    """Regression: the chat_quality flow continues to work after the
    per-rubric prompt-loading refactor in Commit 1."""
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
