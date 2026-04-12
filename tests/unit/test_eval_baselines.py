from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_judge.eval.baseline import (
    BaselineIntegrityError,
    create_baseline_from_run,
    validate_latest_baseline,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_create_baseline_from_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    baselines = tmp_path / "baselines"

    # minimal run artifacts
    _write(
        run_dir / "manifest.json",
        json.dumps({"rubric": {"id": "chat_quality"}, "dataset_id": "math_basic"}),
    )
    # Legacy filename still supported as input...
    _write(
        run_dir / "results.jsonl",
        '{"case_index":0,"case_id":"c1","rubric_id":"chat_quality","judge_decision":"fail"}\n',
    )
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 0}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    assert (snap_dir / "manifest.json").exists()
    # Snapshot contract is standardized to judgments.jsonl...
    assert (snap_dir / "judgments.jsonl").exists()
    # ...and we also keep a compat copy for older tooling.
    assert (snap_dir / "results.jsonl").exists()
    assert (snap_dir / "metrics.json").exists()

    # Optional: ensure the compat file is the same content as the canonical one
    assert (snap_dir / "results.jsonl").read_text(encoding="utf-8") == (
        snap_dir / "judgments.jsonl"
    ).read_text(encoding="utf-8")

    # Ensure latest.json exists and points to this baseline_id
    latest_path = baselines / ref.suite / ref.rubric_id / "latest.json"
    assert latest_path.exists()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest.get("baseline_id") == ref.baseline_id


def test_create_baseline_prefers_judgments_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    baselines = tmp_path / "baselines"

    _write(
        run_dir / "manifest.json",
        json.dumps({"rubric": {"id": "chat_quality"}, "dataset_id": "math_basic"}),
    )
    _write(
        run_dir / "results.jsonl",
        '{"case_index":0,"case_id":"c1","rubric_id":"chat_quality","judge_decision":"pass"}\n',
    )
    _write(
        run_dir / "judgments.jsonl",
        '{"case_index":0,"case_id":"c1","rubric_id":"chat_quality","judge_decision":"fail"}\n',
    )
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 0}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    data = (snap_dir / "judgments.jsonl").read_text(encoding="utf-8")
    # Should have copied judgments.jsonl (preferred) not results.jsonl
    assert '"judge_decision":"fail"' in data

    # Compat file should match canonical snapshot content.
    assert (snap_dir / "results.jsonl").read_text(encoding="utf-8") == (
        snap_dir / "judgments.jsonl"
    ).read_text(encoding="utf-8")


def test_validate_latest_baseline_happy_path(tmp_path: Path) -> None:
    """
    Governance P0:
    latest.json must point to a snapshot that satisfies snapshot contract:
      - manifest.json
      - metrics.json
      - judgments.jsonl
    """
    baselines = tmp_path / "baselines"

    # Create a baseline via the supported path (writes latest.json)
    run_dir = tmp_path / "run"
    _write(
        run_dir / "manifest.json",
        json.dumps({"rubric": {"id": "chat_quality"}, "dataset_id": "math_basic"}),
    )
    _write(
        run_dir / "judgments.jsonl",
        '{"case_index":0,"case_id":"c1","rubric_id":"chat_quality","judge_decision":"pass"}\n',
    )
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 1}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = validate_latest_baseline(
        suite=ref.suite,
        rubric_id=ref.rubric_id,
        baselines_dir=baselines,
    )

    assert (
        snap_dir
        == baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    )


def test_validate_latest_baseline_fails_when_manifest_missing(tmp_path: Path) -> None:
    """
    Matches your real folder-state issue (a snapshot missing manifest.json).
    This MUST fail deterministically in CI.
    """
    baselines = tmp_path / "baselines"
    suite = "golden"
    rubric_id = "chat_quality"
    baseline_id = "20260303T030650Z"

    # Write latest.json pointer
    latest_path = baselines / suite / rubric_id / "latest.json"
    _write(
        latest_path,
        json.dumps(
            {"suite": suite, "rubric_id": rubric_id, "baseline_id": baseline_id}
        ),
    )

    # Create an incomplete snapshot (missing manifest.json)
    snap_dir = baselines / suite / rubric_id / "snapshots" / baseline_id
    _write(snap_dir / "judgments.jsonl", '{"case_index":0,"case_id":"c1"}\n')
    _write(snap_dir / "metrics.json", json.dumps({"total": 1}))

    with pytest.raises(BaselineIntegrityError):
        validate_latest_baseline(
            suite=suite, rubric_id=rubric_id, baselines_dir=baselines
        )
