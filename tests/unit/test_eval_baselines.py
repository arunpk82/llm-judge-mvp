from __future__ import annotations

import json
from pathlib import Path

from llm_judge.eval.baseline import create_baseline_from_run


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
        '{"case_index":0,"rubric_id":"chat_quality","judge_decision":"fail"}\n',
    )
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 1}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    assert (snap_dir / "manifest.json").exists()
    # ...but snapshot contract is standardized to judgments.jsonl
    assert (snap_dir / "judgments.jsonl").exists()
    assert (snap_dir / "metrics.json").exists()


def test_create_baseline_prefers_judgments_jsonl(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    baselines = tmp_path / "baselines"

    _write(
        run_dir / "manifest.json",
        json.dumps({"rubric": {"id": "chat_quality"}, "dataset_id": "math_basic"}),
    )
    _write(
        run_dir / "results.jsonl",
        '{"case_index":0,"rubric_id":"chat_quality","judge_decision":"pass"}\n',
    )
    _write(
        run_dir / "judgments.jsonl",
        '{"case_index":0,"rubric_id":"chat_quality","judge_decision":"fail"}\n',
    )
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 0}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    data = (snap_dir / "judgments.jsonl").read_text(encoding="utf-8")
    # Should have copied judgments.jsonl (preferred) not results.jsonl
    assert '"judge_decision":"fail"' in data
