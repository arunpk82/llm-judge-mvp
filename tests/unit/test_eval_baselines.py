from __future__ import annotations

import json
from pathlib import Path

from llm_judge.eval.baseline import create_baseline_from_run


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def test_create_baseline_from_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    baselines = tmp_path / "baselines"

    # minimal run artifacts
    _write(
        run_dir / "manifest.json",
        json.dumps({"rubric": {"id": "chat_quality"}, "dataset_id": "math_basic"}),
    )
    _write(run_dir / "results.jsonl", '{"case_id":"c1","score":1}\n')
    _write(run_dir / "metrics.json", json.dumps({"total": 1, "pass": 1}))

    ref = create_baseline_from_run(run_dir=run_dir, baselines_dir=baselines)

    snap_dir = baselines / ref.suite / ref.rubric_id / "snapshots" / ref.baseline_id
    assert (snap_dir / "manifest.json").exists()
    assert (snap_dir / "results.jsonl").exists()
    assert (snap_dir / "metrics.json").exists()

    latest = baselines / ref.suite / ref.rubric_id / "latest.json"
    assert latest.exists()
    data = json.loads(latest.read_text(encoding="utf-8"))
    assert data["baseline_id"] == ref.baseline_id
