from __future__ import annotations

import json
from pathlib import Path

from llm_judge.eval.diff import main as diff_main


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _write_json(p: Path, obj: dict) -> None:
    _write_text(p, json.dumps(obj, indent=2, sort_keys=True))


def _write_judgments(p: Path, lines: list[dict]) -> None:
    txt = "\n".join(json.dumps(x) for x in lines) + "\n"
    _write_text(p, txt)


def _mk_run_dir(root: Path, name: str, judgments: list[dict], metrics: dict) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    _write_json(d / "manifest.json", {"dummy": True})
    _write_judgments(d / "judgments.jsonl", judgments)
    _write_json(d / "metrics.json", metrics)
    return d


def test_diff_ok_no_violations(tmp_path: Path) -> None:
    base = _mk_run_dir(
        tmp_path,
        "baseline",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "fail",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.8, "cohen_kappa": 0.1},
    )
    cand = _mk_run_dir(
        tmp_path,
        "candidate",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "fail",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.8, "cohen_kappa": 0.1},
    )

    out = tmp_path / "out"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--fail-on",
            "none",
        ]
    )
    assert rc == 0
    assert (out / "diff_report.json").exists()
    assert (out / "diff_summary.txt").exists()


def test_diff_policy_violation_on_decision_flip(tmp_path: Path) -> None:
    base = _mk_run_dir(
        tmp_path,
        "baseline",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "pass",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.8},
    )
    cand = _mk_run_dir(
        tmp_path,
        "candidate",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "fail",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.8},
    )

    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--fail-on",
            "decision_flip",
        ]
    )
    assert rc == 2


def test_diff_policy_violation_on_metric_drop(tmp_path: Path) -> None:
    base = _mk_run_dir(
        tmp_path,
        "baseline",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "fail",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.80, "cohen_kappa": 0.20},
    )
    cand = _mk_run_dir(
        tmp_path,
        "candidate",
        judgments=[
            {
                "case_index": 0,
                "rubric_id": "chat_quality",
                "judge_decision": "fail",
                "judge_scores": {"tone": 4},
                "judge_flags": [],
            },
        ],
        metrics={"f1_fail": 0.60, "cohen_kappa": 0.19},
    )

    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--max-metric-drop",
            "f1_fail=0.05",
        ]
    )
    assert rc == 2
