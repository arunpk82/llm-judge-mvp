from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    _write_json(d / "manifest.json", {"schema_version": "1.0", "dummy": True})
    _write_judgments(d / "judgments.jsonl", judgments)
    _write_json(d / "metrics.json", metrics)
    return d


def _read_summary(out_dir: Path) -> str:
    p = out_dir / "diff_summary.txt"
    assert p.exists()
    return p.read_text(encoding="utf-8")


def _read_policy(out_dir: Path) -> dict:
    p = out_dir / "policy_result.json"
    assert p.exists(), f"Missing policy_result.json at {p}"
    return json.loads(p.read_text(encoding="utf-8"))


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
    assert "POLICY: VIOLATION" not in _read_summary(out)

    # EPIC-3: policy_result.json must always exist and be machine-readable
    policy = _read_policy(out)
    assert policy["artifact_type"] == "policy_result"
    assert policy["status"] == "OK"
    assert policy["exit_code"] == 0


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

    out = tmp_path / "out_flip"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--fail-on",
            "decision_flip",
        ]
    )
    assert rc == 2
    assert "POLICY: VIOLATION" in _read_summary(out)

    policy = _read_policy(out)
    assert policy["artifact_type"] == "policy_result"
    assert policy["status"] == "VIOLATION"
    assert policy["exit_code"] == 2
    assert policy["decision_flips"] == 1


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

    out = tmp_path / "out_drop"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--max-metric-drop",
            "f1_fail=0.05",
        ]
    )
    assert rc == 2
    summary = _read_summary(out)
    assert "POLICY: VIOLATION" in summary
    assert "Metric drop too large" in summary

    policy = _read_policy(out)
    assert policy["status"] == "VIOLATION"
    assert policy["exit_code"] == 2
    assert policy["tolerances"].get("f1_fail") == 0.05


def test_diff_metric_drop_within_tolerance_is_ok(tmp_path: Path) -> None:
    """
    P0: ensure we don't block healthy PRs.
    baseline - candidate <= tolerance should pass.
    """
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
        metrics={"cohen_kappa": 0.20},
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
        metrics={"cohen_kappa": 0.19},
    )

    out = tmp_path / "out_within_tol"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--max-metric-drop",
            "cohen_kappa=0.02",
            "--fail-on",
            "metric_drop",
        ]
    )
    assert rc == 0
    assert "POLICY: VIOLATION" not in _read_summary(out)

    policy = _read_policy(out)
    assert policy["status"] == "OK"
    assert policy["exit_code"] == 0
    assert policy["tolerances"].get("cohen_kappa") == 0.02


def test_diff_policy_violation_when_candidate_metric_missing(tmp_path: Path) -> None:
    """
    P0: missing metric must be a hard fail (governance).
    """
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
        # cohen_kappa is missing
        metrics={"f1_fail": 0.80},
    )

    out = tmp_path / "out_missing_metric"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--max-metric-drop",
            "cohen_kappa=0.01",
            "--fail-on",
            "metric_drop",
        ]
    )

    # We expect a governance failure. If your current implementation returns a different
    # code for "invalid inputs", switch this to that code—but it should never be rc=0.
    assert rc == 2
    assert "POLICY: VIOLATION" in _read_summary(out)

    policy = _read_policy(out)
    assert policy["status"] == "VIOLATION"
    assert policy["exit_code"] == 2
    assert policy["tolerances"].get("cohen_kappa") == 0.01


@pytest.mark.parametrize(
    "bad_value",
    [
        None,
        "0.7",
        {"x": 1},
        ["0.7"],
    ],
)
def test_diff_policy_violation_when_candidate_metric_not_numeric(
    tmp_path: Path, bad_value: object
) -> None:
    """
    P0: defend against schema corruption.
    Non-numeric metric values must not silently pass.
    """
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
        metrics={"f1_fail": 0.80},
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
        metrics={"f1_fail": bad_value},  # intentionally corrupted
    )

    out = tmp_path / "out_bad_metric"
    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
            "--max-metric-drop",
            "f1_fail=0.01",
            "--fail-on",
            "metric_drop",
        ]
    )

    assert rc == 2
    assert "POLICY: VIOLATION" in _read_summary(out)

    policy = _read_policy(out)
    assert policy["status"] == "VIOLATION"
    assert policy["exit_code"] == 2
    assert policy["tolerances"].get("f1_fail") == 0.01


def test_diff_fails_on_schema_mismatch(tmp_path: Path) -> None:
    base = tmp_path / "baseline"
    cand = tmp_path / "candidate"

    base.mkdir()
    cand.mkdir()

    _write_json(base / "manifest.json", {"schema_version": "1.0"})
    _write_json(base / "metrics.json", {})
    _write_judgments(base / "judgments.jsonl", [])

    _write_json(cand / "manifest.json", {"schema_version": "2.0"})
    _write_json(cand / "metrics.json", {})
    _write_judgments(cand / "judgments.jsonl", [])

    out = tmp_path / "out"

    rc = diff_main(
        [
            "--baseline",
            str(base),
            "--candidate",
            str(cand),
            "--out",
            str(out),
        ]
    )

    assert rc != 0
