from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_judge.eval.schema import assert_compatible_schema

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_POLICY_VIOLATION = 2

# Policy artifact contract (kept local to avoid coupling to schema.py versions for now)
POLICY_RESULT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class ResolvedRun:
    run_dir: Path
    manifest_path: Path
    case_path: Path
    metrics_path: Path
    case_filename: str


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def _resolve_case_artifact(run_dir: Path) -> tuple[Path, str]:
    """
    Prefer judgments.jsonl. Fallback to results.jsonl.
    """
    p = run_dir / "judgments.jsonl"
    if p.exists():
        return p, "judgments.jsonl"
    p = run_dir / "results.jsonl"
    if p.exists():
        return p, "results.jsonl"
    raise FileNotFoundError(
        f"Missing required per-case artifact: {run_dir/'judgments.jsonl'} (or fallback {run_dir/'results.jsonl'})"
    )


def resolve_run_dir(run_dir: Path) -> ResolvedRun:
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"

    _require(manifest_path)
    _require(metrics_path)

    case_path, case_filename = _resolve_case_artifact(run_dir)

    return ResolvedRun(
        run_dir=run_dir,
        manifest_path=manifest_path,
        case_path=case_path,
        metrics_path=metrics_path,
        case_filename=case_filename,
    )


def resolve_baseline(baseline: Path) -> ResolvedRun:
    """
    baseline can be:
      1) A snapshot directory containing manifest.json/(judgments.jsonl|results.jsonl)/metrics.json
      2) A latest.json pointer file created by baseline-create
    """
    baseline = baseline.resolve()
    if baseline.is_file() and baseline.name.endswith(".json"):
        data = _read_json(baseline)
        baseline_id = data.get("baseline_id")
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise ValueError(f"Invalid baseline pointer file (missing baseline_id): {baseline}")

        snap_dir = baseline.parent / "snapshots" / baseline_id.strip()
        return resolve_run_dir(snap_dir)

    return resolve_run_dir(baseline)


def _load_judgments(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    """
    Keyed by (case_index, rubric_id) to avoid collisions.
    Assumes per-line JSON objects with at least:
      - case_index: int
      - rubric_id: str
    """
    out: dict[tuple[int, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on {path}:{line_no}: {e}") from e

            case_index = obj.get("case_index")
            rubric_id = obj.get("rubric_id")
            if not isinstance(case_index, int):
                raise ValueError(f"Missing/invalid case_index on {path}:{line_no}")
            if not isinstance(rubric_id, str) or not rubric_id.strip():
                raise ValueError(f"Missing/invalid rubric_id on {path}:{line_no}")

            out[(case_index, rubric_id)] = obj
    return out


def _as_set(x: Any) -> set[str]:
    if x is None:
        return set()
    if isinstance(x, list):
        return {str(v) for v in x}
    return {str(x)}


def _numeric(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _diff_metrics(baseline_metrics: dict[str, Any], candidate_metrics: dict[str, Any]) -> dict[str, Any]:
    diffs: dict[str, Any] = {"deltas": {}, "only_in_baseline": [], "only_in_candidate": []}

    b_keys = set(baseline_metrics.keys())
    c_keys = set(candidate_metrics.keys())

    diffs["only_in_baseline"] = sorted(list(b_keys - c_keys))
    diffs["only_in_candidate"] = sorted(list(c_keys - b_keys))

    common = sorted(list(b_keys & c_keys))
    for k in common:
        bv = _numeric(baseline_metrics.get(k))
        cv = _numeric(candidate_metrics.get(k))
        if bv is None or cv is None:
            continue
        diffs["deltas"][k] = {"baseline": bv, "candidate": cv, "delta": cv - bv}

    return diffs


def _diff_judgments(
    base: dict[tuple[int, str], dict[str, Any]],
    cand: dict[tuple[int, str], dict[str, Any]],
) -> dict[str, Any]:
    base_keys = set(base.keys())
    cand_keys = set(cand.keys())

    missing_in_candidate = sorted(list(base_keys - cand_keys))
    new_in_candidate = sorted(list(cand_keys - base_keys))
    common = sorted(list(base_keys & cand_keys))

    decision_flips: list[dict[str, Any]] = []
    score_deltas: list[dict[str, Any]] = []
    flag_diffs: list[dict[str, Any]] = []

    for key in common:
        b = base[key]
        c = cand[key]

        b_dec = b.get("judge_decision")
        c_dec = c.get("judge_decision")
        if isinstance(b_dec, str) and isinstance(c_dec, str) and b_dec != c_dec:
            decision_flips.append(
                {"case_index": key[0], "rubric_id": key[1], "baseline": b_dec, "candidate": c_dec}
            )

        b_scores_any = b.get("judge_scores")
        c_scores_any = c.get("judge_scores")
        b_scores: dict[str, Any] = b_scores_any if isinstance(b_scores_any, dict) else {}
        c_scores: dict[str, Any] = c_scores_any if isinstance(c_scores_any, dict) else {}

        dims = sorted(set(b_scores.keys()) | set(c_scores.keys()))
        deltas: dict[str, Any] = {}
        for d in dims:
            bv = _numeric(b_scores.get(d))
            cv = _numeric(c_scores.get(d))
            if bv is None or cv is None:
                continue
            delta = cv - bv
            if delta != 0:
                deltas[d] = {"baseline": bv, "candidate": cv, "delta": delta}

        if deltas:
            score_deltas.append({"case_index": key[0], "rubric_id": key[1], "deltas": deltas})

        b_flags = _as_set(b.get("judge_flags"))
        c_flags = _as_set(c.get("judge_flags"))
        if b_flags != c_flags:
            flag_diffs.append(
                {
                    "case_index": key[0],
                    "rubric_id": key[1],
                    "added": sorted(list(c_flags - b_flags)),
                    "removed": sorted(list(b_flags - c_flags)),
                }
            )

    return {
        "n_cases_baseline": len(base),
        "n_cases_candidate": len(cand),
        "missing_in_candidate": [{"case_index": k[0], "rubric_id": k[1]} for k in missing_in_candidate],
        "new_in_candidate": [{"case_index": k[0], "rubric_id": k[1]} for k in new_in_candidate],
        "decision_flips": decision_flips,
        "score_deltas": score_deltas,
        "flag_diffs": flag_diffs,
    }


def _parse_metric_drop_rules(values: list[str]) -> dict[str, float]:
    """
    --max-metric-drop f1_fail=0.01 --max-metric-drop cohen_kappa=0.02
    Means: baseline - candidate <= tolerance
    """
    out: dict[str, float] = {}
    for v in values:
        if "=" not in v:
            raise ValueError(f"Invalid --max-metric-drop '{v}'. Expected key=value")
        k, s = v.split("=", 1)
        k = k.strip()
        s = s.strip()
        if not k:
            raise ValueError(f"Invalid --max-metric-drop '{v}'. Empty key")
        try:
            tol = float(s)
        except ValueError as e:
            raise ValueError(f"Invalid --max-metric-drop '{v}'. Value must be float") from e
        out[k] = tol
    return out


def _check_policy(
    *,
    diff: dict[str, Any],
    fail_on_decision_flip: bool,
    max_abs_score_delta: float | None,
    max_metric_drop: dict[str, float],
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> list[str]:
    violations: list[str] = []

    j = diff["judgments"]

    if fail_on_decision_flip and j["decision_flips"]:
        violations.append(f"Decision flips detected: {len(j['decision_flips'])}")

    if max_abs_score_delta is not None:
        for item in j["score_deltas"]:
            for dim, d in item["deltas"].items():
                if abs(float(d["delta"])) > max_abs_score_delta:
                    violations.append(
                        f"Score delta too large: case_index={item['case_index']} rubric_id={item['rubric_id']} "
                        f"dim={dim} delta={d['delta']} (threshold={max_abs_score_delta})"
                    )

    for k, tol in max_metric_drop.items():
        b_has_key = k in baseline_metrics
        c_has_key = k in candidate_metrics

        if not b_has_key and not c_has_key:
            continue
        if not b_has_key or not c_has_key:
            violations.append(f"Metric '{k}' not present in both baseline and candidate metrics.")
            continue

        b_val = _numeric(baseline_metrics[k])
        c_val = _numeric(candidate_metrics[k])

        if b_val is None and c_val is None:
            continue
        if b_val is None or c_val is None:
            violations.append(
                f"Metric '{k}' has incompatible types: "
                f"baseline={baseline_metrics[k]!r} candidate={candidate_metrics[k]!r}"
            )
            continue

        drop = b_val - c_val
        if drop > tol:
            violations.append(
                f"Metric drop too large: {k} baseline={b_val} candidate={c_val} drop={drop} (tolerance={tol})"
            )

    return violations


def _build_summary(diff: dict[str, Any], violations: list[str]) -> str:
    j = diff["judgments"]
    lines: list[str] = []
    lines.append("LLM-Judge Eval Diff Summary")
    lines.append("=" * 28)
    lines.append(f"Baseline:  {diff['baseline_dir']}")
    lines.append(f"Candidate: {diff['candidate_dir']}")
    lines.append(f"Schema:    baseline={diff.get('baseline_schema_version')} candidate={diff.get('candidate_schema_version')}")
    lines.append("")
    lines.append(f"Cases baseline:  {j['n_cases_baseline']}")
    lines.append(f"Cases candidate: {j['n_cases_candidate']}")
    lines.append(f"Decision flips:  {len(j['decision_flips'])}")
    lines.append(f"Score deltas:    {len(j['score_deltas'])}")
    lines.append(f"Flag diffs:      {len(j['flag_diffs'])}")
    lines.append("")
    if violations:
        lines.append("POLICY: VIOLATION")
        for v in violations[:25]:
            lines.append(f"- {v}")
        if len(violations) > 25:
            lines.append(f"... ({len(violations) - 25} more)")
    else:
        lines.append("POLICY: OK")
    lines.append("")
    return "\n".join(lines)


def _policy_result(
    *,
    diff: dict[str, Any],
    violations: list[str],
    exit_code: int,
    fail_on: str,
    max_abs_score_delta: float | None,
    max_metric_drop: dict[str, float],
) -> dict[str, Any]:
    j = diff["judgments"]

    # Capture metric violations as a structured subset (useful for dashboards).
    metric_violations: list[str] = []
    for v in violations:
        if v.startswith("Metric drop too large") or v.startswith("Metric '") or v.startswith("Metric "):
            metric_violations.append(v)

    return {
        "schema_version": POLICY_RESULT_SCHEMA_VERSION,
        "artifact_type": "policy_result",
        "status": "VIOLATION" if violations else "OK",
        "exit_code": exit_code,
        "baseline_dir": diff["baseline_dir"],
        "candidate_dir": diff["candidate_dir"],
        "baseline_schema_version": diff.get("baseline_schema_version"),
        "candidate_schema_version": diff.get("candidate_schema_version"),
        "decision_flips": len(j["decision_flips"]),
        "score_deltas": len(j["score_deltas"]),
        "flag_diffs": len(j["flag_diffs"]),
        "fail_on": fail_on,
        "max_abs_score_delta": max_abs_score_delta,
        "tolerances": dict(max_metric_drop),
        "metric_violations": metric_violations,
        "violations": list(violations),
    }


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-judge-eval-diff")
    p.add_argument("--baseline", required=True, help="Baseline snapshot dir OR latest.json pointer")
    p.add_argument("--candidate", required=True, help="Candidate run directory")
    p.add_argument("--out", default=None, help="Output directory for diff artifacts (default: <candidate>/diff)")
    p.add_argument(
        "--fail-on",
        choices=["none", "decision_flip", "metric_drop"],
        default="none",
        help="Policy: fail if condition is met",
    )
    p.add_argument(
        "--max-abs-score-delta",
        default=None,
        type=float,
        help="Policy: fail if any abs(judge_score_delta) exceeds this value",
    )
    p.add_argument(
        "--max-metric-drop",
        action="append",
        default=[],
        help="Policy: fail if baseline - candidate metric exceeds tolerance. Format: key=value. Repeatable.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_cli().parse_args(argv)

    try:
        baseline_run = resolve_baseline(Path(args.baseline))
        candidate_run = resolve_run_dir(Path(args.candidate))

        base_manifest = _read_json(baseline_run.manifest_path)
        cand_manifest = _read_json(candidate_run.manifest_path)

        # --- Production-grade artifact contract enforcement ---
        assert_compatible_schema(
            baseline_version=base_manifest.get("schema_version"),
            candidate_version=cand_manifest.get("schema_version"),
        )

        base_j = _load_judgments(baseline_run.case_path)
        cand_j = _load_judgments(candidate_run.case_path)

        base_m = _read_json(baseline_run.metrics_path)
        cand_m = _read_json(candidate_run.metrics_path)

        diff = {
            "baseline_dir": str(baseline_run.run_dir),
            "candidate_dir": str(candidate_run.run_dir),
            "baseline_schema_version": base_manifest.get("schema_version"),
            "candidate_schema_version": cand_manifest.get("schema_version"),
            "judgments": _diff_judgments(base_j, cand_j),
            "metrics": _diff_metrics(base_m, cand_m),
        }

        max_metric_drop = _parse_metric_drop_rules(args.max_metric_drop)

        violations = _check_policy(
            diff=diff,
            fail_on_decision_flip=(args.fail_on == "decision_flip"),
            max_abs_score_delta=args.max_abs_score_delta,
            max_metric_drop=max_metric_drop,
            baseline_metrics=base_m,
            candidate_metrics=cand_m,
        )

        out_dir = Path(args.out) if args.out else (candidate_run.run_dir / "diff")
        out_dir = out_dir.resolve()

        exit_code = EXIT_POLICY_VIOLATION if violations else EXIT_OK

        # Rich diff artifact (machine)
        _write_json(out_dir / "diff_report.json", {"diff": diff, "violations": violations})

        # Human summary
        summary = _build_summary(diff, violations)
        _write_text(out_dir / "diff_summary.txt", summary)

        # Policy result (machine, stable)
        policy = _policy_result(
            diff=diff,
            violations=violations,
            exit_code=exit_code,
            fail_on=args.fail_on,
            max_abs_score_delta=args.max_abs_score_delta,
            max_metric_drop=max_metric_drop,
        )
        _write_json(out_dir / "policy_result.json", policy)

        print(summary)

        return exit_code

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def compute_diff_summary(*, baseline: ResolvedRun, candidate: ResolvedRun) -> dict[str, Any]:
    """
    Machine-readable diff summary used by CI gates and baseline promotion.

    Returns a stable dict containing:
      - metrics_diffs (numeric deltas)
      - judgment_diffs (flips/deltas/flag_diffs)
      - schema compatibility metadata
    """
    b_manifest = _read_json(baseline.manifest_path)
    c_manifest = _read_json(candidate.manifest_path)

    # Enforce schema compatibility (already used in CLI)
    assert_compatible_schema(
        baseline_version=b_manifest.get("schema_version"),
        candidate_version=c_manifest.get("schema_version"),
    )

    b_metrics = _read_json(baseline.metrics_path)
    c_metrics = _read_json(candidate.metrics_path)

    b_j = _load_judgments(baseline.case_path)
    c_j = _load_judgments(candidate.case_path)

    return {
        "baseline": {"run_dir": str(baseline.run_dir)},
        "candidate": {"run_dir": str(candidate.run_dir)},
        "metrics": _diff_metrics(b_metrics, c_metrics),
        "judgments": _diff_judgments(b_j, c_j),
    }

if __name__ == "__main__":
    raise SystemExit(main())