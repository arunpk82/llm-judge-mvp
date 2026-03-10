from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

EXIT_OK = 0
EXIT_RUNTIME_ERROR = 1
EXIT_POLICY_VIOLATION = 2

DEFAULT_REGISTRY_PATH = Path("reports/run_registry.jsonl")
DEFAULT_REPORT_PATH = Path("reports/drift/drift_report.json")


@dataclass(frozen=True)
class DriftPolicy:
    schema_version: str
    policy_id: str

    # point drift: baseline -> latest
    required_metrics: list[str]
    max_metric_drop: dict[str, float]     # candidate-baseline >= -tol
    min_metric_value: dict[str, float]    # candidate >= min

    # trend drift: window oldest -> newest
    trend_window: int
    max_trend_drop: dict[str, float]      # newest-oldest >= -tol

    # selection
    dataset_id: str | None
    rubric_id: str | None
    judge_engine: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def _numeric(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def load_policy(path: Path) -> DriftPolicy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("drift policy must be a YAML mapping")

    def _list(key: str) -> list[str]:
        v = raw.get(key, [])
        if not isinstance(v, list):
            return []
        return [str(x) for x in v]

    def _dict_float(key: str) -> dict[str, float]:
        v = raw.get(key, {})
        if not isinstance(v, dict):
            return {}
        out: dict[str, float] = {}
        for k, val in v.items():
            try:
                out[str(k)] = float(val)
            except Exception:
                continue
        return out

    def _opt_str(key: str) -> str | None:
        v = raw.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    schema_version = str(raw.get("schema_version", "1")).strip()
    policy_id = str(raw.get("policy_id", "drift_policy")).strip()

    return DriftPolicy(
        schema_version=schema_version,
        policy_id=policy_id,
        required_metrics=_list("required_metrics"),
        max_metric_drop=_dict_float("max_metric_drop"),
        min_metric_value=_dict_float("min_metric_value"),
        trend_window=int(raw.get("trend_window", 20)),
        max_trend_drop=_dict_float("max_trend_drop"),
        dataset_id=_opt_str("dataset_id"),
        rubric_id=_opt_str("rubric_id"),
        judge_engine=_opt_str("judge_engine"),
    )


def _select_entries(
    entries: list[dict[str, Any]],
    *,
    dataset_id: str | None,
    rubric_id: str | None,
    judge_engine: str | None,
) -> list[dict[str, Any]]:
    out = entries
    if dataset_id is not None:
        out = [e for e in out if str(e.get("dataset_id")) == dataset_id]
    if rubric_id is not None:
        out = [e for e in out if str(e.get("rubric_id")) == rubric_id]
    if judge_engine is not None:
        out = [e for e in out if str(e.get("judge_engine")) == judge_engine]
    # created_at_utc is ISO-like "....Z" in our registry writer; string sort works.
    out.sort(key=lambda e: str(e.get("created_at_utc", "")))
    return out


def _resolve_baseline_metrics(*, suite: str, rubric_id: str, baselines_dir: Path) -> tuple[Path, dict[str, Any]]:
    latest_ptr = baselines_dir / suite / rubric_id / "latest.json"
    if not latest_ptr.exists():
        raise FileNotFoundError(f"Missing baseline pointer: {latest_ptr}")

    data = _read_json(latest_ptr)
    baseline_id = data.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id.strip():
        raise ValueError(f"Invalid baseline pointer (missing baseline_id): {latest_ptr}")

    snap_dir = latest_ptr.parent / "snapshots" / baseline_id.strip()
    metrics_path = snap_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline metrics.json: {metrics_path}")

    return metrics_path, _read_json(metrics_path)


def _point_drift_checks(
    *,
    policy: DriftPolicy,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    violations: list[str] = []
    checks: list[dict[str, Any]] = []

    # required metrics presence
    for m in policy.required_metrics:
        if m not in baseline_metrics:
            violations.append(f"Baseline missing required metric: {m}")
        if m not in candidate_metrics:
            violations.append(f"Candidate missing required metric: {m}")

    # min value checks on candidate
    for m, minv in policy.min_metric_value.items():
        cv = _numeric(candidate_metrics.get(m))
        if cv is None:
            continue
        ok = cv >= minv
        checks.append({"type": "min_metric_value", "metric": m, "candidate": cv, "min": minv, "ok": ok})
        if not ok:
            violations.append(f"Metric below minimum: {m} candidate={cv} min={minv}")

    # max drop checks: (candidate - baseline) >= -tol
    for m, tol in policy.max_metric_drop.items():
        bv = _numeric(baseline_metrics.get(m))
        cv = _numeric(candidate_metrics.get(m))
        if bv is None or cv is None:
            continue
        delta = cv - bv
        ok = delta >= -tol
        checks.append(
            {
                "type": "max_metric_drop",
                "metric": m,
                "baseline": bv,
                "candidate": cv,
                "delta": delta,
                "tolerance": tol,
                "ok": ok,
            }
        )
        if not ok:
            violations.append(f"Metric drop too large: {m} baseline={bv} candidate={cv} drop={-delta} tol={tol}")

    return checks, violations


def _trend_drift_checks(
    *,
    policy: DriftPolicy,
    window_entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    violations: list[str] = []
    checks: list[dict[str, Any]] = []

    if len(window_entries) < 2:
        return checks, violations

    oldest = window_entries[0]
    newest = window_entries[-1]

    oldest_metrics = oldest.get("metrics", {})
    newest_metrics = newest.get("metrics", {})

    if not isinstance(oldest_metrics, dict) or not isinstance(newest_metrics, dict):
        return checks, violations

    for m, tol in policy.max_trend_drop.items():
        ov = _numeric(oldest_metrics.get(m))
        nv = _numeric(newest_metrics.get(m))
        if ov is None or nv is None:
            continue
        delta = nv - ov
        ok = delta >= -tol
        checks.append(
            {
                "type": "max_trend_drop",
                "metric": m,
                "oldest_run": str(oldest.get("run_id")),
                "newest_run": str(newest.get("run_id")),
                "oldest": ov,
                "newest": nv,
                "delta": delta,
                "tolerance": tol,
                "ok": ok,
            }
        )
        if not ok:
            violations.append(
                f"Trend drop too large: {m} oldest={ov} newest={nv} drop={-delta} tol={tol} "
                f"(window={len(window_entries)})"
            )

    return checks, violations


def check_drift(
    *,
    registry_path: Path,
    policy_path: Path,
    baselines_dir: Path,
    suite: str,
    rubric_id: str,
    output_path: Path,
) -> tuple[int, dict[str, Any]]:
    policy = load_policy(policy_path)

    entries = list(_iter_jsonl(registry_path))
    selected = _select_entries(
        entries,
        dataset_id=policy.dataset_id or suite,
        rubric_id=policy.rubric_id or rubric_id,
        judge_engine=policy.judge_engine,
    )

    if not selected:
        report = {
            "schema_version": "1.0",
            "created_at_utc": _utc_now_iso(),
            "policy_id": policy.policy_id,
            "status": "NO_DATA",
            "reason": "No matching runs found in registry for drift check filters.",
            "filters": {
                "dataset_id": policy.dataset_id or suite,
                "rubric_id": policy.rubric_id or rubric_id,
                "judge_engine": policy.judge_engine,
            },
        }
        _write_json(output_path, report)
        return EXIT_RUNTIME_ERROR, report

    latest = selected[-1]
    latest_run_dir = Path(str(latest.get("run_dir", ""))).resolve()
    latest_metrics_path = latest_run_dir / "metrics.json"
    if not latest_metrics_path.exists():
        raise FileNotFoundError(f"Latest run metrics.json missing: {latest_metrics_path}")

    candidate_metrics = _read_json(latest_metrics_path)

    baseline_metrics_path, baseline_metrics = _resolve_baseline_metrics(
        suite=suite,
        rubric_id=rubric_id,
        baselines_dir=baselines_dir,
    )

    point_checks, point_violations = _point_drift_checks(
        policy=policy,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )

    window_n = max(2, int(policy.trend_window))
    window_entries = selected[-window_n:]
    trend_checks, trend_violations = _trend_drift_checks(policy=policy, window_entries=window_entries)

    violations = point_violations + trend_violations
    status = "PASS" if not violations else "FAIL"

    report = {
        "schema_version": "1.0",
        "created_at_utc": _utc_now_iso(),
        "policy": {
            "schema_version": policy.schema_version,
            "policy_id": policy.policy_id,
            "dataset_id": policy.dataset_id or suite,
            "rubric_id": policy.rubric_id or rubric_id,
            "judge_engine": policy.judge_engine,
            "trend_window": policy.trend_window,
        },
        "baseline": {
            "suite": suite,
            "rubric_id": rubric_id,
            "metrics_path": str(baseline_metrics_path),
        },
        "latest_run": {
            "run_id": str(latest.get("run_id")),
            "run_dir": str(latest_run_dir),
            "metrics_path": str(latest_metrics_path),
        },
        "checks": {
            "point": point_checks,
            "trend": trend_checks,
        },
        "violations": violations,
        "status": status,
    }

    _write_json(output_path, report)

    if violations:
        return EXIT_POLICY_VIOLATION, report
    return EXIT_OK, report


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drift detection for LLM-Judge (L3).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="Check drift vs baseline + trend window")
    p_check.add_argument("--policy", required=True, help="Path to drift policy YAML")
    p_check.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="Path to run_registry.jsonl")
    p_check.add_argument("--baselines-dir", default="baselines", help="Baselines root (default: baselines)")
    p_check.add_argument("--suite", required=True, help="Suite/dataset id (e.g., golden)")
    p_check.add_argument("--rubric-id", required=True, help="Rubric id (e.g., chat_quality)")
    p_check.add_argument("--out", default=str(DEFAULT_REPORT_PATH), help="Output report JSON path")
    p_check.set_defaults(func=_cmd_check)

    return p


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        code, report = check_drift(
            registry_path=Path(args.registry),
            policy_path=Path(args.policy),
            baselines_dir=Path(args.baselines_dir),
            suite=str(args.suite),
            rubric_id=str(args.rubric_id),
            output_path=Path(args.out),
        )
        # concise console output
        print(f"Drift status: {report.get('status')}")
        if report.get("violations"):
            print("VIOLATIONS:")
            for v in report["violations"]:
                print(f"- {v}")
        print(f"Report: {args.out}")
        return int(code)
    except SystemExit:
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        return EXIT_RUNTIME_ERROR


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())