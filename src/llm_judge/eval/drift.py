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

    # EPIC-6.1: heartbeat — max hours between eval_run events
    heartbeat_max_hours: float | None = None

    # EPIC-6.1: cross-dimensional correlation window (hours)
    correlation_window_hours: float = 24.0

    # EPIC-6.2: per-metric response actions
    # Maps metric name → action type: "block", "warn", "log"
    response_actions: dict[str, str] | None = None


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

    def _dict_str(key: str) -> dict[str, str] | None:
        v = raw.get(key)
        if v is None or not isinstance(v, dict):
            return None
        return {str(k): str(val) for k, val in v.items()}

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
        heartbeat_max_hours=float(raw["heartbeat_max_hours"]) if raw.get("heartbeat_max_hours") else None,
        correlation_window_hours=float(raw.get("correlation_window_hours", 24.0)),
        response_actions=_dict_str("response_actions"),
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


# =====================================================================
# EPIC-6.1: Heartbeat check
# =====================================================================

def _heartbeat_check(
    *,
    heartbeat_max_hours: float,
    event_registry_path: Path | None = None,
) -> dict[str, Any]:
    """
    Check if eval_run events are arriving within the expected interval.

    Returns a dict with: ok, last_event_timestamp, hours_since_last, threshold.
    If no events exist, flags as a heartbeat failure.
    """
    try:
        from llm_judge.eval.event_registry import (
            DEFAULT_EVENT_REGISTRY_PATH,
            query_events,
        )

        reg_path = event_registry_path or DEFAULT_EVENT_REGISTRY_PATH
        eval_events = query_events(event_type="eval_run", registry_path=reg_path)
    except Exception:
        return {
            "ok": True,
            "skipped": True,
            "reason": "Event registry unavailable — heartbeat check skipped",
        }

    if not eval_events:
        return {
            "ok": False,
            "last_event_timestamp": None,
            "hours_since_last": None,
            "threshold_hours": heartbeat_max_hours,
            "reason": "No eval_run events found in event registry",
        }

    # Events are in append order — last one is most recent
    last_ts = eval_events[-1].get("timestamp", "")
    try:
        last_dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        hours_since = (now - last_dt).total_seconds() / 3600.0
        ok = hours_since <= heartbeat_max_hours
    except (ValueError, TypeError):
        return {
            "ok": False,
            "last_event_timestamp": last_ts,
            "reason": f"Could not parse timestamp: {last_ts}",
        }

    return {
        "ok": ok,
        "last_event_timestamp": last_ts,
        "hours_since_last": round(hours_since, 2),
        "threshold_hours": heartbeat_max_hours,
    }


# =====================================================================
# EPIC-6.1: Cross-dimensional correlation
# =====================================================================

def _correlate_with_events(
    *,
    violation_timestamp: str,
    window_hours: float = 24.0,
    event_registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Find governance events (rule_change, dataset_registration, baseline_promotion)
    within a time window around a drift violation.

    Returns events sorted by temporal proximity to the violation.
    """
    try:
        from llm_judge.eval.event_registry import (
            DEFAULT_EVENT_REGISTRY_PATH,
            query_events_in_window,
        )

        reg_path = event_registry_path or DEFAULT_EVENT_REGISTRY_PATH
        nearby = query_events_in_window(
            center_timestamp=violation_timestamp,
            window_hours=window_hours,
            event_types=["rule_change", "dataset_registration", "baseline_promotion"],
            registry_path=reg_path,
        )
    except Exception:
        return []

    # Sort by proximity to violation timestamp
    try:
        center = datetime.fromisoformat(violation_timestamp.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return nearby

    def _proximity(event: dict[str, Any]) -> float:
        try:
            ts = datetime.fromisoformat(event.get("timestamp", "").replace("Z", "+00:00"))
            return abs((ts - center).total_seconds())
        except Exception:
            return float("inf")

    nearby.sort(key=_proximity)
    return nearby


def _emit_drift_alert(
    *,
    report: dict[str, Any],
    violations: list[str],
    correlated_events: list[dict[str, Any]],
    event_registry_path: Path | None = None,
) -> None:
    """Emit a drift_alert event to the cross-capability registry."""
    try:
        from llm_judge.eval.event_registry import (
            DEFAULT_EVENT_REGISTRY_PATH,
            append_event,
        )

        reg_path = event_registry_path or DEFAULT_EVENT_REGISTRY_PATH
        append_event(
            event_type="drift_alert",
            source="eval/drift.py",
            actor="system",
            related_ids={
                "policy_id": report.get("policy", {}).get("policy_id", ""),
                "latest_run_id": report.get("latest_run", {}).get("run_id", ""),
            },
            payload={
                "status": report.get("status"),
                "violation_count": len(violations),
                "violations": violations[:5],
                "correlated_event_count": len(correlated_events),
                "correlated_event_types": list({
                    e.get("event_type") for e in correlated_events
                }),
            },
            registry_path=reg_path,
        )
    except Exception:
        pass  # best-effort


# =====================================================================
# EPIC-6.2: Causation analysis
# =====================================================================

def build_causation_report(
    *,
    drift_report: dict[str, Any],
    event_registry_path: Path | None = None,
    window_hours: float = 24.0,
) -> dict[str, Any]:
    """
    Build a causation report correlating a drift alert with governance events.

    Finds events within ±window_hours of the drift report timestamp,
    ranks them by temporal proximity, and classifies likely causes.
    """
    report_ts = drift_report.get("created_at_utc", _utc_now_iso())
    violations = drift_report.get("violations", [])

    correlated = _correlate_with_events(
        violation_timestamp=report_ts,
        window_hours=window_hours,
        event_registry_path=event_registry_path,
    )

    # Classify each correlated event by likely causal relationship
    causes: list[dict[str, Any]] = []
    for event in correlated:
        event_ts = event.get("timestamp", "")
        try:
            evt_dt = datetime.fromisoformat(event_ts.replace("Z", "+00:00"))
            rpt_dt = datetime.fromisoformat(report_ts.replace("Z", "+00:00"))
            hours_delta = (evt_dt - rpt_dt).total_seconds() / 3600.0
        except (ValueError, TypeError):
            hours_delta = None

        causes.append({
            "event_type": event.get("event_type"),
            "timestamp": event_ts,
            "source": event.get("source"),
            "related_ids": event.get("related_ids", {}),
            "hours_from_drift": round(hours_delta, 2) if hours_delta is not None else None,
            "direction": "before" if hours_delta and hours_delta < 0 else "after",
            "causal_likelihood": (
                "high" if hours_delta is not None and abs(hours_delta) < 2
                else "medium" if hours_delta is not None and abs(hours_delta) < 12
                else "low"
            ),
        })

    return {
        "schema_version": "1.0",
        "created_at_utc": _utc_now_iso(),
        "drift_report_timestamp": report_ts,
        "violation_count": len(violations),
        "violations": violations,
        "correlated_events": len(causes),
        "causes": causes,
        "window_hours": window_hours,
    }


# =====================================================================
# EPIC-6.2: Response action classification
# =====================================================================

VALID_RESPONSE_ACTIONS = frozenset({"block", "warn", "log"})


def classify_response_actions(
    *,
    violations: list[str],
    response_actions: dict[str, str] | None,
) -> dict[str, list[str]]:
    """
    Classify violations by response action type.

    Returns: {"block": [...], "warn": [...], "log": [...]}
    Each violation is placed in the bucket of its metric's configured action.
    Unmatched violations default to "warn".
    """
    result: dict[str, list[str]] = {"block": [], "warn": [], "log": []}

    if not response_actions:
        # No policy — all violations default to "warn"
        result["warn"] = list(violations)
        return result

    for v in violations:
        action = "warn"  # default
        # Match violation to metric name in response_actions
        for metric, act in response_actions.items():
            if metric in v:
                action = act if act in VALID_RESPONSE_ACTIONS else "warn"
                break
        result[action].append(v)

    return result


# =====================================================================
# EPIC-6.2: Drift issue lifecycle state machine
# =====================================================================

DRIFT_ISSUE_STATES = ("detected", "triaged", "responding", "resolved")
VALID_TRANSITIONS: dict[str, list[str]] = {
    "detected": ["triaged", "resolved"],
    "triaged": ["responding", "resolved"],
    "responding": ["resolved", "triaged"],  # can go back to triaged if response fails
    "resolved": [],  # terminal
}

DEFAULT_DRIFT_ISSUES_PATH = Path("reports/drift/drift_issues.jsonl")


def _load_drift_issues(
    path: Path = DEFAULT_DRIFT_ISSUES_PATH,
) -> list[dict[str, Any]]:
    """Load all drift issue records."""
    if not path.exists():
        return []
    issues: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                issues.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return issues


def create_drift_issue(
    *,
    drift_report: dict[str, Any],
    causation_report: dict[str, Any] | None = None,
    response_classification: dict[str, list[str]] | None = None,
    issues_path: Path = DEFAULT_DRIFT_ISSUES_PATH,
    event_registry_path: Path | None = None,
) -> dict[str, Any]:
    """
    Create a new drift issue in Detected state.

    This is the entry point for the lifecycle state machine.
    """
    import hashlib

    report_ts = drift_report.get("created_at_utc", _utc_now_iso())
    # Generate a stable issue ID from the report timestamp
    issue_id = "drift-" + hashlib.sha256(report_ts.encode()).hexdigest()[:12]

    issue = {
        "issue_id": issue_id,
        "state": "detected",
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
        "drift_report_timestamp": report_ts,
        "violations": drift_report.get("violations", []),
        "response_classification": response_classification,
        "causation_summary": {
            "correlated_events": causation_report.get("correlated_events", 0)
            if causation_report else 0,
            "top_cause": causation_report["causes"][0] if causation_report
            and causation_report.get("causes") else None,
        },
        "history": [
            {
                "timestamp": _utc_now_iso(),
                "from_state": None,
                "to_state": "detected",
                "actor": "system",
                "note": "Drift issue created automatically from drift check",
            }
        ],
    }

    issues_path.parent.mkdir(parents=True, exist_ok=True)
    with issues_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(issue, sort_keys=True) + "\n")

    # Emit lifecycle event to cross-capability registry
    try:
        from llm_judge.eval.event_registry import (
            DEFAULT_EVENT_REGISTRY_PATH,
            append_event,
        )

        reg_path = event_registry_path or DEFAULT_EVENT_REGISTRY_PATH
        append_event(
            event_type="drift_response",
            source="eval/drift.py",
            actor="system",
            related_ids={"issue_id": issue_id},
            payload={
                "action": "created",
                "state": "detected",
                "violation_count": len(issue["violations"]),
            },
            registry_path=reg_path,
        )
    except Exception:
        pass

    return issue


def transition_drift_issue(
    *,
    issue_id: str,
    to_state: str,
    actor: str = "operator",
    note: str = "",
    issues_path: Path = DEFAULT_DRIFT_ISSUES_PATH,
    event_registry_path: Path | None = None,
) -> dict[str, Any]:
    """
    Transition a drift issue to a new state.

    Validates the transition is allowed by the state machine.
    Appends the updated issue to the issues log (append-only).
    """
    if to_state not in DRIFT_ISSUE_STATES:
        raise ValueError(f"Invalid state: '{to_state}'. Valid: {DRIFT_ISSUE_STATES}")

    issues = _load_drift_issues(issues_path)
    # Find the latest record for this issue_id
    matching = [i for i in issues if i.get("issue_id") == issue_id]
    if not matching:
        raise ValueError(f"Drift issue not found: {issue_id}")

    current = matching[-1]  # latest state
    current_state = current.get("state", "detected")

    allowed = VALID_TRANSITIONS.get(current_state, [])
    if to_state not in allowed:
        raise ValueError(
            f"Invalid transition: {current_state} → {to_state}. "
            f"Allowed from '{current_state}': {allowed}"
        )

    updated = {
        **current,
        "state": to_state,
        "updated_at": _utc_now_iso(),
    }
    updated["history"] = list(current.get("history", [])) + [
        {
            "timestamp": _utc_now_iso(),
            "from_state": current_state,
            "to_state": to_state,
            "actor": actor,
            "note": note,
        }
    ]

    with issues_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(updated, sort_keys=True) + "\n")

    # Emit lifecycle event
    try:
        from llm_judge.eval.event_registry import (
            DEFAULT_EVENT_REGISTRY_PATH,
            append_event,
        )

        reg_path = event_registry_path or DEFAULT_EVENT_REGISTRY_PATH
        append_event(
            event_type="drift_response",
            source="eval/drift.py",
            actor=actor,
            related_ids={"issue_id": issue_id},
            payload={
                "action": "transition",
                "from_state": current_state,
                "to_state": to_state,
                "note": note,
            },
            registry_path=reg_path,
        )
    except Exception:
        pass

    return updated


def check_drift(
    *,
    registry_path: Path,
    policy_path: Path,
    baselines_dir: Path,
    suite: str,
    rubric_id: str,
    output_path: Path,
    event_registry_path: Path | None = None,
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
        report: dict[str, Any] = {
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

    # --- EPIC-6.1: Heartbeat check ---
    heartbeat_result: dict[str, Any] | None = None
    if policy.heartbeat_max_hours is not None:
        heartbeat_result = _heartbeat_check(
            heartbeat_max_hours=policy.heartbeat_max_hours,
            event_registry_path=event_registry_path,
        )
        if heartbeat_result and not heartbeat_result.get("ok", True):
            reason = heartbeat_result.get("reason", "Heartbeat threshold exceeded")
            violations.append(f"Heartbeat: {reason}")
            status = "FAIL"

    # --- EPIC-6.1: Cross-dimensional correlation ---
    correlated_events: list[dict[str, Any]] = []
    if violations:
        report_ts = _utc_now_iso()
        correlated_events = _correlate_with_events(
            violation_timestamp=report_ts,
            window_hours=policy.correlation_window_hours,
            event_registry_path=event_registry_path,
        )

    report = {
        "schema_version": "1.1",
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

    # EPIC-6.1 additions (added after dict creation for mypy compatibility)
    report["heartbeat"] = heartbeat_result
    report["correlated_events"] = [
        {
            "event_type": e.get("event_type"),
            "timestamp": e.get("timestamp"),
            "source": e.get("source"),
            "related_ids": e.get("related_ids", {}),
        }
        for e in correlated_events[:10]
    ]

    _write_json(output_path, report)

    # --- EPIC-6.1: Emit drift_alert event if violations detected ---
    if violations:
        _emit_drift_alert(
            report=report,
            violations=violations,
            correlated_events=correlated_events,
            event_registry_path=event_registry_path,
        )

        # --- EPIC-6.2: Response classification ---
        response_class = classify_response_actions(
            violations=violations,
            response_actions=policy.response_actions,
        )
        report["response_classification"] = response_class

        # --- EPIC-6.2: Causation report ---
        causation = build_causation_report(
            drift_report=report,
            event_registry_path=event_registry_path,
            window_hours=policy.correlation_window_hours,
        )
        report["causation"] = causation

        # --- EPIC-6.2: Create drift issue ---
        try:
            issue = create_drift_issue(
                drift_report=report,
                causation_report=causation,
                response_classification=response_class,
                event_registry_path=event_registry_path,
            )
            report["drift_issue_id"] = issue["issue_id"]
        except Exception:
            pass  # best-effort

        # Re-write report with response + causation + issue
        _write_json(output_path, report)

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
    p_check.add_argument("--event-registry", default=None, help="Path to event_registry.jsonl (optional)")
    p_check.set_defaults(func=_cmd_check)

    return p


def _cmd_check(args: argparse.Namespace) -> int:
    try:
        event_reg = Path(args.event_registry) if args.event_registry else None
        code, report = check_drift(
            registry_path=Path(args.registry),
            policy_path=Path(args.policy),
            baselines_dir=Path(args.baselines_dir),
            suite=str(args.suite),
            rubric_id=str(args.rubric_id),
            output_path=Path(args.out),
            event_registry_path=event_reg,
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