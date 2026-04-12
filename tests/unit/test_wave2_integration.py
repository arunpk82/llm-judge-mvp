"""
Wave 2 Integration Tests — verify cross-capability event registry,
rule aging, drift detection, causation analysis, and response lifecycle.

Wave 2 closes cross-cutting pattern #3: "Detection without response."
These tests verify the full chain: detect → correlate → classify → respond.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# =====================================================================
# Fixtures
# =====================================================================


def _make_event_registry(tmp: Path, events: list[dict[str, Any]] | None = None) -> Path:
    reg = tmp / "event_registry.jsonl"
    if events:
        with reg.open("w", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
    else:
        reg.write_text("")
    return reg


# =====================================================================
# EPIC 5.1: Event Registry
# =====================================================================


class TestEventRegistry:
    """Cross-capability event registry with typed events."""

    def test_append_and_query(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import append_event, query_events

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="eval_run",
            source="test",
            actor="ci",
            related_ids={"run_id": "r1", "rubric_id": "math_basic"},
            payload={"kappa": 0.82},
            registry_path=reg,
        )
        append_event(
            event_type="baseline_promotion",
            source="test",
            actor="op",
            related_ids={"baseline_id": "b1", "rubric_id": "math_basic"},
            registry_path=reg,
        )

        all_events = query_events(registry_path=reg)
        assert len(all_events) == 2

        runs = query_events(event_type="eval_run", registry_path=reg)
        assert len(runs) == 1
        assert runs[0]["payload"]["kappa"] == 0.82

    def test_query_by_related_id(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import append_event, query_events

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="eval_run",
            source="test",
            related_ids={"rubric_id": "math_basic"},
            registry_path=reg,
        )
        append_event(
            event_type="rule_change",
            source="test",
            related_ids={},
            registry_path=reg,
        )

        by_rubric = query_events(
            related_id_key="rubric_id",
            related_id_value="math_basic",
            registry_path=reg,
        )
        assert len(by_rubric) == 1
        assert by_rubric[0]["event_type"] == "eval_run"

    def test_invalid_event_type_rejected(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import append_event

        reg = tmp_path / "events.jsonl"
        with pytest.raises(ValueError, match="Invalid event_type"):
            append_event(event_type="invalid", source="test", registry_path=reg)

    def test_window_query(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import (
            append_event,
            query_events_in_window,
        )

        reg = tmp_path / "events.jsonl"
        e = append_event(
            event_type="eval_run",
            source="test",
            registry_path=reg,
        )
        append_event(
            event_type="rule_change",
            source="test",
            registry_path=reg,
        )

        assert e is not None
        results = query_events_in_window(
            center_timestamp=e.timestamp,
            window_hours=1.0,
            registry_path=reg,
        )
        assert len(results) == 2

    def test_window_query_with_type_filter(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import (
            append_event,
            query_events_in_window,
        )

        reg = tmp_path / "events.jsonl"
        e = append_event(
            event_type="eval_run",
            source="test",
            registry_path=reg,
        )
        append_event(
            event_type="rule_change",
            source="test",
            registry_path=reg,
        )

        assert e is not None
        runs_only = query_events_in_window(
            center_timestamp=e.timestamp,
            window_hours=1.0,
            event_types=["eval_run"],
            registry_path=reg,
        )
        assert len(runs_only) == 1
        assert runs_only[0]["event_type"] == "eval_run"

    def test_rule_snapshot_emission(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import iter_events
        from llm_judge.rules.lifecycle import emit_rule_snapshot

        reg = tmp_path / "events.jsonl"
        emit_rule_snapshot(actor="test", registry_path=reg)

        events = list(iter_events(reg))
        assert len(events) == 1
        assert events[0]["event_type"] == "rule_change"
        assert events[0]["payload"]["action"] == "snapshot"
        assert events[0]["payload"]["rule_count"] >= 4


# =====================================================================
# EPIC 3.2: Rule Aging
# =====================================================================


class TestRuleAging:
    """Rule aging computation and deprecation enforcement."""

    def test_fresh_rule_not_stale(self) -> None:
        from llm_judge.rules.lifecycle import RuleMeta, compute_aging

        rule = RuleMeta(
            name="test.fresh",
            version=1,
            owner="eval-team",
            status="production",
            introduced="2026-03-01",
            last_reviewed="2026-03-27",
            review_period_days=365,
        )
        aging = compute_aging(rule)
        assert not aging.stale
        assert aging.days_since_review is not None
        assert aging.days_since_review < 365

    def test_stale_rule_detected(self) -> None:
        from llm_judge.rules.lifecycle import RuleMeta, compute_aging

        rule = RuleMeta(
            name="test.old",
            version=1,
            owner="eval-team",
            status="production",
            introduced="2024-01-01",
            last_reviewed="2024-06-01",
            review_period_days=180,
        )
        aging = compute_aging(rule)
        assert aging.stale
        assert aging.days_since_review is not None
        assert aging.days_since_review > 500

    def test_deprecation_enforced_after_warning(self) -> None:
        from llm_judge.rules.lifecycle import RuleMeta, compute_aging

        rule = RuleMeta(
            name="test.dep",
            version=1,
            owner="eval-team",
            status="deprecated",
            introduced="2025-01-01",
            deprecated_at="2025-01-15",
            deprecation_warning_days=30,
        )
        aging = compute_aging(rule)
        assert aging.deprecated
        assert aging.deprecation_enforced

    def test_deprecation_within_warning_not_enforced(self) -> None:
        from datetime import timedelta

        from llm_judge.rules.lifecycle import RuleMeta, _today, compute_aging

        rule = RuleMeta(
            name="test.recent_dep",
            version=1,
            owner="eval-team",
            status="deprecated",
            introduced="2025-01-01",
            deprecated_at=str(_today() - timedelta(days=5)),
            deprecation_warning_days=30,
        )
        aging = compute_aging(rule)
        assert aging.deprecated
        assert not aging.deprecation_enforced

    def test_check_aging_all_rules(self) -> None:
        from llm_judge.rules.lifecycle import check_aging

        reports = check_aging()
        assert len(reports) >= 4
        stale = [r for r in reports if r.stale]
        assert len(stale) == 0, "All current rules should be recently reviewed"

    def test_deprecated_rules_excluded_from_plan(self) -> None:
        from llm_judge.rules.lifecycle import get_deprecated_enforced_rules

        # Current manifest has no deprecated rules
        enforced = get_deprecated_enforced_rules()
        assert len(enforced) == 0

    def test_audit_trail(self, tmp_path: Path) -> None:
        from llm_judge.rules.lifecycle import append_audit_entry, read_audit_log

        audit = tmp_path / "audit.jsonl"
        append_audit_entry(
            rule_id="test.rule",
            action="created",
            actor="ci",
            audit_path=audit,
        )
        append_audit_entry(
            rule_id="test.rule",
            action="reviewed",
            actor="eng",
            audit_path=audit,
        )
        append_audit_entry(
            rule_id="other.rule",
            action="deprecated",
            actor="ci",
            audit_path=audit,
        )

        all_entries = read_audit_log(audit_path=audit)
        assert len(all_entries) == 3

        filtered = read_audit_log(rule_id="test.rule", audit_path=audit)
        assert len(filtered) == 2


# =====================================================================
# EPIC 6.1: Multi-dimensional Drift
# =====================================================================


class TestDriftDetection:
    """Heartbeat check and cross-dimensional correlation."""

    def test_heartbeat_ok(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import _heartbeat_check
        from llm_judge.eval.event_registry import append_event

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="eval_run",
            source="test",
            registry_path=reg,
        )
        result = _heartbeat_check(
            heartbeat_max_hours=72,
            event_registry_path=reg,
        )
        assert result["ok"]

    def test_heartbeat_no_events(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import _heartbeat_check

        reg = tmp_path / "events.jsonl"
        reg.write_text("")
        result = _heartbeat_check(
            heartbeat_max_hours=72,
            event_registry_path=reg,
        )
        assert not result["ok"]

    def test_correlation_finds_governance_events(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import _correlate_with_events, _utc_now_iso
        from llm_judge.eval.event_registry import append_event

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="rule_change",
            source="test",
            registry_path=reg,
        )
        append_event(
            event_type="baseline_promotion",
            source="test",
            registry_path=reg,
        )
        # eval_run should NOT be returned by correlation
        append_event(
            event_type="eval_run",
            source="test",
            registry_path=reg,
        )

        correlated = _correlate_with_events(
            violation_timestamp=_utc_now_iso(),
            window_hours=1.0,
            event_registry_path=reg,
        )
        types = {e["event_type"] for e in correlated}
        assert "rule_change" in types
        assert "baseline_promotion" in types
        assert "eval_run" not in types

    def test_drift_alert_emitted(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import _emit_drift_alert
        from llm_judge.eval.event_registry import query_events

        reg = tmp_path / "events.jsonl"
        _emit_drift_alert(
            report={
                "policy": {"policy_id": "test"},
                "latest_run": {"run_id": "r1"},
                "status": "FAIL",
            },
            violations=["Metric drop: f1_fail"],
            correlated_events=[],
            event_registry_path=reg,
        )
        alerts = query_events(event_type="drift_alert", registry_path=reg)
        assert len(alerts) == 1
        assert alerts[0]["payload"]["violation_count"] == 1

    def test_policy_loads_with_new_fields(self) -> None:
        from llm_judge.eval.drift import load_policy

        policy = load_policy(Path("configs/policies/drift.yaml"))
        assert policy.heartbeat_max_hours == 72
        assert policy.correlation_window_hours == 24
        assert policy.response_actions is not None
        assert policy.response_actions["f1_fail"] == "block"


# =====================================================================
# EPIC 6.2: Causation & Response
# =====================================================================


class TestCausationAndResponse:
    """Causation analysis, response classification, and lifecycle."""

    def test_causation_report(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import _utc_now_iso, build_causation_report
        from llm_judge.eval.event_registry import append_event

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="rule_change",
            source="test",
            registry_path=reg,
        )

        report = build_causation_report(
            drift_report={"created_at_utc": _utc_now_iso(), "violations": ["drop"]},
            event_registry_path=reg,
            window_hours=1.0,
        )
        assert report["correlated_events"] == 1
        assert report["causes"][0]["causal_likelihood"] == "high"

    def test_response_classification_with_policy(self) -> None:
        from llm_judge.eval.drift import classify_response_actions

        violations = [
            "Metric drop: f1_fail baseline=0.95",
            "Metric drop: f1_pass baseline=0.90",
            "Trend drop: recall_fail oldest=0.80",
        ]
        result = classify_response_actions(
            violations=violations,
            response_actions={
                "f1_fail": "block",
                "f1_pass": "warn",
                "recall_fail": "log",
            },
        )
        assert len(result["block"]) == 1
        assert len(result["warn"]) == 1
        assert len(result["log"]) == 1

    def test_response_classification_default_warn(self) -> None:
        from llm_judge.eval.drift import classify_response_actions

        result = classify_response_actions(
            violations=["some violation"],
            response_actions=None,
        )
        assert len(result["warn"]) == 1

    def test_drift_issue_lifecycle(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import (
            _utc_now_iso,
            create_drift_issue,
            transition_drift_issue,
        )
        from llm_judge.eval.event_registry import query_events

        issues = tmp_path / "issues.jsonl"
        events = tmp_path / "events.jsonl"

        # Create issue
        issue = create_drift_issue(
            drift_report={"created_at_utc": _utc_now_iso(), "violations": ["drop"]},
            issues_path=issues,
            event_registry_path=events,
        )
        assert issue["state"] == "detected"

        # Transition through lifecycle
        t1 = transition_drift_issue(
            issue_id=issue["issue_id"],
            to_state="triaged",
            actor="eng",
            note="Investigating",
            issues_path=issues,
            event_registry_path=events,
        )
        assert t1["state"] == "triaged"

        t2 = transition_drift_issue(
            issue_id=issue["issue_id"],
            to_state="responding",
            actor="eng",
            note="Rolling back",
            issues_path=issues,
            event_registry_path=events,
        )
        assert t2["state"] == "responding"

        t3 = transition_drift_issue(
            issue_id=issue["issue_id"],
            to_state="resolved",
            actor="eng",
            note="Metrics recovered",
            issues_path=issues,
            event_registry_path=events,
        )
        assert t3["state"] == "resolved"
        assert len(t3["history"]) == 4

        # Verify all lifecycle events emitted
        responses = query_events(
            event_type="drift_response",
            registry_path=events,
        )
        assert len(responses) == 4  # created + 3 transitions

    def test_invalid_transition_rejected(self, tmp_path: Path) -> None:
        from llm_judge.eval.drift import (
            _utc_now_iso,
            create_drift_issue,
            transition_drift_issue,
        )

        issues = tmp_path / "issues.jsonl"
        events = tmp_path / "events.jsonl"

        issue = create_drift_issue(
            drift_report={"created_at_utc": _utc_now_iso(), "violations": ["x"]},
            issues_path=issues,
            event_registry_path=events,
        )

        # detected → resolved (allowed)
        transition_drift_issue(
            issue_id=issue["issue_id"],
            to_state="resolved",
            issues_path=issues,
            event_registry_path=events,
        )

        # resolved → detected (not allowed — terminal)
        with pytest.raises(ValueError, match="Invalid transition"):
            transition_drift_issue(
                issue_id=issue["issue_id"],
                to_state="detected",
                issues_path=issues,
                event_registry_path=events,
            )
