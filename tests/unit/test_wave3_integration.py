"""
Wave 3 Integration Tests — smoke test, hit-rate tracking,
streaming progress, event queries, structured alerts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# =====================================================================
# EPIC 2.2: Smoke test + hit-rate tracking
# =====================================================================


class TestSmokeTestSpec:
    """RunSpec smoke test configuration."""

    def test_smoke_test_spec_parsed(self, tmp_path: Path) -> None:
        import yaml

        from llm_judge.eval.spec import RunSpec

        spec_data = {
            "run_id_prefix": "test",
            "dataset": {"dataset_id": "math_basic", "version": "v1"},
            "rubric_id": "math_basic",
            "judge_engine": "deterministic",
            "output_dir": str(tmp_path),
            "smoke_test": {"n": 5, "min_pass_rate": 0.6},
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec_data))

        spec = RunSpec.from_yaml(spec_path)
        assert spec.smoke_test is not None
        assert spec.smoke_test.n == 5
        assert spec.smoke_test.min_pass_rate == 0.6

    def test_smoke_test_spec_optional(self, tmp_path: Path) -> None:
        import yaml

        from llm_judge.eval.spec import RunSpec

        spec_data = {
            "run_id_prefix": "test",
            "dataset": {"dataset_id": "math_basic", "version": "v1"},
            "rubric_id": "math_basic",
            "judge_engine": "deterministic",
            "output_dir": str(tmp_path),
        }
        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(yaml.dump(spec_data))

        spec = RunSpec.from_yaml(spec_path)
        assert spec.smoke_test is None

    def test_smoke_test_defaults(self) -> None:
        from llm_judge.eval.spec import SmokeTestSpec

        st = SmokeTestSpec()
        assert st.n == 10
        assert st.min_pass_rate == 0.5


# =====================================================================
# EPIC 5.2: Event trace queries
# =====================================================================


class TestEventTrace:
    """Trace queries for run lineage."""

    def test_trace_by_run_id(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import append_event, trace_by_run_id

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="eval_run",
            source="test",
            related_ids={"run_id": "run-42", "rubric_id": "math_basic"},
            registry_path=reg,
        )
        append_event(
            event_type="rule_change",
            source="test",
            related_ids={"run_id": "run-42"},
            registry_path=reg,
        )
        append_event(
            event_type="eval_run",
            source="test",
            related_ids={"run_id": "run-99"},
            registry_path=reg,
        )

        trace = trace_by_run_id(run_id="run-42", registry_path=reg)
        assert len(trace) == 2
        assert all(e["related_ids"]["run_id"] == "run-42" for e in trace)

    def test_trace_empty_result(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import append_event, trace_by_run_id

        reg = tmp_path / "events.jsonl"
        append_event(
            event_type="eval_run",
            source="test",
            related_ids={"run_id": "run-1"},
            registry_path=reg,
        )

        trace = trace_by_run_id(run_id="nonexistent", registry_path=reg)
        assert len(trace) == 0


# =====================================================================
# EPIC 8.2: Structured alerts
# =====================================================================


class TestStructuredAlerts:
    """Alert output to reports/alerts/ directory."""

    def test_write_alert(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import write_structured_alert

        alerts_dir = tmp_path / "alerts"
        path = write_structured_alert(
            alert_type="drift_alert",
            severity="high",
            persona="engineer",
            title="F1 dropped below threshold",
            recommended_action="Investigate recent rule changes",
            details={"metric": "f1_fail", "value": 0.72, "threshold": 0.80},
            correlation_id="drift-abc123",
            alerts_dir=alerts_dir,
        )

        assert path.exists()
        alert = json.loads(path.read_text())
        assert alert["alert_type"] == "drift_alert"
        assert alert["severity"] == "high"
        assert alert["persona"] == "engineer"
        assert alert["correlation_id"] == "drift-abc123"

    def test_alert_unique_ids(self, tmp_path: Path) -> None:
        from llm_judge.eval.event_registry import write_structured_alert

        alerts_dir = tmp_path / "alerts"
        p1 = write_structured_alert(
            alert_type="drift_alert",
            severity="high",
            persona="engineer",
            title="Alert 1",
            recommended_action="Fix",
            alerts_dir=alerts_dir,
        )
        p2 = write_structured_alert(
            alert_type="heartbeat",
            severity="medium",
            persona="qa_lead",
            title="Alert 2",
            recommended_action="Check",
            alerts_dir=alerts_dir,
        )

        assert p1 != p2
        assert len(list(alerts_dir.glob("*.json"))) == 2


# =====================================================================
# EPIC 8.1: Documentation completeness
# =====================================================================


class TestWave3Documentation:
    """GETTING_STARTED.md and documentation completeness."""

    def test_getting_started_exists(self) -> None:
        path = Path("docs/GETTING_STARTED.md")
        if not path.exists():
            pytest.skip(
                "docs/GETTING_STARTED.md not found (running outside project root)"
            )
        assert path.stat().st_size > 500, "GETTING_STARTED.md should be substantial"

    def test_all_docs_populated(self) -> None:
        docs_dir = Path("docs")
        if not docs_dir.exists():
            pytest.skip("docs/ not found")

        for name in ("API.md", "DEV_GUIDE.md", "RUNBOOK.md", "GETTING_STARTED.md"):
            path = docs_dir / name
            assert path.exists(), f"Missing: {path}"
            assert path.stat().st_size > 100, f"{name} too small"
