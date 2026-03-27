"""
L4 Integration Tests — Hybrid Adjudication Engine.

Tests judge registry, calibration pipeline, trust gate, bias detection,
confidence routing, and human adjudication queue.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest
import yaml

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse

# =====================================================================
# Mock judge for testing (no API calls)
# =====================================================================

class MockLLMJudge(JudgeEngine):
    """Deterministic mock that returns configurable responses."""

    def __init__(self, decision: Literal["pass", "fail"] = "pass", confidence: float = 0.8) -> None:
        self._decision = decision
        self._confidence = confidence
        self._call_count = 0

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        self._call_count += 1
        return PredictResponse(
            decision=self._decision,
            overall_score=4.0 if self._decision == "pass" else 2.0,
            scores={"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4},
            confidence=self._confidence,
            flags=[],
            explanations={"relevance": "Test explanation"},
        )


class FlippingJudge(JudgeEngine):
    """Returns different decisions on alternate calls — simulates position bias."""

    def __init__(self) -> None:
        self._count = 0

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        self._count += 1
        decision: Literal["pass", "fail"] = "pass" if self._count % 2 == 1 else "fail"
        return PredictResponse(
            decision=decision,
            overall_score=4.0 if decision == "pass" else 2.0,
            scores={"relevance": 4, "clarity": 3, "correctness": 4, "tone": 3},
            confidence=0.6,
            flags=[],
        )


def _make_golden_case(case_id: str, decision: str = "pass") -> dict[str, Any]:
    return {
        "conversation": [{"role": "user", "content": "What is 2+2?"}],
        "candidate_answer": "4",
        "case_id": case_id,
        "rubric_id": "chat_quality",
        "human_decision": decision,
        "human_scores": {"relevance": 5, "clarity": 4, "correctness": 5, "tone": 4},
    }


# =====================================================================
# EPIC 7.1: Judge Registry
# =====================================================================

class TestJudgeRegistry:
    """Judge registration and configuration."""

    def test_load_registry(self) -> None:
        from llm_judge.calibration import load_judge_registry

        reg_path = Path("configs/judges/registry.yaml")
        if not reg_path.exists():
            pytest.skip("Judge registry not found")

        judges, trust_gate = load_judge_registry(reg_path)
        assert len(judges) >= 2
        assert "gpt-4o-mini" in judges
        assert judges["gpt-4o-mini"].provider == "openai"
        assert trust_gate.enforce is True

    def test_registry_missing_raises(self) -> None:
        from llm_judge.calibration import load_judge_registry

        with pytest.raises(FileNotFoundError):
            load_judge_registry(Path("/nonexistent/registry.yaml"))

    def test_judge_meta_fields(self) -> None:
        from llm_judge.calibration import load_judge_registry

        reg_path = Path("configs/judges/registry.yaml")
        if not reg_path.exists():
            pytest.skip("Judge registry not found")

        judges, _ = load_judge_registry(reg_path)
        meta = judges["gpt-4o-mini"]
        assert meta.model == "gpt-4o-mini"
        assert meta.calibration_config.min_accuracy == 0.70
        assert meta.calibration_config.golden_dataset_id == "golden"


# =====================================================================
# EPIC 7.1: Calibration Pipeline
# =====================================================================

class TestCalibrationPipeline:
    """Calibration against golden dataset."""

    def test_calibration_passes(self) -> None:
        from llm_judge.calibration import (
            JudgeCalibrationConfig,
            JudgeMeta,
            run_calibration,
        )

        judge = MockLLMJudge(decision="pass", confidence=0.9)
        meta = JudgeMeta(
            judge_id="test-judge", provider="mock", model="mock-v1",
            prompt_version="v1", domain="general", status="registered",
            calibration_config=JudgeCalibrationConfig(
                min_accuracy=0.50, min_dimension_accuracy=0.40,
            ),
        )
        cases = [_make_golden_case(f"c{i}", "pass") for i in range(10)]

        result = run_calibration(judge=judge, judge_meta=meta, golden_cases=cases)
        assert result.cases_evaluated == 10
        assert result.overall_accuracy == 1.0  # mock always returns "pass"
        assert result.passed

    def test_calibration_fails_low_accuracy(self) -> None:
        from llm_judge.calibration import (
            JudgeCalibrationConfig,
            JudgeMeta,
            run_calibration,
        )

        judge = MockLLMJudge(decision="fail", confidence=0.9)  # always "fail"
        meta = JudgeMeta(
            judge_id="bad-judge", provider="mock", model="mock-v1",
            prompt_version="v1", domain="general", status="registered",
            calibration_config=JudgeCalibrationConfig(min_accuracy=0.80),
        )
        # All golden cases expect "pass" — judge says "fail"
        cases = [_make_golden_case(f"c{i}", "pass") for i in range(10)]

        result = run_calibration(judge=judge, judge_meta=meta, golden_cases=cases)
        assert result.overall_accuracy == 0.0
        assert not result.passed
        assert len(result.failure_reasons) > 0

    def test_calibration_result_serialization(self, tmp_path: Path) -> None:
        from llm_judge.calibration import (
            JudgeCalibrationConfig,
            JudgeMeta,
            run_calibration,
            save_calibration_result,
        )

        judge = MockLLMJudge()
        meta = JudgeMeta(
            judge_id="test-ser", provider="mock", model="mock",
            prompt_version="v1", domain="general", status="registered",
            calibration_config=JudgeCalibrationConfig(min_accuracy=0.50),
        )
        cases = [_make_golden_case("c1", "pass")]
        result = run_calibration(judge=judge, judge_meta=meta, golden_cases=cases)

        path = save_calibration_result(result, calibration_dir=tmp_path)
        assert path.exists()

        data = json.loads(path.read_text())
        assert data["judge_id"] == "test-ser"
        assert "overall_accuracy" in data


# =====================================================================
# EPIC 7.1: Trust Gate
# =====================================================================

class TestTrustGate:
    """Block uncalibrated judges from production."""

    def test_deterministic_always_allowed(self) -> None:
        from llm_judge.calibration import check_trust_gate

        allowed, reason = check_trust_gate(
            judge_id="anything", engine_choice="deterministic",
        )
        assert allowed

    def test_unregistered_judge_blocked(self, tmp_path: Path) -> None:
        from llm_judge.calibration import check_trust_gate

        # Create minimal registry
        reg = tmp_path / "registry.yaml"
        reg.write_text(yaml.dump({
            "schema_version": "1",
            "judges": {},
            "trust_gate": {"enforce": True},
        }))

        allowed, reason = check_trust_gate(
            judge_id="unknown-judge", engine_choice="llm",
            registry_path=reg,
        )
        assert not allowed
        assert "not registered" in reason

    def test_calibrated_judge_allowed(self, tmp_path: Path) -> None:
        from llm_judge.calibration import (
            CalibrationResult,
            check_trust_gate,
            save_calibration_result,
        )

        # Create registry with calibrated judge
        reg = tmp_path / "registry.yaml"
        reg.write_text(yaml.dump({
            "schema_version": "1",
            "judges": {
                "test-judge": {
                    "provider": "mock", "model": "mock",
                    "prompt_version": "v1", "domain": "general",
                    "status": "calibrated",
                    "calibration": {"min_accuracy": 0.50},
                },
            },
            "trust_gate": {"enforce": True},
        }))

        # Save passing calibration result
        cal_dir = tmp_path / "calibration"
        result = CalibrationResult(
            judge_id="test-judge",
            golden_dataset_id="golden",
            golden_dataset_version="v1",
            timestamp="2026-03-27T12:00:00Z",
            cases_evaluated=10,
            decision_matches=9,
            passed=True,
        )
        save_calibration_result(result, calibration_dir=cal_dir)

        allowed, reason = check_trust_gate(
            judge_id="test-judge", engine_choice="llm",
            registry_path=reg, calibration_dir=cal_dir,
        )
        assert allowed

    def test_calibrated_judge_wrapper(self) -> None:
        from llm_judge.calibration import CalibratedJudge

        judge = MockLLMJudge(confidence=0.4)

        # Should raise because no registry exists for trust gate
        # In real usage, the registry would have a calibrated judge
        with pytest.raises(RuntimeError, match="Trust gate blocked"):
            CalibratedJudge(
                judge, "nonexistent-judge",
                registry_path=Path("/nonexistent/registry.yaml"),
            )


# =====================================================================
# EPIC 7.2: Bias Detection
# =====================================================================

class TestBiasDetection:
    """Position and length bias detection."""

    def test_no_position_bias(self) -> None:
        from llm_judge.calibration.bias import check_position_bias

        judge = MockLLMJudge()  # always returns same result
        cases = [_make_golden_case(f"c{i}") for i in range(5)]

        result = check_position_bias(judge=judge, cases=cases, max_cases=5)
        assert result["status"] == "PASS"
        assert result["inconsistency_rate"] == 0.0

    def test_position_bias_detected(self) -> None:
        from llm_judge.calibration.bias import check_position_bias

        judge = FlippingJudge()  # alternates pass/fail
        cases = [_make_golden_case(f"c{i}") for i in range(5)]

        result = check_position_bias(judge=judge, cases=cases, max_cases=5)
        assert result["inconsistency_rate"] > 0
        assert result["status"] == "FAIL"

    def test_length_bias_no_correlation(self) -> None:
        from llm_judge.calibration.bias import check_length_bias

        cases = [
            {"case_id": f"c{i}", "candidate_answer": "x" * (10 + i)}
            for i in range(10)
        ]
        # Scores don't correlate with length
        judgments = [
            {"case_id": f"c{i}", "judge_scores": {"relevance": 3, "clarity": 3}}
            for i in range(10)
        ]

        result = check_length_bias(judgments=judgments, cases=cases)
        assert result["status"] == "PASS"
        assert abs(result["correlation"]) < 0.5

    def test_debiased_judge_averages(self) -> None:
        from llm_judge.calibration.bias import DebiasedJudge

        judge = MockLLMJudge(decision="pass", confidence=0.8)
        debiased = DebiasedJudge(judge, n_evaluations=3)

        from llm_judge.schemas import Message
        req = PredictRequest(
            conversation=[Message(role="user", content="Hi")],
            candidate_answer="Hello",
            rubric_id="chat_quality",
        )
        resp = debiased.evaluate(req)
        assert resp.decision == "pass"
        assert resp.confidence == 0.8


# =====================================================================
# EPIC 7.3: Human Adjudication
# =====================================================================

class TestHumanAdjudication:
    """Confidence routing and adjudication queue."""

    def test_route_low_confidence(self) -> None:
        from llm_judge.calibration.adjudication import should_route_to_human

        route, reason = should_route_to_human(confidence=0.3, threshold=0.6)
        assert route
        assert "Low confidence" in reason

    def test_dont_route_high_confidence(self) -> None:
        from llm_judge.calibration.adjudication import should_route_to_human

        route, reason = should_route_to_human(confidence=0.9, threshold=0.6)
        assert not route

    def test_route_on_disagreement_flag(self) -> None:
        from llm_judge.calibration.adjudication import should_route_to_human

        route, reason = should_route_to_human(
            confidence=0.8, threshold=0.6,
            flags=["position_bias_disagreement:2/3"],
        )
        assert route

    def test_enqueue_and_resolve(self, tmp_path: Path) -> None:
        from llm_judge.calibration.adjudication import (
            enqueue_case,
            load_queue,
            resolve_case,
        )

        queue = tmp_path / "queue.jsonl"

        # Enqueue
        case = enqueue_case(
            case_id="c1", run_id="run-1", rubric_id="chat_quality",
            llm_decision="pass", llm_confidence=0.4,
            llm_scores={"relevance": 4, "clarity": 3},
            conversation=[{"role": "user", "content": "Hi"}],
            candidate_answer="Hello",
            queue_path=queue,
        )
        assert case.state == "pending"

        # Check queue
        pending = load_queue(state="pending", queue_path=queue)
        assert len(pending) == 1

        # Resolve
        resolved = resolve_case(
            case_id="c1",
            human_decision="fail",
            human_scores={"relevance": 2, "clarity": 2},
            human_notes="Answer is too brief",
            adjudicator="qa_lead",
            queue_path=queue,
        )
        assert resolved["state"] == "resolved"
        assert resolved["human_decision"] == "fail"

        # Check queue again
        pending_after = load_queue(state="pending", queue_path=queue)
        assert len(pending_after) == 0

    def test_resolve_already_resolved_raises(self, tmp_path: Path) -> None:
        from llm_judge.calibration.adjudication import enqueue_case, resolve_case

        queue = tmp_path / "queue.jsonl"
        enqueue_case(
            case_id="c2", run_id="r", rubric_id="r",
            llm_decision="pass", llm_confidence=0.3,
            llm_scores={}, conversation=[], candidate_answer="x",
            queue_path=queue,
        )
        resolve_case(
            case_id="c2", human_decision="pass", queue_path=queue,
        )

        with pytest.raises(ValueError, match="already resolved"):
            resolve_case(
                case_id="c2", human_decision="fail", queue_path=queue,
            )

    def test_queue_stats(self, tmp_path: Path) -> None:
        from llm_judge.calibration.adjudication import (
            enqueue_case,
            get_queue_stats,
            resolve_case,
        )

        queue = tmp_path / "queue.jsonl"
        enqueue_case(
            case_id="s1", run_id="r", rubric_id="r",
            llm_decision="pass", llm_confidence=0.3,
            llm_scores={}, conversation=[], candidate_answer="x",
            queue_path=queue,
        )
        enqueue_case(
            case_id="s2", run_id="r", rubric_id="r",
            llm_decision="fail", llm_confidence=0.4,
            llm_scores={}, conversation=[], candidate_answer="y",
            queue_path=queue,
        )
        resolve_case(
            case_id="s1", human_decision="pass", queue_path=queue,
        )

        stats = get_queue_stats(queue_path=queue)
        assert stats["total_cases"] == 2
        assert stats["resolved"] == 1
        assert stats["llm_human_agreement_rate"] == 1.0  # both said "pass"
