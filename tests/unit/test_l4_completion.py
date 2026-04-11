"""
L4 Completion Tests — Steps 3, 6, 9, 10.

Tests prompt versioning, hallucination detection, test case generation,
and feedback loop analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# =====================================================================
# Step 3: Prompt Versioning
# =====================================================================


class TestPromptVersioning:
    """Versioned adjudication prompts."""

    def test_load_prompt_v1(self) -> None:
        from llm_judge.calibration.prompts import load_prompt

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")

        pt = load_prompt("chat_quality", "v1", prompts_dir)
        assert pt.prompt_id == "chat_quality"
        assert pt.version == "v1"
        assert "relevance" in pt.dimensions
        assert len(pt.system_prompt) > 10

    def test_load_latest_prompt(self) -> None:
        from llm_judge.calibration.prompts import load_latest_prompt

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")

        pt = load_latest_prompt("chat_quality", prompts_dir)
        assert pt.prompt_id == "chat_quality"

    def test_list_prompts(self) -> None:
        from llm_judge.calibration.prompts import list_prompts

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")

        prompts = list_prompts(prompts_dir)
        assert len(prompts) >= 1
        assert prompts[0]["prompt_id"] == "chat_quality"

    def test_render_prompt(self) -> None:
        from llm_judge.calibration.prompts import load_prompt, render_prompt

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")

        pt = load_prompt("chat_quality", "v1", prompts_dir)
        sys_msg, user_msg = render_prompt(
            pt,
            conversation="USER: What is AI?",
            candidate_answer="AI stands for artificial intelligence.",
        )
        assert "quality evaluator" in sys_msg.lower()
        assert "What is AI?" in user_msg
        assert "artificial intelligence" in user_msg

    def test_diff_same_version(self) -> None:
        from llm_judge.calibration.prompts import diff_prompts

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")

        result = diff_prompts("chat_quality", "v1", "v1", prompts_dir)
        assert not result["has_changes"]

    def test_prompt_missing_raises(self) -> None:
        from llm_judge.calibration.prompts import load_prompt

        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent", "v99")


# =====================================================================
# Step 6: Hallucination Detection
# =====================================================================


class TestHallucinationDetection:
    """Detect ungrounded claims and fabricated citations."""

    def test_grounded_response_passes(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Paris is the capital of France.",
            context="France is a country in Europe. Paris is the capital of France.",
            case_id="c1",
        )
        assert result.risk_score < 0.3
        assert result.grounding_ratio > 0.5

    def test_ungrounded_response_flagged(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="Studies show that 97% of scientists agree on climate change according to NASA research published in 2023.",
            context="What is the weather like today?",
            case_id="c2",
        )
        assert result.risk_score > 0.1
        assert result.grounding_ratio < 0.5

    def test_fabricated_citation_detected(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="According to Smith et al. (2024), the results clearly demonstrate improvement [1].",
            context="The weather is sunny today.",
            case_id="c3",
        )
        assert result.unverifiable_citations >= 1

    def test_batch_analysis(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucinations_batch

        cases = [
            {
                "case_id": "c1",
                "conversation": [{"role": "user", "content": "What is Python?"}],
                "candidate_answer": "Python is a programming language.",
            },
            {
                "case_id": "c2",
                "conversation": [{"role": "user", "content": "Hi"}],
                "candidate_answer": "According to studies, 85% of users prefer this approach [1].",
            },
        ]
        judgments = [{"case_id": "c1"}, {"case_id": "c2"}]

        result = check_hallucinations_batch(cases=cases, judgments=judgments)
        assert result["total_checked"] == 2
        assert result["avg_grounding_ratio"] >= 0

    def test_empty_response_handled(self) -> None:
        from llm_judge.calibration.hallucination import check_hallucination

        result = check_hallucination(
            response="",
            context="Some context here.",
            case_id="empty",
        )
        assert result.risk_score == 0.0


# =====================================================================
# Step 9: Test Case Generation
# =====================================================================


class TestCaseGeneration:
    """Generate evaluation test cases from templates and documents."""

    def test_generate_happy_path(self) -> None:
        from llm_judge.calibration.testgen import generate_template_cases

        cases = generate_template_cases(categories=["happy_path"])
        assert len(cases) > 0
        for c in cases:
            assert c.category == "happy_path"
            assert c.expected_decision == "pass"
            assert c.generation_method == "template"

    def test_generate_adversarial(self) -> None:
        from llm_judge.calibration.testgen import generate_template_cases

        cases = generate_template_cases(categories=["adversarial"])
        assert len(cases) > 0
        for c in cases:
            assert c.category == "adversarial"
            assert c.expected_decision == "fail"

    def test_generate_edge_cases(self) -> None:
        from llm_judge.calibration.testgen import generate_template_cases

        cases = generate_template_cases(categories=["edge_case"])
        assert len(cases) > 0

    def test_generate_all_categories(self) -> None:
        from llm_judge.calibration.testgen import generate_template_cases

        cases = generate_template_cases()
        categories = {c.category for c in cases}
        assert "happy_path" in categories
        assert "edge_case" in categories
        assert "adversarial" in categories

    def test_generate_from_document(self) -> None:
        from llm_judge.calibration.testgen import generate_from_document

        doc = (
            "Python is a high-level programming language created by Guido van Rossum. "
            "It emphasizes code readability and supports multiple paradigms. "
            "Python is widely used in data science and machine learning applications. "
            "The language has a large standard library and active community."
        )
        cases = generate_from_document(document_text=doc, max_cases=3)
        assert len(cases) > 0
        assert cases[0].category == "rag_verification"

    def test_unique_case_ids(self) -> None:
        from llm_judge.calibration.testgen import generate_template_cases

        cases = generate_template_cases()
        ids = [c.case_id for c in cases]
        assert len(ids) == len(set(ids)), "Case IDs must be unique"

    def test_export_dataset(self, tmp_path: Path) -> None:
        from llm_judge.calibration.testgen import (
            export_generated_dataset,
            generate_template_cases,
        )

        cases = generate_template_cases(max_per_category=3)
        ds_dir = export_generated_dataset(
            cases,
            dataset_id="test_gen",
            output_dir=tmp_path,
        )

        assert (ds_dir / "data.jsonl").exists()
        assert (ds_dir / "dataset.yaml").exists()

        # Verify JSONL is valid
        lines = (ds_dir / "data.jsonl").read_text().strip().split("\n")
        assert len(lines) == len(cases)
        for line in lines:
            row = json.loads(line)
            assert "case_id" in row
            assert "conversation" in row


# =====================================================================
# Step 10: Feedback Loop
# =====================================================================


class TestFeedbackLoop:
    """Analyze LLM-human disagreements and generate recommendations."""

    def _make_adjudication_data(self) -> list[dict[str, Any]]:
        return [
            {
                "state": "resolved",
                "llm_decision": "pass",
                "human_decision": "pass",
                "llm_scores": {
                    "relevance": 4,
                    "clarity": 4,
                    "correctness": 4,
                    "tone": 4,
                },
                "human_scores": {
                    "relevance": 4,
                    "clarity": 3,
                    "correctness": 5,
                    "tone": 4,
                },
            },
            {
                "state": "resolved",
                "llm_decision": "pass",
                "human_decision": "fail",  # disagreement
                "llm_scores": {
                    "relevance": 4,
                    "clarity": 4,
                    "correctness": 4,
                    "tone": 4,
                },
                "human_scores": {
                    "relevance": 2,
                    "clarity": 2,
                    "correctness": 2,
                    "tone": 3,
                },
            },
            {
                "state": "resolved",
                "llm_decision": "pass",
                "human_decision": "pass",
                "llm_scores": {
                    "relevance": 5,
                    "clarity": 5,
                    "correctness": 5,
                    "tone": 5,
                },
                "human_scores": {
                    "relevance": 3,
                    "clarity": 3,
                    "correctness": 4,
                    "tone": 3,
                },
            },
            {
                "state": "pending",  # not resolved — should be excluded
                "llm_decision": "fail",
            },
        ]

    def test_analyze_feedback(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)

        assert report.total_resolved == 3
        assert 0 < report.decision_agreement_rate < 1  # 2/3 agreement

    def test_dimension_feedback(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)

        assert "relevance" in report.dimension_feedback
        rel = report.dimension_feedback["relevance"]
        assert rel.total_cases == 3
        assert rel.llm_higher > 0  # LLM scored higher in case 2 and 3

    def test_bias_detection(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)

        # With these test cases, LLM is generally lenient (scores higher than human)
        lenient_dims = [
            dim
            for dim, df in report.dimension_feedback.items()
            if df.bias_direction == "lenient"
        ]
        assert len(lenient_dims) > 0

    def test_recommendations_generated(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)

        assert len(report.recommendations) > 0

    def test_empty_data_handled(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        report = analyze_feedback(adjudication_data=[])
        assert report.total_resolved == 0
        assert len(report.recommendations) > 0  # should recommend getting data

    def test_save_report(self, tmp_path: Path) -> None:
        from llm_judge.calibration.feedback import (
            analyze_feedback,
            save_feedback_report,
        )

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)
        path = save_feedback_report(report, output_dir=tmp_path)

        assert path.exists()
        saved = json.loads(path.read_text())
        assert saved["total_resolved"] == 3
        assert "recommendations" in saved

    def test_systematic_patterns_detected(self) -> None:
        from llm_judge.calibration.feedback import analyze_feedback

        data = self._make_adjudication_data()
        report = analyze_feedback(adjudication_data=data)

        assert len(report.systematic_patterns) > 0
