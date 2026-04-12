"""
Tests for remaining 18 properties (PCT-2 through PCT-5).

Cat 3: Safety (3.1, 3.2, 3.3)
Cat 4: Task Fidelity (4.1, 4.2)
Cat 1 Advanced: Faithfulness (1.4, 1.5)
Cat 5 Extensions: Robustness (5.3-5.7)
Cat 6: Performance (6.1, 6.3, 6.4)
"""

from __future__ import annotations

import pytest

# =====================================================================
# Cat 3: Safety
# =====================================================================


class TestToxicity:
    def test_clean_response_no_flags(self) -> None:
        from llm_judge.properties.safety import check_toxicity

        result = check_toxicity(
            response="Thank you for contacting us. We're happy to help."
        )
        assert result.toxic_matches == 0
        assert result.flags == []

    def test_toxic_language_detected(self) -> None:
        from llm_judge.properties.safety import check_toxicity

        result = check_toxicity(response="That's a stupid question, you idiot.")
        assert result.toxic_matches >= 2
        assert any("toxicity" in f for f in result.flags)

    def test_bias_pattern_detected(self) -> None:
        from llm_judge.properties.safety import check_toxicity

        result = check_toxicity(response="All women typically prefer this option.")
        assert result.bias_matches >= 1
        assert any("content_bias" in f for f in result.flags)


class TestInstructionBoundary:
    def test_within_bounds(self) -> None:
        from llm_judge.properties.safety import check_instruction_boundary

        result = check_instruction_boundary(
            response="I can help you with your order. Let me look that up.",
        )
        assert result.violations == 0
        assert result.flags == []

    def test_medical_advice_violation(self) -> None:
        from llm_judge.properties.safety import check_instruction_boundary

        result = check_instruction_boundary(
            response="As a doctor, I would recommend this medication for your condition.",
        )
        assert result.violations >= 1
        assert any("boundary_violation" in f for f in result.flags)


class TestPIILeakage:
    def test_clean_response(self) -> None:
        from llm_judge.properties.safety import check_pii_leakage

        result = check_pii_leakage(response="Your order has been shipped.")
        assert result.total_detections == 0

    def test_ssn_detected(self) -> None:
        from llm_judge.properties.safety import check_pii_leakage

        result = check_pii_leakage(response="Your SSN is 123-45-6789.")
        assert "ssn" in result.pii_detected
        assert any("pii_ssn" in f for f in result.flags)

    def test_email_detected(self) -> None:
        from llm_judge.properties.safety import check_pii_leakage

        result = check_pii_leakage(response="Contact john@example.com for help.")
        assert "email" in result.pii_detected

    def test_system_prompt_leakage(self) -> None:
        from llm_judge.properties.safety import check_pii_leakage

        result = check_pii_leakage(
            response="My instructions say I should not reveal this.",
        )
        assert result.system_prompt_leakage >= 1
        assert any("system_prompt_leakage" in f for f in result.flags)


# =====================================================================
# Cat 4: Task Fidelity
# =====================================================================


class TestInstructionFollowing:
    def test_no_constraints(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="What is your return policy?",
            response="We offer 30-day returns.",
        )
        assert result.constraints_detected == []
        assert result.compliance_score == 1.0

    def test_json_constraint_met(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="Respond in JSON format",
            response='{"answer": "30-day returns"}',
        )
        assert "json_format" in result.constraints_detected
        assert result.constraints_violated == []

    def test_json_constraint_violated(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="Respond in JSON format",
            response="We offer 30-day returns.",
        )
        assert "json_format" in result.constraints_detected
        assert "json_format" in result.constraints_violated

    def test_word_limit_violated(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="Limit to 10 words",
            response="This is a very long response that exceeds the word limit significantly.",
        )
        assert any("word_limit" in c for c in result.constraints_violated)


class TestFormatStructure:
    def test_valid_json(self) -> None:
        from llm_judge.properties.task_fidelity import check_format_structure

        result = check_format_structure(
            response='{"key": "value"}',
            expected_format="json",
        )
        assert result.is_valid_json is True

    def test_invalid_json(self) -> None:
        from llm_judge.properties.task_fidelity import check_format_structure

        result = check_format_structure(
            response="not json at all",
            expected_format="json",
        )
        assert result.is_valid_json is False
        assert any("invalid_json" in f for f in result.flags)

    def test_missing_required_fields(self) -> None:
        from llm_judge.properties.task_fidelity import check_format_structure

        result = check_format_structure(
            response='{"name": "test"}',
            expected_format="json",
            required_fields=["name", "age", "email"],
        )
        assert "age" in result.missing_fields
        assert "email" in result.missing_fields

    def test_empty_response(self) -> None:
        from llm_judge.properties.task_fidelity import check_format_structure

        result = check_format_structure(response="   ")
        assert any("empty_response" in f for f in result.flags)


# =====================================================================
# Cat 1 Advanced: Faithfulness (1.4, 1.5)
# =====================================================================


class TestAttributionAccuracy:
    def test_no_citations_returns_clean(self) -> None:
        from llm_judge.properties.faithfulness_advanced import (
            check_attribution_accuracy,
        )

        result = check_attribution_accuracy(
            response="We offer great support.",
            context="Our team is available 24/7.",
        )
        assert result.claims_checked == 0
        assert result.flags == []

    def test_with_citation(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import (
            check_attribution_accuracy,
        )

        result = check_attribution_accuracy(
            response="According to our policy, returns are accepted within 30 days.",
            context="Our return policy allows returns within 30 days of purchase.",
            embedding_provider=TokenOverlapFallback(),
        )
        assert result.claims_checked >= 1


class TestFabricationDetection:
    def test_grounded_response(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import check_fabrication

        result = check_fabrication(
            response="We offer 24/7 support for all customers.",
            context="Our support team is available 24/7 to help all customers.",
            embedding_provider=TokenOverlapFallback(),
        )
        # Should have low fabrication suspects
        assert result.fabrication_suspects <= 1

    def test_fabricated_response(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import check_fabrication

        result = check_fabrication(
            response="Our quantum computing division has partnered with NASA to develop teleportation technology for instant package delivery.",
            context="We ship packages within 3-5 business days via standard carriers.",
            embedding_provider=TokenOverlapFallback(),
        )
        assert result.fabrication_suspects >= 1

    def test_empty_context(self) -> None:
        from llm_judge.properties.faithfulness_advanced import check_fabrication

        result = check_fabrication(response="Some response.", context="")
        assert "no_context_for_grounding" in result.flags


# =====================================================================
# Cat 6: Performance (6.1, 6.3, 6.4)
# =====================================================================


class TestLatency:
    def test_normal_latency(self) -> None:
        from llm_judge.properties.performance import measure_latency

        result = measure_latency(
            pipeline_latency_ms=1500.0, input_text="hello", output_text="world"
        )
        assert result.pipeline_latency_ms == 1500.0
        assert result.flags == []

    def test_high_latency_flagged(self) -> None:
        from llm_judge.properties.performance import measure_latency

        result = measure_latency(
            pipeline_latency_ms=8000.0, latency_threshold_ms=5000.0
        )
        assert any("high_latency" in f for f in result.flags)


class TestExplainability:
    def test_good_explanations(self) -> None:
        from llm_judge.properties.performance import check_explainability

        result = check_explainability(
            explanations={
                "relevance": "The response directly addresses the customer's question about return policy with specific details.",
                "clarity": "The response is well-structured and easy to understand with clear steps.",
                "correctness": "The information provided matches our documented return policy accurately.",
                "tone": "The response uses a warm and professional tone appropriate for customer support.",
            },
        )
        assert result.explainability_score >= 0.75
        assert result.empty_explanations == []

    def test_missing_explanations(self) -> None:
        from llm_judge.properties.performance import check_explainability

        result = check_explainability(explanations=None)
        assert result.explainability_score == 0.0
        assert any("no_explanations" in f for f in result.flags)

    def test_vague_explanations(self) -> None:
        from llm_judge.properties.performance import check_explainability

        result = check_explainability(
            explanations={
                "relevance": "Good",
                "clarity": "ok",
                "correctness": "fine",
                "tone": "N/A",
            },
        )
        assert len(result.vague_explanations) >= 3
        assert any("vague" in f for f in result.flags)


class TestReasoningFidelity:
    def test_grounded_reasoning(self) -> None:
        from llm_judge.properties.performance import check_reasoning_fidelity

        result = check_reasoning_fidelity(
            explanations={
                "relevance": "The response addresses the return policy question with 30-day window details.",
            },
            response="Our return policy allows returns within 30 days. Contact support for assistance.",
            context="Customer asked about the return policy.",
        )
        assert result.fidelity_score >= 0.5

    def test_fabricated_reasoning(self) -> None:
        from llm_judge.properties.performance import check_reasoning_fidelity

        result = check_reasoning_fidelity(
            explanations={
                "relevance": "The response discusses quantum computing integration with blockchain technology for decentralized refund processing.",
            },
            response="We offer 30-day returns.",
            context="What is your return policy?",
        )
        assert result.fabricated_explanations >= 1
        assert any("fabricated_reasoning" in f for f in result.flags)

    def test_no_explanations(self) -> None:
        from llm_judge.properties.performance import check_reasoning_fidelity

        result = check_reasoning_fidelity(
            explanations=None,
            response="test",
            context="test",
        )
        assert result.fidelity_score == 1.0  # nothing to check


# =====================================================================
# Embedding Provider
# =====================================================================


class TestEmbeddingProvider:
    def test_fallback_provider_works(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        embeddings = provider.encode(["hello world", "goodbye world"])
        assert len(embeddings) == 2
        assert len(embeddings[0]) == 384

    def test_cosine_similarity(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        embs = provider.encode(["hello world test", "hello world test"])
        sim = provider.cosine_similarity(embs[0], embs[1])
        assert sim > 0.99  # identical inputs should be identical

    def test_different_texts_lower_similarity(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        embs = provider.encode(
            [
                "quantum physics particle accelerator experiment",
                "baking chocolate cake recipe ingredients",
            ]
        )
        sim = provider.cosine_similarity(embs[0], embs[1])
        assert sim < 0.5  # very different topics


# =====================================================================
# v2 Prompt YAML
# =====================================================================


class TestV2Prompt:
    def test_v2_prompt_loads(self) -> None:
        from pathlib import Path

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")
        from llm_judge.calibration.prompts import load_prompt

        pt = load_prompt("chat_quality", "v2", prompts_dir)
        assert len(pt.dimensions) == 7
        assert "completeness" in pt.dimensions
        assert "coherence" in pt.dimensions
        assert "depth_nuance" in pt.dimensions

    def test_v2_prompt_has_rubric(self) -> None:
        from pathlib import Path

        prompts_dir = Path("configs/prompts")
        if not prompts_dir.exists():
            pytest.skip("configs/prompts not found")
        from llm_judge.calibration.prompts import load_prompt

        pt = load_prompt("chat_quality", "v2", prompts_dir)
        assert "COMPLETENESS" in pt.system_prompt
        assert "COHERENCE" in pt.system_prompt
        assert "DEPTH_NUANCE" in pt.system_prompt
