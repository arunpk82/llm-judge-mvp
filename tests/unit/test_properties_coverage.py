"""
Coverage boost tests for new property modules.

Targets uncovered branches in:
- properties/robustness.py (calibration diagnostics)
- properties/safety.py (edge cases)
- properties/task_fidelity.py (edge cases)
- properties/faithfulness_advanced.py (edge cases)
- properties/performance.py (edge cases)
- properties/__init__.py (embedding provider)
- integrated_judge.py (pipeline branches)
- property_config.py (additional edges)
"""

from __future__ import annotations

from typing import Any, Literal

# =====================================================================
# Robustness module — calibration diagnostics with mock judge
# =====================================================================
from llm_judge.judge_base import JudgeEngine as _JE
from llm_judge.schemas import Message, PredictRequest, PredictResponse


class _MockJudge(_JE):
    """Deterministic mock judge for robustness tests."""

    def __init__(
        self, score: float = 3.5, decision: Literal["pass", "fail"] = "pass"
    ) -> None:
        self._score = score
        self._decision = decision
        self._call_count = 0

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        self._call_count += 1
        return PredictResponse(
            decision=self._decision,
            overall_score=self._score,
            scores={"relevance": 4, "clarity": 4, "correctness": 3, "tone": 4},
            confidence=0.8,
            flags=[],
            explanations={
                "relevance": "Good",
                "clarity": "Clear",
                "correctness": "OK",
                "tone": "Nice",
            },
        )


def _make_case(
    case_id: str = "test_001", answer: str = "Test answer."
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "conversation": [{"role": "user", "content": "What is your policy?"}],
        "candidate_answer": answer,
        "rubric_id": "chat_quality",
        "human_decision": "pass",
        "expected_decision": "pass",
    }


class TestSelfPreferenceBias:
    def test_insufficient_data(self) -> None:
        from llm_judge.properties.robustness import check_self_preference_bias

        result = check_self_preference_bias(
            judge=_MockJudge(),
            llm_written_cases=[_make_case()],
            human_written_cases=[_make_case()],
        )
        assert result.status == "INSUFFICIENT_DATA"

    def test_no_bias(self) -> None:
        from llm_judge.properties.robustness import check_self_preference_bias

        result = check_self_preference_bias(
            judge=_MockJudge(score=3.5),
            llm_written_cases=[_make_case(f"llm_{i}") for i in range(5)],
            human_written_cases=[_make_case(f"human_{i}") for i in range(5)],
        )
        assert result.status == "PASS"
        assert result.preference_delta == 0.0

    def test_bias_detected(self) -> None:
        from llm_judge.properties.robustness import check_self_preference_bias

        class BiasedJudge(_JE):
            def __init__(self) -> None:
                self._call = 0

            def evaluate(self, request: PredictRequest) -> PredictResponse:
                self._call += 1
                score = 4.5 if self._call <= 5 else 2.0
                return PredictResponse(
                    decision="pass",
                    overall_score=score,
                    scores={"relevance": 4},
                    confidence=0.8,
                    flags=[],
                )

        result = check_self_preference_bias(
            judge=BiasedJudge(),
            llm_written_cases=[_make_case(f"llm_{i}") for i in range(5)],
            human_written_cases=[_make_case(f"human_{i}") for i in range(5)],
        )
        assert result.status == "FAIL"
        assert abs(result.preference_delta) > 0.5


class TestConsistency:
    def test_insufficient_data(self) -> None:
        from llm_judge.properties.robustness import check_consistency

        result = check_consistency(judge=_MockJudge(), paraphrase_pairs=[])
        assert result.status == "INSUFFICIENT_DATA"

    def test_consistent_judge(self) -> None:
        from llm_judge.properties.robustness import check_consistency

        pairs = [(_make_case(f"a_{i}"), _make_case(f"b_{i}")) for i in range(5)]
        result = check_consistency(judge=_MockJudge(), paraphrase_pairs=pairs)
        assert result.status == "PASS"
        assert result.consistent == 5


class TestAdversarialResilience:
    def test_insufficient_data(self) -> None:
        from llm_judge.properties.robustness import check_adversarial_resilience

        result = check_adversarial_resilience(judge=_MockJudge(), adversarial_cases=[])
        assert result.status == "INSUFFICIENT_DATA"

    def test_resilient_judge(self) -> None:
        from llm_judge.properties.robustness import check_adversarial_resilience

        cases = [_make_case(f"adv_{i}") for i in range(5)]
        result = check_adversarial_resilience(
            judge=_MockJudge(), adversarial_cases=cases
        )
        assert result.status == "PASS"
        assert result.pass_rate >= 0.85

    def test_failing_judge(self) -> None:
        from llm_judge.properties.robustness import check_adversarial_resilience

        cases = [_make_case(f"adv_{i}") for i in range(5)]
        for c in cases:
            c["expected_decision"] = "fail"  # judge always returns pass
        result = check_adversarial_resilience(
            judge=_MockJudge(), adversarial_cases=cases
        )
        assert result.status == "FAIL"


class TestEdgeCases:
    def test_deterministic_judge_handles_edges(self) -> None:
        from llm_judge.properties.robustness import check_edge_cases

        result = check_edge_cases(judge=_MockJudge())
        assert result.tested == 4
        assert result.handled_gracefully == 4
        assert result.crashed == 0

    def test_crashing_judge(self) -> None:
        from llm_judge.properties.robustness import check_edge_cases

        class CrashJudge(_JE):
            def evaluate(self, request: PredictRequest) -> PredictResponse:
                raise RuntimeError("boom")

        result = check_edge_cases(judge=CrashJudge())
        assert result.crashed == 4
        assert result.status == "FAIL"
        assert any("edge_case_crashes" in f for f in result.flags)


class TestReproducibility:
    def test_insufficient_data(self) -> None:
        from llm_judge.properties.robustness import check_reproducibility

        result = check_reproducibility(judge=_MockJudge(), cases=[])
        assert result.status == "INSUFFICIENT_DATA"

    def test_reproducible_judge(self) -> None:
        from llm_judge.properties.robustness import check_reproducibility

        cases = [_make_case(f"rep_{i}") for i in range(5)]
        result = check_reproducibility(judge=_MockJudge(), cases=cases)
        assert result.status == "PASS"
        assert result.identity_rate >= 0.90


# =====================================================================
# Safety edge cases
# =====================================================================


class TestSafetyEdges:
    def test_no_toxic_no_bias(self) -> None:
        from llm_judge.properties.safety import check_toxicity

        result = check_toxicity(response="Thank you for your patience.")
        assert result.toxic_matches == 0
        assert result.bias_matches == 0

    def test_multiple_pii_types(self) -> None:
        from llm_judge.properties.safety import check_pii_leakage

        result = check_pii_leakage(
            response="Call 555-123-4567 or email test@example.com, SSN 123-45-6789"
        )
        assert len(result.pii_detected) >= 3
        assert result.total_detections >= 3

    def test_boundary_no_violation(self) -> None:
        from llm_judge.properties.safety import check_instruction_boundary

        result = check_instruction_boundary(
            response="I'd be happy to help you with your shipping question.",
        )
        assert result.violations == 0


# =====================================================================
# Task Fidelity edge cases
# =====================================================================


class TestTaskFidelityEdges:
    def test_json_in_markdown_fences(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="Respond in JSON format",
            response='```json\n{"answer": "hello"}\n```',
        )
        assert "json_format" not in result.constraints_violated

    def test_step_by_step_detected(self) -> None:
        from llm_judge.properties.task_fidelity import check_instruction_following

        result = check_instruction_following(
            query="Explain step by step how to return an item",
            response="Step 1: Go to orders. Step 2: Click return.",
        )
        assert "step_by_step" in result.constraints_detected

    def test_format_structure_json_with_fields(self) -> None:
        from llm_judge.properties.task_fidelity import check_format_structure

        result = check_format_structure(
            response='{"name": "test", "age": 30}',
            expected_format="json",
            required_fields=["name", "age"],
        )
        assert result.is_valid_json
        assert result.has_required_fields
        assert result.missing_fields == []


# =====================================================================
# Performance edge cases
# =====================================================================


class TestPerformanceEdges:
    def test_explainability_partial(self) -> None:
        from llm_judge.properties.performance import check_explainability

        result = check_explainability(
            explanations={
                "relevance": "The response fully addresses the user's question about returns with specific policy details.",
                "clarity": "",
                "correctness": "Accurate information provided.",
                "tone": "Good",
            },
        )
        assert "clarity" in result.empty_explanations
        assert result.explainability_score < 1.0

    def test_reasoning_fidelity_with_eval_vocabulary(self) -> None:
        from llm_judge.properties.performance import check_reasoning_fidelity

        result = check_reasoning_fidelity(
            explanations={
                "relevance": "The response provides a relevant answer to the question."
            },
            response="We offer 30-day returns.",
            context="What is your return policy?",
        )
        # All evaluation vocabulary — should be considered grounded
        assert result.fidelity_score >= 0.5

    def test_latency_within_threshold(self) -> None:
        from llm_judge.properties.performance import measure_latency

        result = measure_latency(
            pipeline_latency_ms=2000.0,
            input_text="a" * 400,
            output_text="b" * 200,
            latency_threshold_ms=5000.0,
        )
        assert result.estimated_input_tokens == 100
        assert result.estimated_output_tokens == 50
        assert result.flags == []


# =====================================================================
# Embedding provider edge cases
# =====================================================================


class TestEmbeddingEdges:
    def test_max_similarity(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        embs = provider.encode(["hello world", "hello world", "goodbye moon"])
        query = embs[0]
        candidates = embs[1:]
        sim = provider.max_similarity(query, candidates)
        assert sim > 0.9

    def test_max_similarity_empty(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        embs = provider.encode(["hello"])
        assert provider.max_similarity(embs[0], []) == 0.0

    def test_dimension(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback(dimension=128)
        assert provider.dimension() == 128
        embs = provider.encode(["test"])
        assert len(embs[0]) == 128

    def test_cosine_zero_vector(self) -> None:
        from llm_judge.properties import TokenOverlapFallback

        provider = TokenOverlapFallback()
        zero = [0.0] * 384
        normal = provider.encode(["hello world"])[0]
        assert provider.cosine_similarity(zero, normal) == 0.0


# =====================================================================
# Advanced faithfulness edge cases
# =====================================================================


class TestFaithfulnessEdges:
    def test_attribution_empty_context_sentences(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import (
            check_attribution_accuracy,
        )

        result = check_attribution_accuracy(
            response="According to sources, the sky is blue.",
            context="Hi",  # too short to split into sentences
            embedding_provider=TokenOverlapFallback(),
        )
        assert "no_context_sentences" in result.flags

    def test_fabrication_empty_response(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import check_fabrication

        result = check_fabrication(
            response="Short.",
            context="Some context here for testing purposes.",
            embedding_provider=TokenOverlapFallback(),
        )
        assert result.sentences_checked == 0

    def test_attribution_no_citations(self) -> None:
        from llm_judge.properties import TokenOverlapFallback
        from llm_judge.properties.faithfulness_advanced import (
            check_attribution_accuracy,
        )

        result = check_attribution_accuracy(
            response="The weather is nice today.",
            context="We sell products online.",
            embedding_provider=TokenOverlapFallback(),
        )
        assert result.claims_checked == 0


# =====================================================================
# Property config additional coverage
# =====================================================================


class TestPropertyConfigEdges:
    def test_get_enabled_by_category_empty(self) -> None:
        from llm_judge.property_config import PropertyConfig, PropertyRegistry

        reg = PropertyRegistry(
            {
                "test": PropertyConfig(
                    name="test",
                    id="1.1",
                    category="faithfulness",
                    enabled=False,
                    gate_mode="informational",
                ),
            }
        )
        assert reg.get_enabled_by_category("faithfulness") == {}
        assert reg.get_enabled_by_category("nonexistent") == {}

    def test_detection_coverage_all_disabled(self) -> None:
        from llm_judge.property_config import PropertyConfig, PropertyRegistry

        reg = PropertyRegistry(
            {
                "a": PropertyConfig(
                    name="a",
                    id="1",
                    category="faithfulness",
                    enabled=False,
                    gate_mode="informational",
                ),
                "b": PropertyConfig(
                    name="b",
                    id="2",
                    category="safety",
                    enabled=False,
                    gate_mode="informational",
                ),
            }
        )
        cov = reg.detection_coverage()
        assert cov.enabled == 0
        assert cov.disabled == 2
        assert cov.enabled_pct == 0.0

    def test_filter_for_unknown_persona(self) -> None:
        from llm_judge.property_config import PropertyConfig, PropertyRegistry

        reg = PropertyRegistry(
            {
                "test": PropertyConfig(
                    name="test",
                    id="1.1",
                    category="faithfulness",
                    enabled=True,
                    gate_mode="informational",
                    visibility={"ml_engineer": "show"},
                ),
            }
        )
        filtered = reg.filter_for_persona({"test": {"score": 1}}, "unknown_persona")
        assert filtered == {}


# =====================================================================
# Integrated judge edge cases (no LLM call)
# =====================================================================


class TestIntegratedJudgeEdges:
    def test_enriched_response_to_dict_no_hallucination(self) -> None:
        from llm_judge.integrated_judge import EnrichedResponse

        resp = EnrichedResponse(
            predict_response=PredictResponse(
                decision="pass",
                overall_score=4.0,
                scores={"relevance": 4},
                confidence=0.9,
                flags=[],
            ),
        )
        d = resp.to_dict()
        assert "hallucination" not in d
        assert d["prompt_version"] is None

    def test_enriched_response_all_flags_empty(self) -> None:
        from llm_judge.integrated_judge import EnrichedResponse

        resp = EnrichedResponse(
            predict_response=PredictResponse(
                decision="pass",
                overall_score=4.0,
                scores={"relevance": 4},
                confidence=0.9,
                flags=[],
            ),
        )
        assert resp.all_flags() == []

    def test_build_context(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there"),
            ],
            candidate_answer="Test",
            rubric_id="chat_quality",
        )
        ctx = _build_context(req)
        assert "Hello" in ctx
        assert "Hi there" in ctx

    def test_build_query(self) -> None:
        from llm_judge.integrated_judge import _build_query

        req = PredictRequest(
            conversation=[
                Message(role="user", content="First question"),
                Message(role="assistant", content="Answer"),
                Message(role="user", content="Follow up"),
            ],
            candidate_answer="Test",
            rubric_id="chat_quality",
        )
        assert _build_query(req) == "Follow up"

    def test_prop_id_gm_missing(self) -> None:
        from llm_judge.integrated_judge import _prop_id_gm
        from llm_judge.property_config import PropertyRegistry

        reg = PropertyRegistry({})
        pid, gm = _prop_id_gm(reg, "nonexistent")
        assert pid == "nonexistent"
        assert gm == "informational"
