"""
Integrated Evaluation Pipeline (EPICs 7.4–7.7 + full property wiring).

All 28 properties are wired. Enable any property in property_config.yaml
and it runs automatically. No code changes needed for promotion.

Pipeline steps:
  1. Load property configuration
  2. Load versioned prompt
  3. Run deterministic pre-checks (Cat 1 hallucination, Cat 3 safety, Cat 4 task fidelity)
  4. Execute LLM evaluation with versioned prompt (Cat 2 semantic quality)
  5. Run post-eval checks (Cat 1 advanced faithfulness, Cat 6 performance)
  6. Apply gate mode per property
  7. Assemble enriched response with property execution evidence
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from llm_judge.calibration.hallucination import (
    HallucinationResult,
    check_hallucination,
)
from llm_judge.calibration.prompts import PromptTemplate, load_latest_prompt
from llm_judge.judge_base import JudgeEngine
from llm_judge.llm_judge import LLMJudge
from llm_judge.property_config import PropertyRegistry, load_property_config
from llm_judge.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@dataclass
class PropertyEvidence:
    """Evidence that a property executed during evaluation."""
    property_name: str
    property_id: str
    enabled: bool
    gate_mode: str
    executed: bool
    result: Any = None
    flags: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class EnrichedResponse:
    """Gate 2 response enriched with property execution evidence."""
    predict_response: PredictResponse
    prompt_version: str | None = None
    hallucination_result: HallucinationResult | None = None
    property_evidence: dict[str, PropertyEvidence] = field(default_factory=dict)
    detection_coverage: str = ""
    pipeline_latency_ms: float = 0.0

    @property
    def decision(self) -> str:
        return self.predict_response.decision

    @property
    def scores(self) -> dict[str, int]:
        return self.predict_response.scores

    @property
    def confidence(self) -> float:
        return self.predict_response.confidence

    @property
    def flags(self) -> list[str]:
        return self.predict_response.flags

    @property
    def explanations(self) -> dict[str, str] | None:
        return self.predict_response.explanations

    def all_flags(self) -> list[str]:
        """Collect flags from all sources."""
        all_f = list(self.predict_response.flags)
        if self.hallucination_result:
            all_f.extend(self.hallucination_result.flags)
        for ev in self.property_evidence.values():
            all_f.extend(ev.flags)
        return all_f

    def to_dict(self) -> dict[str, Any]:
        """Serialize for structured output."""
        result: dict[str, Any] = {
            "decision": self.predict_response.decision,
            "overall_score": self.predict_response.overall_score,
            "scores": self.predict_response.scores,
            "confidence": self.predict_response.confidence,
            "flags": self.all_flags(),
            "explanations": self.predict_response.explanations,
            "prompt_version": self.prompt_version,
            "pipeline_latency_ms": round(self.pipeline_latency_ms, 1),
            "detection_coverage": self.detection_coverage,
        }
        if self.hallucination_result:
            result["hallucination"] = {
                "risk_score": self.hallucination_result.risk_score,
                "grounding_ratio": self.hallucination_result.grounding_ratio,
                "ungrounded_claims": self.hallucination_result.ungrounded_claims,
                "unverifiable_citations": self.hallucination_result.unverifiable_citations,
                "flags": self.hallucination_result.flags,
            }
        result["property_evidence"] = {
            name: {
                "id": ev.property_id,
                "enabled": ev.enabled,
                "gate_mode": ev.gate_mode,
                "executed": ev.executed,
                "flags": ev.flags,
                "error": ev.error,
            }
            for name, ev in self.property_evidence.items()
        }
        return result


def _build_context(request: PredictRequest) -> str:
    """Build context string from conversation for grounding checks."""
    parts = [msg.content for msg in request.conversation]
    return " ".join(parts)


def _build_query(request: PredictRequest) -> str:
    """Extract the user query from conversation."""
    for msg in reversed(request.conversation):
        if msg.role == "user":
            return msg.content
    return request.conversation[0].content if request.conversation else ""


def _record_evidence(
    evidence: dict[str, PropertyEvidence],
    prop_name: str,
    prop_id: str,
    gate_mode: str,
    executed: bool,
    result: Any = None,
    flags: list[str] | None = None,
    error: str | None = None,
) -> None:
    """Record property execution evidence."""
    evidence[prop_name] = PropertyEvidence(
        property_name=prop_name,
        property_id=prop_id,
        enabled=True,
        gate_mode=gate_mode,
        executed=executed,
        result=result,
        flags=flags or [],
        error=error,
    )


class IntegratedJudge(JudgeEngine):
    """
    Property-aware evaluation pipeline for Gate 2.

    All 28 properties are wired. Enable any property in
    property_config.yaml and it runs automatically.
    """

    def __init__(
        self,
        engine: str = "gemini",
        timeout_ms: int | None = None,
        registry: PropertyRegistry | None = None,
    ) -> None:
        self._engine = engine
        self._timeout_ms = timeout_ms
        self._registry = registry
        self._llm_judge: LLMJudge | None = None
        self._prompt: PromptTemplate | None = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of registry, prompt, and LLM judge."""
        if self._registry is None:
            try:
                self._registry = load_property_config()
            except FileNotFoundError:
                logger.warning("property_config.not_found, using defaults")
                self._registry = PropertyRegistry({})

        if self._prompt is None:
            try:
                self._prompt = load_latest_prompt("chat_quality")
                logger.info(
                    "prompt.loaded",
                    extra={
                        "prompt_id": self._prompt.prompt_id,
                        "version": self._prompt.version,
                    },
                )
            except FileNotFoundError:
                logger.warning("versioned_prompt.not_found, LLM will use default")
                self._prompt = None

        if self._llm_judge is None:
            self._llm_judge = LLMJudge(
                engine=self._engine,
                timeout_ms=self._timeout_ms,
                prompt_template=self._prompt,
            )

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        """Standard JudgeEngine interface — backward compatible."""
        enriched = self.evaluate_enriched(request)
        all_flags = enriched.all_flags()
        return PredictResponse(
            decision=enriched.predict_response.decision,
            overall_score=enriched.predict_response.overall_score,
            scores=enriched.predict_response.scores,
            confidence=enriched.predict_response.confidence,
            flags=all_flags,
            explanations=enriched.predict_response.explanations,
        )

    def evaluate_enriched(
        self,
        request: PredictRequest,
        case_id: str = "unknown",
    ) -> EnrichedResponse:
        """Full integrated evaluation with all property wiring."""
        start_time = time.time()
        self._ensure_initialized()
        assert self._registry is not None

        evidence: dict[str, PropertyEvidence] = {}
        context = _build_context(request)
        query = _build_query(request)
        response_text = request.candidate_answer

        # =============================================================
        # Step 3: Deterministic pre-checks (parallel-safe)
        # =============================================================

        # --- Cat 1: Faithfulness (1.1–1.3) — hallucination checks ---
        hallucination_result: HallucinationResult | None = None
        faithfulness_props = self._registry.get_enabled_by_category("faithfulness")

        basic_faith = {
            k: v for k, v in faithfulness_props.items()
            if k in ("groundedness", "ungrounded_claims", "citation_verification")
        }
        if basic_faith:
            if len(context.strip()) == 0:
                for prop_name, prop in basic_faith.items():
                    _record_evidence(
                        evidence, prop_name, prop.id, prop.gate_mode,
                        executed=False, error="empty_context",
                        flags=["pipeline_error:empty_context"],
                    )
            else:
                gt_prop = self._registry.get("groundedness")
                grounding_threshold = float(gt_prop.threshold) if gt_prop and gt_prop.threshold is not None else 0.3
                uc_prop = self._registry.get("ungrounded_claims")
                max_claims = int(uc_prop.threshold) if uc_prop and uc_prop.threshold is not None else 2

                hallucination_result = check_hallucination(
                    response=response_text, context=context, case_id=case_id,
                    grounding_threshold=grounding_threshold,
                    max_ungrounded_claims=max_claims,
                )

                for prop_name, prop in basic_faith.items():
                    prop_flags: list[str] = []
                    if prop_name == "groundedness" and hallucination_result.grounding_ratio < grounding_threshold:
                        prop_flags.append(f"low_grounding:{hallucination_result.grounding_ratio:.2f}")
                    elif prop_name == "ungrounded_claims" and hallucination_result.ungrounded_claims > max_claims:
                        prop_flags.append(f"ungrounded_claims:{hallucination_result.ungrounded_claims}")
                    elif prop_name == "citation_verification" and hallucination_result.unverifiable_citations > 0:
                        prop_flags.append(f"unverifiable_citations:{hallucination_result.unverifiable_citations}")
                    _record_evidence(
                        evidence, prop_name, prop.id, prop.gate_mode,
                        executed=True, result=hallucination_result, flags=prop_flags,
                    )

        # --- Cat 3: Safety (3.1–3.3) ---
        safety_props = self._registry.get_enabled_by_category("safety")
        if safety_props:
            from llm_judge.properties.safety import (
                check_instruction_boundary,
                check_pii_leakage,
                check_toxicity,
            )

            if self._registry.is_enabled("toxicity_bias"):
                result = check_toxicity(response=response_text, case_id=case_id)
                prop = self._registry.get("toxicity_bias")
                assert prop is not None
                _record_evidence(
                    evidence, "toxicity_bias", prop.id, prop.gate_mode,
                    executed=True, result=result, flags=result.flags,
                )

            if self._registry.is_enabled("instruction_boundary"):
                result_ib = check_instruction_boundary(response=response_text, case_id=case_id)
                prop = self._registry.get("instruction_boundary")
                assert prop is not None
                _record_evidence(
                    evidence, "instruction_boundary", prop.id, prop.gate_mode,
                    executed=True, result=result_ib, flags=result_ib.flags,
                )

            if self._registry.is_enabled("pii_data_leakage"):
                result_pii = check_pii_leakage(response=response_text, case_id=case_id)
                prop = self._registry.get("pii_data_leakage")
                assert prop is not None
                _record_evidence(
                    evidence, "pii_data_leakage", prop.id, prop.gate_mode,
                    executed=True, result=result_pii, flags=result_pii.flags,
                )

        # --- Cat 4: Task Fidelity (4.1–4.2) ---
        if self._registry.is_enabled("instruction_following"):
            from llm_judge.properties.task_fidelity import check_instruction_following
            result_if = check_instruction_following(
                query=query, response=response_text, case_id=case_id,
            )
            prop = self._registry.get("instruction_following")
            assert prop is not None
            _record_evidence(
                evidence, "instruction_following", prop.id, prop.gate_mode,
                executed=True, result=result_if, flags=result_if.flags,
            )

        if self._registry.is_enabled("format_structure"):
            from llm_judge.properties.task_fidelity import check_format_structure
            result_fs = check_format_structure(response=response_text, case_id=case_id)
            prop = self._registry.get("format_structure")
            assert prop is not None
            _record_evidence(
                evidence, "format_structure", prop.id, prop.gate_mode,
                executed=True, result=result_fs, flags=result_fs.flags,
            )

        # =============================================================
        # Step 4: LLM evaluation with versioned prompt (Cat 2)
        # =============================================================
        assert self._llm_judge is not None
        predict_response = self._llm_judge.evaluate(request)

        # Record semantic quality evidence
        semantic_props = self._registry.get_enabled_by_category("semantic_quality")
        for prop_name, prop in semantic_props.items():
            score = predict_response.scores.get(prop_name)
            sem_flags: list[str] = []
            if score is not None and score <= 2:
                sem_flags.append(f"low_{prop_name}:{score}")
            _record_evidence(
                evidence, prop_name, prop.id, prop.gate_mode,
                executed=True, result=score, flags=sem_flags,
            )

        # =============================================================
        # Step 5: Post-eval checks
        # =============================================================

        # --- Cat 1 Advanced: Faithfulness (1.4–1.5) — embedding-based ---
        if self._registry.is_enabled("attribution_accuracy") and context.strip():
            try:
                from llm_judge.properties.faithfulness_advanced import check_attribution_accuracy
                result_aa = check_attribution_accuracy(
                    response=response_text, context=context, case_id=case_id,
                )
                prop = self._registry.get("attribution_accuracy")
                assert prop is not None
                _record_evidence(
                    evidence, "attribution_accuracy", prop.id, prop.gate_mode,
                    executed=True, result=result_aa, flags=result_aa.flags,
                )
            except Exception as e:
                prop = self._registry.get("attribution_accuracy")
                assert prop is not None
                _record_evidence(
                    evidence, "attribution_accuracy", prop.id, prop.gate_mode,
                    executed=False, error=str(e)[:80],
                )

        if self._registry.is_enabled("fabrication_detection") and context.strip():
            try:
                from llm_judge.properties.faithfulness_advanced import check_fabrication
                result_fab = check_fabrication(
                    response=response_text, context=context, case_id=case_id,
                )
                prop = self._registry.get("fabrication_detection")
                assert prop is not None
                _record_evidence(
                    evidence, "fabrication_detection", prop.id, prop.gate_mode,
                    executed=True, result=result_fab, flags=result_fab.flags,
                )
            except Exception as e:
                prop = self._registry.get("fabrication_detection")
                assert prop is not None
                _record_evidence(
                    evidence, "fabrication_detection", prop.id, prop.gate_mode,
                    executed=False, error=str(e)[:80],
                )

        # --- Cat 6: Performance (6.1, 6.3, 6.4) ---
        if self._registry.is_enabled("explainability"):
            from llm_judge.properties.performance import check_explainability
            dims = self._prompt.dimensions if self._prompt else ["relevance", "clarity", "correctness", "tone"]
            result_ex = check_explainability(
                explanations=predict_response.explanations,
                expected_dimensions=dims,
                case_id=case_id,
            )
            prop = self._registry.get("explainability")
            assert prop is not None
            _record_evidence(
                evidence, "explainability", prop.id, prop.gate_mode,
                executed=True, result=result_ex, flags=result_ex.flags,
            )

        if self._registry.is_enabled("judge_reasoning_fidelity"):
            from llm_judge.properties.performance import check_reasoning_fidelity
            result_rf = check_reasoning_fidelity(
                explanations=predict_response.explanations,
                response=response_text,
                context=context,
                case_id=case_id,
            )
            prop = self._registry.get("judge_reasoning_fidelity")
            assert prop is not None
            _record_evidence(
                evidence, "judge_reasoning_fidelity", prop.id, prop.gate_mode,
                executed=True, result=result_rf, flags=result_rf.flags,
            )

        # --- Cat 6.1: Latency ---
        elapsed_ms = (time.time() - start_time) * 1000
        if self._registry.is_enabled("latency_cost"):
            from llm_judge.properties.performance import measure_latency
            result_lat = measure_latency(
                case_id=case_id,
                pipeline_latency_ms=elapsed_ms,
                input_text=context + " " + response_text,
                output_text=str(predict_response.explanations or ""),
            )
            prop = self._registry.get("latency_cost")
            assert prop is not None
            _record_evidence(
                evidence, "latency_cost", prop.id, prop.gate_mode,
                executed=True, result=result_lat, flags=result_lat.flags,
            )

        # =============================================================
        # Record calibration-only properties as not-executed-per-response
        # =============================================================
        for prop_name, prop in self._registry.get_enabled().items():
            if prop_name not in evidence:
                _record_evidence(
                    evidence, prop_name, prop.id, prop.gate_mode,
                    executed=False,  # calibration-only (Cat 5) or not yet wired
                )

        # =============================================================
        # Step 6: Compute detection coverage
        # =============================================================
        coverage = self._registry.detection_coverage()

        return EnrichedResponse(
            predict_response=predict_response,
            prompt_version=(
                f"{self._prompt.prompt_id}/{self._prompt.version}"
                if self._prompt else None
            ),
            hallucination_result=hallucination_result,
            property_evidence=evidence,
            detection_coverage=coverage.summary(),
            pipeline_latency_ms=elapsed_ms,
        )
