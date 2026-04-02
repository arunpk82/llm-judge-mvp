"""
Integrated Evaluation Pipeline (EPICs 7.4–7.7, PCT-1).

Replaces the raw LLMJudge code path with a property-aware pipeline:
  1. Load property configuration
  2. Load versioned prompt (replaces hardcoded _SYSTEM_PROMPT)
  3. Run deterministic pre-checks (hallucination 1.1–1.3)
  4. Execute LLM evaluation with versioned prompt
  5. Apply gate mode per property
  6. Assemble enriched response with property execution evidence

Every Gate 2 evaluation produces evidence that all enabled properties
executed — flags, scores, version IDs, confidence scores.
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
    """
    Gate 2 response enriched with property execution evidence.

    Extends PredictResponse with hallucination results, prompt version,
    property evidence, and detection coverage.
    """
    # Core evaluation (from LLMJudge)
    predict_response: PredictResponse

    # Property evidence
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
    """Build context string from conversation for hallucination checks."""
    parts = [msg.content for msg in request.conversation]
    return " ".join(parts)


class IntegratedJudge(JudgeEngine):
    """
    Property-aware evaluation pipeline for Gate 2.

    Replaces the raw LLMJudge usage with:
    1. Property configuration loading
    2. Versioned prompt loading (EPIC 7.5)
    3. Deterministic pre-checks — hallucination (EPIC 7.6)
    4. LLM evaluation with versioned prompt
    5. Gate mode application per property
    6. Enriched response assembly

    The evaluate() method returns a standard PredictResponse for
    backward compatibility. Use evaluate_enriched() for the full
    property evidence.
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
        """
        Standard JudgeEngine interface — returns PredictResponse.

        For backward compatibility with existing pipeline.
        Internally runs the full integrated pipeline.
        """
        enriched = self.evaluate_enriched(request)
        # Merge all flags into the PredictResponse
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
        """
        Full integrated evaluation with property evidence.

        Steps:
        1. Load property config (once)
        2. Load versioned prompt (once)
        3. Run deterministic pre-checks (hallucination 1.1–1.3)
        4. Execute LLM evaluation with versioned prompt
        5. Assemble enriched response with evidence
        """
        start_time = time.time()
        self._ensure_initialized()
        assert self._registry is not None

        evidence: dict[str, PropertyEvidence] = {}

        # --- Step 3: Deterministic pre-checks (hallucination 1.1–1.3) ---
        hallucination_result: HallucinationResult | None = None

        faithfulness_props = self._registry.get_enabled_by_category("faithfulness")
        if faithfulness_props:
            context = _build_context(request)
            response_text = request.candidate_answer

            # Guard: context must be longer than just the candidate answer
            if len(context.strip()) == 0:
                logger.warning(
                    "hallucination.empty_context",
                    extra={"case_id": case_id},
                )
                for prop_name, prop in faithfulness_props.items():
                    evidence[prop_name] = PropertyEvidence(
                        property_name=prop_name,
                        property_id=prop.id,
                        enabled=True,
                        gate_mode=prop.gate_mode,
                        executed=False,
                        error="empty_context",
                        flags=["pipeline_error:empty_context"],
                    )
            else:
                grounding_threshold = 0.3
                max_claims = 2
                gt_prop = self._registry.get("groundedness")
                if gt_prop and gt_prop.threshold is not None:
                    grounding_threshold = float(gt_prop.threshold)
                uc_prop = self._registry.get("ungrounded_claims")
                if uc_prop and uc_prop.threshold is not None:
                    max_claims = int(uc_prop.threshold)

                hallucination_result = check_hallucination(
                    response=response_text,
                    context=context,
                    case_id=case_id,
                    grounding_threshold=grounding_threshold,
                    max_ungrounded_claims=max_claims,
                )

                # Record evidence for each faithfulness property
                for prop_name, prop in faithfulness_props.items():
                    prop_flags: list[str] = []
                    if prop_name == "groundedness" and hallucination_result:
                        if hallucination_result.grounding_ratio < grounding_threshold:
                            prop_flags.append(
                                f"low_grounding:{hallucination_result.grounding_ratio:.2f}"
                            )
                    elif prop_name == "ungrounded_claims" and hallucination_result:
                        if hallucination_result.ungrounded_claims > max_claims:
                            prop_flags.append(
                                f"ungrounded_claims:{hallucination_result.ungrounded_claims}"
                            )
                    elif prop_name == "citation_verification" and hallucination_result:
                        if hallucination_result.unverifiable_citations > 0:
                            prop_flags.append(
                                f"unverifiable_citations:{hallucination_result.unverifiable_citations}"
                            )

                    evidence[prop_name] = PropertyEvidence(
                        property_name=prop_name,
                        property_id=prop.id,
                        enabled=True,
                        gate_mode=prop.gate_mode,
                        executed=True,
                        result=hallucination_result,
                        flags=prop_flags,
                    )

        # --- Step 4: LLM evaluation with versioned prompt ---
        assert self._llm_judge is not None
        predict_response = self._llm_judge.evaluate(request)

        # Record evidence for semantic quality properties
        semantic_props = self._registry.get_enabled_by_category("semantic_quality")
        for prop_name, prop in semantic_props.items():
            dim_name = prop_name  # property name matches dimension name
            score = predict_response.scores.get(dim_name)
            prop_flags_sem: list[str] = []
            if score is not None and score <= 2:
                prop_flags_sem.append(f"low_{dim_name}:{score}")

            evidence[prop_name] = PropertyEvidence(
                property_name=prop_name,
                property_id=prop.id,
                enabled=True,
                gate_mode=prop.gate_mode,
                executed=True,
                result=score,
                flags=prop_flags_sem,
            )

        # Record evidence for enabled but non-executing properties
        # (robustness/performance properties that run in calibration, not per-response)
        for prop_name, prop in self._registry.get_enabled().items():
            if prop_name not in evidence:
                evidence[prop_name] = PropertyEvidence(
                    property_name=prop_name,
                    property_id=prop.id,
                    enabled=True,
                    gate_mode=prop.gate_mode,
                    executed=False,  # calibration-only properties
                    flags=[],
                )

        # --- Step 5: Compute detection coverage ---
        coverage = self._registry.detection_coverage()

        elapsed_ms = (time.time() - start_time) * 1000

        return EnrichedResponse(
            predict_response=predict_response,
            prompt_version=(
                f"{self._prompt.prompt_id}/{self._prompt.version}"
                if self._prompt
                else None
            ),
            hallucination_result=hallucination_result,
            property_evidence=evidence,
            detection_coverage=coverage.summary(),
            pipeline_latency_ms=elapsed_ms,
        )
