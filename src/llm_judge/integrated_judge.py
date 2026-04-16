"""
Integrated Evaluation Pipeline — all 28 properties wired.

Enable any property in property_config.yaml and it runs automatically.
RAG context retrieval enriches the evaluation context for grounding.
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
from llm_judge.calibration.pipeline_config import PipelineConfig, get_pipeline_config
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
    retrieval_evidence: Any = None  # RetrievalEvidence when retrieval runs
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
        all_f = list(self.predict_response.flags)
        if self.hallucination_result:
            all_f.extend(self.hallucination_result.flags)
        for ev_item in self.property_evidence.values():
            all_f.extend(ev_item.flags)
        return all_f

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
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
            out["hallucination"] = {
                "risk_score": self.hallucination_result.risk_score,
                "grounding_ratio": self.hallucination_result.grounding_ratio,
                "ungrounded_claims": self.hallucination_result.ungrounded_claims,
                "unverifiable_citations": self.hallucination_result.unverifiable_citations,
                "flags": self.hallucination_result.flags,
            }
        out["property_evidence"] = {
            name: {
                "id": ev_item.property_id,
                "enabled": ev_item.enabled,
                "gate_mode": ev_item.gate_mode,
                "executed": ev_item.executed,
                "flags": ev_item.flags,
                "error": ev_item.error,
            }
            for name, ev_item in self.property_evidence.items()
        }
        if self.retrieval_evidence is not None:
            out["retrieval"] = {
                "method": self.retrieval_evidence.method,
                "docs_retrieved": self.retrieval_evidence.docs_retrieved,
                "top_score": self.retrieval_evidence.top_score,
                "doc_ids": self.retrieval_evidence.doc_ids,
                "error": self.retrieval_evidence.error,
            }
        return out


def _build_context(
    request: PredictRequest, source_docs: list[str] | None = None
) -> str:
    """Build evaluation context from conversation and source documents."""
    conversation_context = " ".join(msg.content for msg in request.conversation)
    if source_docs:
        doc_text = "\n\n--- Source Documentation ---\n".join(source_docs)
        return f"{conversation_context}\n\n--- Source Documentation ---\n{doc_text}"
    if request.source_context:
        doc_text = "\n\n--- Source Documentation ---\n".join(request.source_context)
        return f"{conversation_context}\n\n--- Source Documentation ---\n{doc_text}"
    return conversation_context


def _build_query(request: PredictRequest) -> str:
    for msg in reversed(request.conversation):
        if msg.role == "user":
            return msg.content
    return request.conversation[0].content if request.conversation else ""


def _prop_id_gm(reg: PropertyRegistry, name: str) -> tuple[str, str]:
    """Get (id, gate_mode) for a property known to be enabled."""
    p = reg.get(name)
    if p is None:
        return name, "informational"
    return p.id, p.gate_mode


def _threshold_f(reg: PropertyRegistry, name: str, default: float) -> float:
    p = reg.get(name)
    if p is not None and p.threshold is not None:
        return float(p.threshold)
    return default


def _threshold_i(reg: PropertyRegistry, name: str, default: int) -> int:
    p = reg.get(name)
    if p is not None and p.threshold is not None:
        return int(p.threshold)
    return default


def _ev(
    evidence: dict[str, PropertyEvidence],
    name: str,
    pid: str,
    gm: str,
    executed: bool,
    result: Any = None,
    flags: list[str] | None = None,
    error: str | None = None,
) -> None:
    evidence[name] = PropertyEvidence(
        property_name=name,
        property_id=pid,
        enabled=True,
        gate_mode=gm,
        executed=executed,
        result=result,
        flags=flags or [],
        error=error,
    )


class IntegratedJudge(JudgeEngine):
    """Property-aware evaluation pipeline for Gate 2."""

    def __init__(
        self,
        engine: str = "gemini",
        timeout_ms: int | None = None,
        registry: PropertyRegistry | None = None,
        context_retriever: Any | None = None,
        pipeline_config: PipelineConfig | None = None,
    ) -> None:
        self._engine = engine
        self._timeout_ms = timeout_ms
        self._registry = registry
        self._context_retriever = context_retriever
        self._pipeline_config = pipeline_config
        self._llm_judge: LLMJudge | None = None
        self._prompt: PromptTemplate | None = None

    def _ensure_initialized(self) -> None:
        if self._registry is None:
            try:
                self._registry = load_property_config()
            except FileNotFoundError:
                self._registry = PropertyRegistry({})
        if self._prompt is None:
            try:
                self._prompt = load_latest_prompt("chat_quality")
            except FileNotFoundError:
                self._prompt = None
        if self._llm_judge is None:
            self._llm_judge = LLMJudge(
                engine=self._engine,
                timeout_ms=self._timeout_ms,
                prompt_template=self._prompt,
            )
        if self._context_retriever is None:
            try:
                from llm_judge.retrieval.context_retriever import (
                    ContextRetriever,
                    load_retrieval_config,
                )
                from llm_judge.retrieval.knowledge_base import load_knowledge_base

                config = load_retrieval_config()
                if config.enabled:
                    kb = load_knowledge_base()
                    if kb.is_loaded:
                        self._context_retriever = ContextRetriever(
                            vector_store=kb.store,
                            config=config,
                        )
                        logger.info(
                            "retriever.initialized",
                            extra={"docs": kb.document_count, "method": config.method},
                        )
            except Exception as exc:
                logger.info("retriever.not_available", extra={"reason": str(exc)[:80]})
        if self._pipeline_config is None:
            self._pipeline_config = get_pipeline_config()

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        enriched = self.evaluate_enriched(request)
        return PredictResponse(
            decision=enriched.predict_response.decision,
            overall_score=enriched.predict_response.overall_score,
            scores=enriched.predict_response.scores,
            confidence=enriched.predict_response.confidence,
            flags=enriched.all_flags(),
            explanations=enriched.predict_response.explanations,
        )

    def evaluate_enriched(
        self,
        request: PredictRequest,
        case_id: str = "unknown",
    ) -> EnrichedResponse:
        start_time = time.time()
        self._ensure_initialized()
        assert self._registry is not None
        reg = self._registry

        evidence: dict[str, PropertyEvidence] = {}
        query = _build_query(request)
        response_text = request.candidate_answer

        # =============================================================
        # RAG context retrieval (EPIC 7.14)
        # =============================================================
        # ALWAYS retrieve from the system's own knowledge base.
        # Caller-provided source_context is supplementary, not a replacement.
        # This ensures groundedness is judged against trusted KB content,
        # not against whatever the caller passes in.
        retrieved_docs: list[str] | None = None
        retrieval_evidence = None

        if self._context_retriever is not None:
            try:
                retrieved_docs, retrieval_evidence = self._context_retriever.retrieve(
                    query
                )
            except Exception as exc:
                logger.warning(
                    "retrieval.failed",
                    extra={"error": str(exc)[:80]},
                )

        # Merge caller-provided context as supplementary
        if request.source_context:
            supplementary = list(request.source_context)
            if retrieved_docs:
                retrieved_docs = retrieved_docs + supplementary
            else:
                retrieved_docs = supplementary

        context = _build_context(request, source_docs=retrieved_docs)

        # =============================================================
        # Deterministic pre-checks
        # =============================================================

        # --- Cat 1: Faithfulness 1.1-1.3 ---
        hallucination_result: HallucinationResult | None = None
        basic_faith = {
            k: v
            for k, v in reg.get_enabled_by_category("faithfulness").items()
            if k in ("groundedness", "ungrounded_claims", "citation_verification")
        }
        if basic_faith:
            if len(context.strip()) == 0:
                for pn, pc in basic_faith.items():
                    _ev(
                        evidence,
                        pn,
                        pc.id,
                        pc.gate_mode,
                        executed=False,
                        error="empty_context",
                        flags=["pipeline_error:empty_context"],
                    )
            else:
                gt = _threshold_f(reg, "groundedness", 0.3)
                mc = _threshold_i(reg, "ungrounded_claims", 2)
                hallucination_result = check_hallucination(
                    response=response_text,
                    context=context,
                    case_id=case_id,
                    grounding_threshold=gt,
                    max_ungrounded_claims=mc,
                    config=self._pipeline_config,
                )
                for pn, pc in basic_faith.items():
                    pf: list[str] = []
                    if (
                        pn == "groundedness"
                        and hallucination_result.grounding_ratio < gt
                    ):
                        pf.append(
                            f"low_grounding:{hallucination_result.grounding_ratio:.2f}"
                        )
                    elif (
                        pn == "ungrounded_claims"
                        and hallucination_result.ungrounded_claims > mc
                    ):
                        pf.append(
                            f"ungrounded_claims:{hallucination_result.ungrounded_claims}"
                        )
                    elif (
                        pn == "citation_verification"
                        and hallucination_result.unverifiable_citations > 0
                    ):
                        pf.append(
                            f"unverifiable_citations:{hallucination_result.unverifiable_citations}"
                        )
                    _ev(
                        evidence,
                        pn,
                        pc.id,
                        pc.gate_mode,
                        executed=True,
                        result=hallucination_result,
                        flags=pf,
                    )

        # --- Cat 3: Safety 3.1-3.3 ---
        if reg.get_enabled_by_category("safety"):
            from llm_judge.properties.safety import (
                check_instruction_boundary,
                check_pii_leakage,
                check_toxicity,
            )

            if reg.is_enabled("toxicity_bias"):
                tox_result = check_toxicity(response=response_text, case_id=case_id)
                pid, gm = _prop_id_gm(reg, "toxicity_bias")
                _ev(
                    evidence,
                    "toxicity_bias",
                    pid,
                    gm,
                    executed=True,
                    result=tox_result,
                    flags=tox_result.flags,
                )

            if reg.is_enabled("instruction_boundary"):
                ib_result = check_instruction_boundary(
                    response=response_text, case_id=case_id
                )
                pid, gm = _prop_id_gm(reg, "instruction_boundary")
                _ev(
                    evidence,
                    "instruction_boundary",
                    pid,
                    gm,
                    executed=True,
                    result=ib_result,
                    flags=ib_result.flags,
                )

            if reg.is_enabled("pii_data_leakage"):
                pii_result = check_pii_leakage(response=response_text, case_id=case_id)
                pid, gm = _prop_id_gm(reg, "pii_data_leakage")
                _ev(
                    evidence,
                    "pii_data_leakage",
                    pid,
                    gm,
                    executed=True,
                    result=pii_result,
                    flags=pii_result.flags,
                )

        # --- Cat 4: Task Fidelity 4.1-4.2 ---
        if reg.is_enabled("instruction_following"):
            from llm_judge.properties.task_fidelity import check_instruction_following

            if_result = check_instruction_following(
                query=query, response=response_text, case_id=case_id
            )
            pid, gm = _prop_id_gm(reg, "instruction_following")
            _ev(
                evidence,
                "instruction_following",
                pid,
                gm,
                executed=True,
                result=if_result,
                flags=if_result.flags,
            )

        if reg.is_enabled("format_structure"):
            from llm_judge.properties.task_fidelity import check_format_structure

            fs_result = check_format_structure(response=response_text, case_id=case_id)
            pid, gm = _prop_id_gm(reg, "format_structure")
            _ev(
                evidence,
                "format_structure",
                pid,
                gm,
                executed=True,
                result=fs_result,
                flags=fs_result.flags,
            )

        # =============================================================
        # LLM evaluation with versioned prompt (Cat 2)
        # =============================================================
        assert self._llm_judge is not None
        predict_response = self._llm_judge.evaluate(request)

        for pn, pc in reg.get_enabled_by_category("semantic_quality").items():
            score = predict_response.scores.get(pn)
            sf: list[str] = []
            if score is not None and score <= 2:
                sf.append(f"low_{pn}:{score}")
            _ev(
                evidence, pn, pc.id, pc.gate_mode, executed=True, result=score, flags=sf
            )

        # =============================================================
        # Post-eval checks
        # =============================================================

        # --- Cat 1 Advanced: 1.4-1.5 ---
        if reg.is_enabled("attribution_accuracy") and context.strip():
            try:
                from llm_judge.properties.faithfulness_advanced import (
                    check_attribution_accuracy,
                )

                aa_result = check_attribution_accuracy(
                    response=response_text, context=context, case_id=case_id
                )
                pid, gm = _prop_id_gm(reg, "attribution_accuracy")
                _ev(
                    evidence,
                    "attribution_accuracy",
                    pid,
                    gm,
                    executed=True,
                    result=aa_result,
                    flags=aa_result.flags,
                )
            except Exception as exc:
                pid, gm = _prop_id_gm(reg, "attribution_accuracy")
                _ev(
                    evidence,
                    "attribution_accuracy",
                    pid,
                    gm,
                    executed=False,
                    error=str(exc)[:80],
                )

        if reg.is_enabled("fabrication_detection") and context.strip():
            try:
                from llm_judge.properties.faithfulness_advanced import check_fabrication

                fab_result = check_fabrication(
                    response=response_text, context=context, case_id=case_id
                )
                pid, gm = _prop_id_gm(reg, "fabrication_detection")
                _ev(
                    evidence,
                    "fabrication_detection",
                    pid,
                    gm,
                    executed=True,
                    result=fab_result,
                    flags=fab_result.flags,
                )
            except Exception as exc:
                pid, gm = _prop_id_gm(reg, "fabrication_detection")
                _ev(
                    evidence,
                    "fabrication_detection",
                    pid,
                    gm,
                    executed=False,
                    error=str(exc)[:80],
                )

        # --- Cat 6: Performance 6.3, 6.4 ---
        if reg.is_enabled("explainability"):
            from llm_judge.properties.performance import check_explainability

            dims = (
                self._prompt.dimensions
                if self._prompt
                else ["relevance", "clarity", "correctness", "tone"]
            )
            ex_result = check_explainability(
                explanations=predict_response.explanations,
                expected_dimensions=dims,
                case_id=case_id,
            )
            pid, gm = _prop_id_gm(reg, "explainability")
            _ev(
                evidence,
                "explainability",
                pid,
                gm,
                executed=True,
                result=ex_result,
                flags=ex_result.flags,
            )

        if reg.is_enabled("judge_reasoning_fidelity"):
            from llm_judge.properties.performance import check_reasoning_fidelity

            rf_result = check_reasoning_fidelity(
                explanations=predict_response.explanations,
                response=response_text,
                context=context,
                case_id=case_id,
            )
            pid, gm = _prop_id_gm(reg, "judge_reasoning_fidelity")
            _ev(
                evidence,
                "judge_reasoning_fidelity",
                pid,
                gm,
                executed=True,
                result=rf_result,
                flags=rf_result.flags,
            )

        # --- Cat 6.1: Latency ---
        elapsed_ms = (time.time() - start_time) * 1000
        if reg.is_enabled("latency_cost"):
            from llm_judge.properties.performance import measure_latency

            lat_result = measure_latency(
                case_id=case_id,
                pipeline_latency_ms=elapsed_ms,
                input_text=context + " " + response_text,
                output_text=str(predict_response.explanations or ""),
            )
            pid, gm = _prop_id_gm(reg, "latency_cost")
            _ev(
                evidence,
                "latency_cost",
                pid,
                gm,
                executed=True,
                result=lat_result,
                flags=lat_result.flags,
            )

        # =============================================================
        # Calibration-only properties (Cat 5) — not per-response
        # =============================================================
        for pn, pc in reg.get_enabled().items():
            if pn not in evidence:
                _ev(evidence, pn, pc.id, pc.gate_mode, executed=False)

        coverage = reg.detection_coverage()

        return EnrichedResponse(
            predict_response=predict_response,
            prompt_version=(
                f"{self._prompt.prompt_id}/{self._prompt.version}"
                if self._prompt
                else None
            ),
            hallucination_result=hallucination_result,
            property_evidence=evidence,
            retrieval_evidence=retrieval_evidence,
            detection_coverage=coverage.summary(),
            pipeline_latency_ms=elapsed_ms,
        )
