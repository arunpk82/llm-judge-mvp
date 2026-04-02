"""
Performance Properties (Category 6, Properties 6.1, 6.3, 6.4).

6.1 Latency & Cost — wall-clock timing, token estimation
6.3 Explainability — are judge explanations specific and verifiable?
6.4 Judge Reasoning Fidelity — are explanations grounded in the actual response?
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =====================================================================
# 6.1 Latency & Cost
# =====================================================================

@dataclass
class LatencyResult:
    """Result of latency & cost measurement."""
    case_id: str
    pipeline_latency_ms: float
    estimated_input_tokens: int
    estimated_output_tokens: int
    flags: list[str] = field(default_factory=list)


def measure_latency(
    *,
    case_id: str = "unknown",
    pipeline_latency_ms: float = 0.0,
    input_text: str = "",
    output_text: str = "",
    latency_threshold_ms: float = 5000.0,
) -> LatencyResult:
    """Record latency and estimate token usage (Property 6.1)."""
    # Rough token estimation: ~4 chars per token
    input_tokens = len(input_text) // 4
    output_tokens = len(output_text) // 4

    flags: list[str] = []
    if pipeline_latency_ms > latency_threshold_ms:
        flags.append(f"high_latency:{pipeline_latency_ms:.0f}ms")

    return LatencyResult(
        case_id=case_id,
        pipeline_latency_ms=round(pipeline_latency_ms, 1),
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=output_tokens,
        flags=flags,
    )


# =====================================================================
# 6.3 Explainability
# =====================================================================

@dataclass
class ExplainabilityResult:
    """Result of explainability check."""
    case_id: str
    dimensions_explained: int
    dimensions_expected: int
    empty_explanations: list[str] = field(default_factory=list)
    vague_explanations: list[str] = field(default_factory=list)
    explainability_score: float = 1.0
    flags: list[str] = field(default_factory=list)


_VAGUE_PATTERNS = [
    re.compile(r"^(?:good|bad|ok|fine|adequate|acceptable|reasonable)\b", re.I),
    re.compile(r"^(?:the response is|this is|it is)\s+(?:good|bad|ok|fine)", re.I),
    re.compile(r"^(?:n/?a|none|no comment|see above)\s*$", re.I),
]


def check_explainability(
    *,
    explanations: dict[str, str] | None,
    expected_dimensions: list[str] | None = None,
    case_id: str = "unknown",
    min_explanation_length: int = 20,
) -> ExplainabilityResult:
    """
    Check if judge explanations are specific and verifiable (Property 6.3).

    Checks: are all dimensions explained? Are explanations non-empty?
    Are they specific (not just "good" or "ok")? Are they long enough
    to be verifiable?
    """
    if expected_dimensions is None:
        expected_dimensions = ["relevance", "clarity", "correctness", "tone"]

    if explanations is None:
        return ExplainabilityResult(
            case_id=case_id,
            dimensions_explained=0,
            dimensions_expected=len(expected_dimensions),
            empty_explanations=expected_dimensions,
            explainability_score=0.0,
            flags=["no_explanations_provided"],
        )

    empty: list[str] = []
    vague: list[str] = []
    explained = 0

    for dim in expected_dimensions:
        expl = explanations.get(dim, "")
        if not expl or len(expl.strip()) == 0:
            empty.append(dim)
            continue

        explained += 1

        if len(expl.strip()) < min_explanation_length:
            vague.append(dim)
            continue

        for pattern in _VAGUE_PATTERNS:
            if pattern.match(expl.strip()):
                vague.append(dim)
                break

    total = len(expected_dimensions)
    quality_count = explained - len(vague)
    score = quality_count / total if total > 0 else 0.0

    flags: list[str] = []
    if empty:
        flags.append(f"missing_explanations:{','.join(empty)}")
    if vague:
        flags.append(f"vague_explanations:{','.join(vague)}")

    return ExplainabilityResult(
        case_id=case_id,
        dimensions_explained=explained,
        dimensions_expected=total,
        empty_explanations=empty,
        vague_explanations=vague,
        explainability_score=round(score, 2),
        flags=flags,
    )


# =====================================================================
# 6.4 Judge Reasoning Fidelity
# =====================================================================

@dataclass
class ReasoningFidelityResult:
    """Result of judge reasoning fidelity check."""
    case_id: str
    dimensions_checked: int
    grounded_explanations: int
    fabricated_explanations: int
    fidelity_score: float
    fabricated_details: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def check_reasoning_fidelity(
    *,
    explanations: dict[str, str] | None,
    response: str,
    context: str,
    case_id: str = "unknown",
    min_overlap_ratio: float = 0.2,
) -> ReasoningFidelityResult:
    """
    Check if judge explanations reference content actually in the response (Property 6.4).

    For each explanation, checks whether the key terms appear in the
    actual response or context. If the explanation references things
    not in the response, the judge may be hallucinating its reasoning.

    Note: Full verification requires a second judge (verification judge).
    This deterministic check catches obvious fabrication; the verification
    judge handles subtle cases.
    """
    if explanations is None:
        return ReasoningFidelityResult(
            case_id=case_id,
            dimensions_checked=0,
            grounded_explanations=0,
            fabricated_explanations=0,
            fidelity_score=1.0,
        )

    combined_source = (response + " " + context).lower()
    source_tokens = {
        w.strip(".,!?;:\"'()[]{}") for w in combined_source.split()
        if len(w.strip(".,!?;:\"'()[]{}")) > 3
    }

    grounded = 0
    fabricated = 0
    fabricated_details: list[str] = []
    flags: list[str] = []

    for dim, expl in explanations.items():
        if not expl or len(expl.strip()) < 10:
            continue

        expl_tokens = {
            w.lower().strip(".,!?;:\"'()[]{}") for w in expl.split()
            if len(w.strip(".,!?;:\"'()[]{}")) > 3
        }

        if not expl_tokens:
            continue

        # Filter out common evaluation words that don't need grounding
        eval_words = {
            "response", "answer", "question", "score", "rating", "evaluation",
            "dimension", "quality", "provides", "addresses", "clear", "unclear",
            "relevant", "irrelevant", "correct", "incorrect", "tone", "clarity",
            "overall", "candidate", "user", "appropriate", "inappropriate",
        }
        content_tokens = expl_tokens - eval_words

        if not content_tokens:
            grounded += 1  # explanation uses only evaluation vocabulary
            continue

        overlap = content_tokens & source_tokens
        ratio = len(overlap) / len(content_tokens) if content_tokens else 1.0

        if ratio >= min_overlap_ratio:
            grounded += 1
        else:
            fabricated += 1
            fabricated_details.append(f"{dim}: overlap={ratio:.2f}")

    total = grounded + fabricated
    fidelity = grounded / total if total > 0 else 1.0

    if fabricated > 0:
        flags.append(f"fabricated_reasoning:{fabricated}")

    return ReasoningFidelityResult(
        case_id=case_id,
        dimensions_checked=total,
        grounded_explanations=grounded,
        fabricated_explanations=fabricated,
        fidelity_score=round(fidelity, 2),
        fabricated_details=fabricated_details,
        flags=flags,
    )
