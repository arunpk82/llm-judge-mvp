"""
Hallucination Detection (L4 Step 6).

Detects responses where the LLM claims facts without source grounding.
Uses text-based similarity metrics with a pluggable interface for
embedding-based distance when an embedding API is available.

Detection strategies:
  - Token overlap: measures factual grounding via shared tokens
  - Ungrounded claims: detects assertion patterns without source support
  - Citation verification: flags claimed citations that don't exist in context
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Patterns that indicate factual claims in responses
_CLAIM_PATTERNS = [
    re.compile(r"(?:according to|studies show|research indicates|data shows|statistics show)", re.I),
    re.compile(r"(?:it is (?:known|proven|established) that)", re.I),
    re.compile(r"(?:the (?:official|published|documented) (?:number|figure|rate|data))", re.I),
    re.compile(r"\d{1,3}(?:\.\d+)?%", re.I),  # percentage claims
]

# Patterns that indicate cited sources
_CITATION_PATTERNS = [
    re.compile(r"(?:according to [\w\s]+(?:,|\.))", re.I),
    re.compile(r"\[(?:\d+|[a-z]+)\]", re.I),  # [1], [a] style citations
    re.compile(r"\((?:[\w\s]+,?\s*\d{4})\)", re.I),  # (Author, 2024) style
]


@dataclass
class HallucinationResult:
    """Result of hallucination check for a single response."""
    case_id: str
    risk_score: float  # 0.0 (grounded) to 1.0 (likely hallucinated)
    grounding_ratio: float  # token overlap with context
    ungrounded_claims: int  # claims without context support
    unverifiable_citations: int  # citations not found in context
    flags: list[str] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer."""
    return {
        w.lower().strip(".,!?;:\"'()[]{}") 
        for w in text.split() 
        if len(w.strip(".,!?;:\"'()[]{}")) > 2
    }


def _compute_grounding_ratio(response: str, context: str) -> float:
    """
    Compute token overlap between response and context.

    Higher ratio = more grounded in the provided context.
    Lower ratio = response introduces many tokens not in context.
    """
    response_tokens = _tokenize(response)
    context_tokens = _tokenize(context)

    if not response_tokens:
        return 1.0  # empty response is "grounded"

    overlap = response_tokens & context_tokens
    return len(overlap) / len(response_tokens)


def _count_ungrounded_claims(response: str, context: str) -> tuple[int, list[str]]:
    """
    Count factual claims in the response that lack support in the context.

    A claim is "ungrounded" if it uses a claim pattern but the surrounding
    words don't appear in the context.
    """
    claims: list[str] = []

    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(response):
            # Get surrounding context (±30 chars)
            start = max(0, match.start() - 30)
            end = min(len(response), match.end() + 30)
            claim_context = response[start:end].strip()

            # Check if the key terms appear in the source context
            claim_tokens = _tokenize(claim_context)
            context_tokens = _tokenize(context)
            overlap = claim_tokens & context_tokens

            if len(overlap) < len(claim_tokens) * 0.3:
                claims.append(claim_context[:80])

    return len(claims), claims


def _count_unverifiable_citations(response: str, context: str) -> int:
    """Count citations in the response that don't correspond to context sources."""
    count = 0
    context_lower = context.lower()

    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(response):
            cited_text = match.group(0).lower()
            # Check if the cited source appears anywhere in the context
            if cited_text not in context_lower:
                # Also check without brackets/parens
                cleaned = re.sub(r"[^\w\s]", "", cited_text).strip()
                if cleaned and cleaned not in context_lower:
                    count += 1

    return count


def check_hallucination(
    *,
    response: str,
    context: str,
    case_id: str = "unknown",
    grounding_threshold: float = 0.3,
    max_ungrounded_claims: int = 2,
) -> HallucinationResult:
    """
    Check a single response for hallucination risk.

    Args:
        response: The candidate answer to check
        context: The source context (conversation + any reference material)
        case_id: Identifier for this case
        grounding_threshold: Below this token overlap ratio → flagged
        max_ungrounded_claims: Above this count → flagged
    """
    grounding = _compute_grounding_ratio(response, context)
    ungrounded_count, ungrounded_examples = _count_ungrounded_claims(response, context)
    citation_count = _count_unverifiable_citations(response, context)

    flags: list[str] = []

    if grounding < grounding_threshold:
        flags.append(f"low_grounding:{grounding:.2f}")

    if ungrounded_count > max_ungrounded_claims:
        flags.append(f"ungrounded_claims:{ungrounded_count}")

    if citation_count > 0:
        flags.append(f"unverifiable_citations:{citation_count}")

    # Compute composite risk score (0-1)
    risk = 0.0
    risk += max(0, (grounding_threshold - grounding) / grounding_threshold) * 0.4
    risk += min(1.0, ungrounded_count / max(1, max_ungrounded_claims + 2)) * 0.4
    risk += min(1.0, citation_count / 3) * 0.2
    risk = min(1.0, risk)

    return HallucinationResult(
        case_id=case_id,
        risk_score=round(risk, 4),
        grounding_ratio=round(grounding, 4),
        ungrounded_claims=ungrounded_count,
        unverifiable_citations=citation_count,
        flags=flags,
    )


def check_hallucinations_batch(
    *,
    cases: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    grounding_threshold: float = 0.3,
) -> dict[str, Any]:
    """
    Run hallucination checks across a batch of evaluated cases.

    Returns summary statistics and flagged cases.
    """
    results: list[HallucinationResult] = []
    case_map = {str(c.get("case_id", i)): c for i, c in enumerate(cases)}

    for j in judgments:
        case_id = str(j.get("case_id", ""))
        case = case_map.get(case_id, {})

        # Build context from conversation
        conversation = case.get("conversation", [])
        context_parts = [
            msg.get("content", "") 
            for msg in conversation 
            if isinstance(msg, dict)
        ]
        context = " ".join(context_parts)
        response = str(case.get("candidate_answer", ""))

        if not response or not context:
            continue

        result = check_hallucination(
            response=response,
            context=context,
            case_id=case_id,
            grounding_threshold=grounding_threshold,
        )
        results.append(result)

    flagged = [r for r in results if r.flags]
    avg_grounding = (
        sum(r.grounding_ratio for r in results) / len(results) if results else 0
    )
    avg_risk = (
        sum(r.risk_score for r in results) / len(results) if results else 0
    )

    return {
        "total_checked": len(results),
        "flagged": len(flagged),
        "flagged_rate": round(len(flagged) / len(results), 4) if results else 0,
        "avg_grounding_ratio": round(avg_grounding, 4),
        "avg_risk_score": round(avg_risk, 4),
        "flagged_cases": [
            {
                "case_id": r.case_id,
                "risk_score": r.risk_score,
                "grounding_ratio": r.grounding_ratio,
                "flags": r.flags,
            }
            for r in flagged[:20]
        ],
    }
