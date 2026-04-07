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
    grounding_ratio: float  # proportion of grounded sentences
    min_sentence_sim: float  # lowest sentence similarity (catches single fabrications)
    ungrounded_claims: int  # claims without context support
    unverifiable_citations: int  # citations not found in context
    flags: list[str] = field(default_factory=list)


def _tokenize(text: str) -> set[str]:
    """Simple whitespace + punctuation tokenizer (used by 1.2 ungrounded claims)."""
    return {
        w.lower().strip(".,!?;:\"'()[]{}") 
        for w in text.split() 
        if len(w.strip(".,!?;:\"'()[]{}")) > 2
    }


# Sentence splitter — same as faithfulness_advanced.py
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering short fragments."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _compute_grounding_ratio(
    response: str,
    context: str,
    *,
    skip_embeddings: bool = False,
    similarity_threshold: float = 0.6,
) -> tuple[float, float]:
    """
    Compute grounding ratio and min sentence similarity using embeddings.

    For each response sentence, finds the max cosine similarity against
    all context sentences using MiniLM embeddings. A sentence is
    "grounded" if max_similarity >= similarity_threshold.

    Returns: (grounding_ratio, min_sentence_sim)
        - grounding_ratio: proportion of grounded sentences (0.0 to 1.0)
        - min_sentence_sim: lowest max-similarity across all sentences

    Falls back to token overlap if skip_embeddings=True or if
    sentence-transformers is not available.
    """
    response_sentences = _split_sentences(response)
    context_sentences = _split_sentences(context)

    if not response_sentences:
        return 1.0, 1.0  # empty response is "grounded"

    if not context_sentences:
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio

    if skip_embeddings:
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio

    try:
        from llm_judge.properties import get_embedding_provider

        provider = get_embedding_provider()
        response_embeddings = provider.encode(response_sentences)
        context_embeddings = provider.encode(context_sentences)

        grounded_count = 0
        sentence_sims: list[float] = []
        for resp_emb in response_embeddings:
            max_sim = provider.max_similarity(resp_emb, context_embeddings)
            sentence_sims.append(max_sim)
            if max_sim >= similarity_threshold:
                grounded_count += 1

        ratio = grounded_count / len(response_sentences)
        min_sim = min(sentence_sims) if sentence_sims else 1.0
        return ratio, min_sim

    except (RuntimeError, ImportError) as e:
        logger.warning(
            "grounding.embedding_fallback",
            extra={"error": str(e)[:80]},
        )
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio


def _compute_grounding_ratio_token_overlap(response: str, context: str) -> float:
    """
    Legacy token overlap method (bag-of-words).

    Kept as fallback when embeddings are unavailable.
    Known limitations: F1=0.000 on RAGTruth, fires 99.2% on CS domain.
    """
    response_tokens = _tokenize(response)
    context_tokens = _tokenize(context)

    if not response_tokens:
        return 1.0
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
    grounding_threshold: float = 0.8,
    min_sentence_threshold: float = 0.3,
    similarity_threshold: float = 0.6,
    max_ungrounded_claims: int = 2,
    skip_embeddings: bool = False,
) -> HallucinationResult:
    """
    Check a single response for hallucination risk.

    Uses dual threshold detection (Experiment 5):
      - Ratio check: flags if grounding_ratio < grounding_threshold
      - Min-sentence check: flags if any sentence sim < min_sentence_threshold
    Either condition triggers a flag. This catches both fully fabricated
    responses (ratio) and single fabricated claims (min-sentence).

    Args:
        response: The candidate answer to check
        context: The source context (conversation + KB documents)
        case_id: Identifier for this case
        grounding_threshold: Below this ratio → flagged (default 0.80)
        min_sentence_threshold: Any sentence below this → flagged (default 0.30)
        similarity_threshold: Per-sentence sim for grounding ratio (default 0.60)
        max_ungrounded_claims: Above this count → flagged
        skip_embeddings: If True, fall back to token overlap
    """
    grounding, min_sim = _compute_grounding_ratio(
        response, context,
        skip_embeddings=skip_embeddings,
        similarity_threshold=similarity_threshold,
    )
    ungrounded_count, ungrounded_examples = _count_ungrounded_claims(response, context)
    citation_count = _count_unverifiable_citations(response, context)

    flags: list[str] = []

    # Dual threshold: either ratio OR min-sentence triggers
    if grounding < grounding_threshold:
        flags.append(f"low_grounding:{grounding:.2f}")

    if min_sim < min_sentence_threshold:
        flags.append(f"low_min_sentence_sim:{min_sim:.3f}")

    if ungrounded_count > max_ungrounded_claims:
        flags.append(f"ungrounded_claims:{ungrounded_count}")

    if citation_count > 0:
        flags.append(f"unverifiable_citations:{citation_count}")

    # Compute composite risk score (0-1)
    risk = 0.0
    risk += max(0, (grounding_threshold - grounding) / grounding_threshold) * 0.3
    risk += max(0, (min_sentence_threshold - min_sim) / min_sentence_threshold) * 0.3
    risk += min(1.0, ungrounded_count / max(1, max_ungrounded_claims + 2)) * 0.25
    risk += min(1.0, citation_count / 3) * 0.15
    risk = min(1.0, risk)

    return HallucinationResult(
        case_id=case_id,
        risk_score=round(risk, 4),
        grounding_ratio=round(grounding, 4),
        min_sentence_sim=round(min_sim, 4),
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
