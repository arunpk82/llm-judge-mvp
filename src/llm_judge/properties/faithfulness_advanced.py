"""
Advanced Faithfulness Properties (Category 1, Properties 1.4–1.5).

1.4 Attribution Accuracy — does the cited source actually support the claim?
1.5 Fabrication Detection — plausible but entirely invented facts

Both use embedding distance for semantic analysis. Falls back to
token overlap when sentence-transformers is not installed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Sentence splitter — simple but effective for evaluation
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Citation patterns (from hallucination.py — reused)
_CITATION_PATTERNS = [
    re.compile(r"(?:according to ([\w\s]+)(?:,|\.))", re.I),
    re.compile(r"\[(\d+)\]"),
    re.compile(r"\(([\w\s]+,?\s*\d{4})\)", re.I),
]


@dataclass
class AttributionResult:
    """Result of attribution accuracy check (Property 1.4)."""
    case_id: str
    claims_checked: int
    claims_supported: int
    claims_contradicted: int
    claims_neutral: int
    max_similarity_scores: list[float] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.claims_checked == 0:
            return 1.0
        return self.claims_supported / self.claims_checked


@dataclass
class FabricationResult:
    """Result of fabrication detection (Property 1.5)."""
    case_id: str
    sentences_checked: int
    fabrication_suspects: int
    avg_max_similarity: float
    low_similarity_sentences: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def fabrication_rate(self) -> float:
        if self.sentences_checked == 0:
            return 0.0
        return self.fabrication_suspects / self.sentences_checked


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def check_attribution_accuracy(
    *,
    response: str,
    context: str,
    case_id: str = "unknown",
    support_threshold: float = 0.6,
    contradiction_threshold: float = 0.3,
    embedding_provider: object | None = None,
) -> AttributionResult:
    """
    Check if cited sources actually support the claims (Property 1.4).

    For each citation in the response, embeds the claim and the cited
    context, measures cosine similarity. High similarity = supported,
    low similarity = neutral or contradicted.
    """
    from llm_judge.properties import get_embedding_provider, EmbeddingProvider

    provider: EmbeddingProvider = (
        embedding_provider if isinstance(embedding_provider, EmbeddingProvider)
        else get_embedding_provider()
    )

    # Extract sentences with citations
    cited_sentences = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(response):
            start = max(0, match.start() - 100)
            end = min(len(response), match.end() + 100)
            sentence = response[start:end].strip()
            if sentence:
                cited_sentences.append(sentence)

    if not cited_sentences:
        return AttributionResult(
            case_id=case_id,
            claims_checked=0,
            claims_supported=0,
            claims_contradicted=0,
            claims_neutral=0,
        )

    context_sentences = _split_sentences(context)
    if not context_sentences:
        return AttributionResult(
            case_id=case_id,
            claims_checked=len(cited_sentences),
            claims_supported=0,
            claims_contradicted=0,
            claims_neutral=len(cited_sentences),
            flags=["no_context_sentences"],
        )

    # Embed everything
    all_texts = cited_sentences + context_sentences
    all_embeddings = provider.encode(all_texts)
    claim_embeddings = all_embeddings[:len(cited_sentences)]
    context_embeddings = all_embeddings[len(cited_sentences):]

    supported = 0
    contradicted = 0
    neutral = 0
    similarities: list[float] = []
    flags: list[str] = []

    for claim_emb in claim_embeddings:
        max_sim = provider.max_similarity(claim_emb, context_embeddings)
        similarities.append(max_sim)

        if max_sim >= support_threshold:
            supported += 1
        elif max_sim <= contradiction_threshold:
            contradicted += 1
        else:
            neutral += 1

    if contradicted > 0:
        flags.append(f"attribution_contradiction:{contradicted}")
    if neutral > len(cited_sentences) * 0.5:
        flags.append(f"weak_attribution:{neutral}")

    return AttributionResult(
        case_id=case_id,
        claims_checked=len(cited_sentences),
        claims_supported=supported,
        claims_contradicted=contradicted,
        claims_neutral=neutral,
        max_similarity_scores=similarities,
        flags=flags,
    )


def check_fabrication(
    *,
    response: str,
    context: str,
    case_id: str = "unknown",
    fabrication_threshold: float = 0.3,
    embedding_provider: object | None = None,
) -> FabricationResult:
    """
    Detect fabricated content in response (Property 1.5).

    For each sentence in the response, measures max cosine similarity
    against all context sentences. Sentences with very low similarity
    to all context = likely fabricated (introducing information from nowhere).
    """
    from llm_judge.properties import get_embedding_provider, EmbeddingProvider

    provider: EmbeddingProvider = (
        embedding_provider if isinstance(embedding_provider, EmbeddingProvider)
        else get_embedding_provider()
    )

    response_sentences = _split_sentences(response)
    context_sentences = _split_sentences(context)

    if not response_sentences:
        return FabricationResult(
            case_id=case_id,
            sentences_checked=0,
            fabrication_suspects=0,
            avg_max_similarity=1.0,
        )

    if not context_sentences:
        return FabricationResult(
            case_id=case_id,
            sentences_checked=len(response_sentences),
            fabrication_suspects=len(response_sentences),
            avg_max_similarity=0.0,
            flags=["no_context_for_grounding"],
        )

    all_texts = response_sentences + context_sentences
    all_embeddings = provider.encode(all_texts)
    resp_embeddings = all_embeddings[:len(response_sentences)]
    ctx_embeddings = all_embeddings[len(response_sentences):]

    suspects = 0
    low_sim_sentences: list[str] = []
    similarities: list[float] = []
    flags: list[str] = []

    for i, resp_emb in enumerate(resp_embeddings):
        max_sim = provider.max_similarity(resp_emb, ctx_embeddings)
        similarities.append(max_sim)

        if max_sim < fabrication_threshold:
            suspects += 1
            low_sim_sentences.append(response_sentences[i][:80])

    avg_sim = sum(similarities) / len(similarities) if similarities else 0.0

    if suspects > 0:
        flags.append(f"fabrication_suspects:{suspects}")

    return FabricationResult(
        case_id=case_id,
        sentences_checked=len(response_sentences),
        fabrication_suspects=suspects,
        avg_max_similarity=round(avg_sim, 4),
        low_similarity_sentences=low_sim_sentences[:5],
        flags=flags,
    )
