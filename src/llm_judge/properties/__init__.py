"""
Embedding Provider — pluggable interface for semantic distance checks.

Two registered models:
  - all-MiniLM-L6-v2: fast (22MB, 384 dims), runs on every response
  - EmbeddingGemma / gte-modernbert-base: deep analysis, flagged cases

Usage:
    provider = get_embedding_provider()
    embeddings = provider.encode(["sentence one", "sentence two"])
    similarity = provider.cosine_similarity(emb1, emb2)
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interface for embedding models."""

    @abstractmethod
    def encode(self, sentences: list[str]) -> list[list[float]]:
        """Encode sentences into vectors."""
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension."""
        raise NotImplementedError

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def max_similarity(
        self, query: list[float], candidates: list[list[float]],
    ) -> float:
        """Find maximum cosine similarity between query and candidates."""
        if not candidates:
            return 0.0
        return max(self.cosine_similarity(query, c) for c in candidates)


class SentenceTransformerProvider(EmbeddingProvider):
    """Wraps sentence-transformers models (MiniLM, GTE, etc.)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(
                    "embedding.model_loaded",
                    extra={"model": self._model_name},
                )
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )

    def encode(self, sentences: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        embeddings = self._model.encode(sentences, convert_to_numpy=True)
        return [e.tolist() for e in embeddings]

    def dimension(self) -> int:
        self._ensure_loaded()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()


class TokenOverlapFallback(EmbeddingProvider):
    """
    Fallback when sentence-transformers is not installed.

    Uses token overlap (bag-of-words) as a crude embedding.
    Not semantically meaningful but allows the pipeline to run
    with degraded accuracy rather than crashing.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    def encode(self, sentences: list[str]) -> list[list[float]]:
        logger.warning("embedding.using_fallback", extra={
            "reason": "sentence-transformers not available",
        })
        result = []
        for s in sentences:
            tokens = {
                w.lower().strip(".,!?;:\"'()[]{}") for w in s.split()
                if len(w.strip(".,!?;:\"'()[]{}")) > 2
            }
            vec = [0.0] * self._dim
            for token in tokens:
                idx = hash(token) % self._dim
                vec[idx] = 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            result.append(vec)
        return result

    def dimension(self) -> int:
        return self._dim


_provider_cache: dict[str, EmbeddingProvider] = {}


def get_embedding_provider(
    model_name: str = "all-MiniLM-L6-v2",
) -> EmbeddingProvider:
    """Get embedding provider — singleton per model_name.

    Caches providers so the underlying model (e.g. MiniLM) is loaded
    once per process rather than once per evaluation case.
    """
    if model_name in _provider_cache:
        return _provider_cache[model_name]

    try:
        provider: EmbeddingProvider = SentenceTransformerProvider(model_name)
    except RuntimeError:
        logger.warning(
            "embedding.fallback_to_token_overlap",
            extra={"model": model_name},
        )
        provider = TokenOverlapFallback()

    _provider_cache[model_name] = provider
    return provider
