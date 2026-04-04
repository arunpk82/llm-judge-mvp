"""
Vector Store — pluggable interface for document storage and retrieval.

Implementations:
  - InMemoryVectorStore: cosine similarity over a Python list (small KBs)
  - FAISSVectorStore: FAISS index for larger knowledge bases (1000+ docs)

Both implementations use the EmbeddingProvider interface from
llm_judge.properties for consistent embedding generation.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A retrievable document in the knowledge base."""
    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """A document matched by retrieval with its similarity score."""
    document: Document
    score: float


class VectorStore(ABC):
    """Interface for vector-based document storage and retrieval."""

    @abstractmethod
    def add_documents(
        self, documents: list[Document],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        """Add documents to the store. Returns count added."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 3,
    ) -> list[RetrievalResult]:
        """Search for nearest documents by embedding similarity."""
        raise NotImplementedError

    @abstractmethod
    def document_count(self) -> int:
        """Return total number of indexed documents."""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Remove all documents from the store."""
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """
    Simple in-memory vector store using cosine similarity.

    Suitable for small knowledge bases (< 500 docs). For larger
    collections, use FAISSVectorStore.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._embeddings: list[list[float]] = []

    def add_documents(
        self, documents: list[Document],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        if embeddings and len(embeddings) != len(documents):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != "
                f"document count ({len(documents)})"
            )
        self._documents.extend(documents)
        if embeddings:
            self._embeddings.extend(embeddings)
        return len(documents)

    def search(
        self, query_embedding: list[float], top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not self._embeddings:
            return []

        scores = []
        for i, doc_emb in enumerate(self._embeddings):
            score = _cosine_similarity(query_embedding, doc_emb)
            scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:top_k]:
            results.append(RetrievalResult(
                document=self._documents[idx],
                score=round(score, 4),
            ))
        return results

    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings.clear()


class FAISSVectorStore(VectorStore):
    """
    FAISS-backed vector store for larger knowledge bases.

    Handles 1000+ documents efficiently with approximate nearest
    neighbor search. Falls back to InMemoryVectorStore if faiss
    is not installed.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._documents: list[Document] = []
        self._index = None
        self._ensure_faiss()

    def _ensure_faiss(self) -> None:
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self._dimension)  # inner product (cosine on normalized)
            self._faiss = faiss
            logger.info("vectorstore.faiss_initialized", extra={"dimension": self._dimension})
        except ImportError:
            logger.warning(
                "vectorstore.faiss_not_available",
                extra={"fallback": "in_memory"},
            )
            self._index = None
            self._faiss = None

    def add_documents(
        self, documents: list[Document],
        embeddings: list[list[float]] | None = None,
    ) -> int:
        if not embeddings:
            raise ValueError("FAISSVectorStore requires embeddings")
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Embedding count ({len(embeddings)}) != "
                f"document count ({len(documents)})"
            )

        self._documents.extend(documents)

        if self._index is not None and self._faiss is not None:
            import numpy as np
            vectors = np.array(embeddings, dtype=np.float32)
            # L2-normalize for cosine similarity via inner product
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vectors = vectors / norms
            self._index.add(vectors)
        else:
            # Fallback: store raw embeddings
            if not hasattr(self, "_fallback_embeddings"):
                self._fallback_embeddings: list[list[float]] = []
            self._fallback_embeddings.extend(embeddings)

        return len(documents)

    def search(
        self, query_embedding: list[float], top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not self._documents:
            return []

        if self._index is not None and self._faiss is not None:
            import numpy as np
            query = np.array([query_embedding], dtype=np.float32)
            norm = np.linalg.norm(query)
            if norm > 0:
                query = query / norm
            k = min(top_k, len(self._documents))
            scores, indices = self._index.search(query, k)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append(RetrievalResult(
                    document=self._documents[idx],
                    score=round(float(score), 4),
                ))
            return results
        else:
            # Fallback to linear search
            fallback = getattr(self, "_fallback_embeddings", [])
            scores = []
            for i, doc_emb in enumerate(fallback):
                score = _cosine_similarity(query_embedding, doc_emb)
                scores.append((i, score))
            scores.sort(key=lambda x: x[1], reverse=True)
            return [
                RetrievalResult(
                    document=self._documents[idx],
                    score=round(score, 4),
                )
                for idx, score in scores[:top_k]
            ]

    def document_count(self) -> int:
        return len(self._documents)

    def clear(self) -> None:
        self._documents.clear()
        if self._index is not None and self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self._dimension)
        if hasattr(self, "_fallback_embeddings"):
            self._fallback_embeddings.clear()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_vector_store(
    dimension: int = 384, prefer_faiss: bool = True,
) -> VectorStore:
    """Get the best available vector store implementation."""
    if prefer_faiss:
        try:
            import faiss  # noqa: F401
            return FAISSVectorStore(dimension=dimension)
        except ImportError:
            pass
    return InMemoryVectorStore()
