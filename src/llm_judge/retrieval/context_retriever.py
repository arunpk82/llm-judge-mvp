"""
Context Retriever — retrieves source documents for evaluation grounding.

Three retrieval methods, config-driven:
  - cosine_similarity: semantic search via EmbeddingProvider
  - bm25: keyword-based ranking (requires rank-bm25)
  - hybrid: weighted combination of cosine + BM25

Runs inside Gate 2 (IntegratedJudge.evaluate_enriched), between
_build_query() and faithfulness property checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_judge.retrieval import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalConfig:
    """Configuration for context retrieval."""

    enabled: bool = True
    method: str = "cosine_similarity"  # cosine_similarity | bm25 | hybrid
    top_k: int = 3
    similarity_threshold: float = 0.2
    hybrid_alpha: float = 0.7
    embedding_model: str = "all-MiniLM-L6-v2"
    knowledge_base_path: str | None = None
    vector_store_backend: str = "auto"


@dataclass
class RetrievalEvidence:
    """Evidence of a retrieval operation for pipeline tracing."""

    method: str
    docs_retrieved: int = 0
    top_score: float = 0.0
    doc_ids: list[str] = field(default_factory=list)
    error: str | None = None


def _tokenize_bm25(text: str) -> list[str]:
    """Simple tokenizer for BM25."""
    return [
        w.lower().strip(".,!?;:\"'()[]{}")
        for w in text.split()
        if len(w.strip(".,!?;:\"'()[]{}")) > 2
    ]


class ContextRetriever:
    """Retrieves source documents for evaluation grounding.

    Contract with IntegratedJudge:
        retrieve(query) -> tuple[list[str] | None, RetrievalEvidence | None]
        First element is the list of document texts (or None if retrieval fails).
        Second is the evidence for pipeline tracing.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        config: RetrievalConfig | None = None,
        embedding_provider: Any | None = None,
    ) -> None:
        self._store = vector_store
        self._config = config or RetrievalConfig()
        self._provider = embedding_provider
        self._bm25_index: object | None = None
        self._bm25_docs: list[tuple[str, str]] | None = None

    def _ensure_provider(self) -> Any:
        if self._provider is None:
            from llm_judge.properties import get_embedding_provider

            self._provider = get_embedding_provider(self._config.embedding_model)
        return self._provider

    def retrieve(self, query: str) -> tuple[list[str] | None, RetrievalEvidence | None]:
        """Retrieve source documents for the given query.

        Returns:
            Tuple of (document_texts, evidence). document_texts is None
            if retrieval fails or no documents found.
        """
        method = self._config.method

        try:
            if method == "cosine_similarity":
                return self._retrieve_cosine(query)
            elif method == "bm25":
                return self._retrieve_bm25(query)
            elif method == "hybrid":
                return self._retrieve_hybrid(query)
            else:
                logger.warning("retriever.unknown_method", extra={"method": method})
                return self._retrieve_cosine(query)
        except Exception as exc:
            evidence = RetrievalEvidence(method=method, error=str(exc)[:80])
            return None, evidence

    def _retrieve_cosine(
        self, query: str
    ) -> tuple[list[str] | None, RetrievalEvidence]:
        """Semantic retrieval via embedding cosine similarity."""
        provider = self._ensure_provider()
        query_embedding = provider.encode([query])[0]
        results = self._store.search(query_embedding, top_k=self._config.top_k)
        filtered = [r for r in results if r.score >= self._config.similarity_threshold]

        if not filtered:
            return None, RetrievalEvidence(method="cosine_similarity")

        docs = [r.document.content for r in filtered]
        evidence = RetrievalEvidence(
            method="cosine_similarity",
            docs_retrieved=len(docs),
            top_score=filtered[0].score,
            doc_ids=[r.document.doc_id for r in filtered],
        )
        return docs, evidence

    def _retrieve_bm25(self, query: str) -> tuple[list[str] | None, RetrievalEvidence]:
        """Keyword-based retrieval via BM25 ranking."""
        self._ensure_bm25_index()
        if self._bm25_index is None or self._bm25_docs is None:
            return None, RetrievalEvidence(
                method="bm25",
                error="bm25 index not available",
            )

        query_tokens = _tokenize_bm25(query)
        if not query_tokens:
            return None, RetrievalEvidence(method="bm25")

        scores = self._bm25_index.get_scores(query_tokens)  # type: ignore
        scored = sorted(
            zip(scores, self._bm25_docs),
            key=lambda x: x[0],
            reverse=True,
        )
        top = [
            (s, did, text) for s, (did, text) in scored[: self._config.top_k] if s > 0
        ]

        if not top:
            return None, RetrievalEvidence(method="bm25")

        docs = [text for _, _, text in top]
        evidence = RetrievalEvidence(
            method="bm25",
            docs_retrieved=len(docs),
            top_score=float(top[0][0]),
            doc_ids=[did for _, did, _ in top],
        )
        return docs, evidence

    def _retrieve_hybrid(
        self, query: str
    ) -> tuple[list[str] | None, RetrievalEvidence]:
        """Weighted combination of cosine similarity and BM25."""
        cosine_docs, cosine_ev = self._retrieve_cosine(query)
        bm25_docs, bm25_ev = self._retrieve_bm25(query)
        alpha = self._config.hybrid_alpha

        # Build merged results keyed by doc_id
        merged: dict[str, tuple[float, str]] = {}

        if cosine_docs and cosine_ev:
            for did, text in zip(cosine_ev.doc_ids, cosine_docs):
                # Approximate normalized score
                merged[did] = (alpha, text)

        if bm25_docs and bm25_ev:
            for did, text in zip(bm25_ev.doc_ids, bm25_docs):
                if did in merged:
                    prev, txt = merged[did]
                    merged[did] = (prev + (1 - alpha), txt)
                else:
                    merged[did] = ((1 - alpha), text)

        if not merged:
            return None, RetrievalEvidence(method="hybrid")

        top = sorted(merged.items(), key=lambda x: x[1][0], reverse=True)[
            : self._config.top_k
        ]
        docs = [text for _, (_, text) in top]
        evidence = RetrievalEvidence(
            method="hybrid",
            docs_retrieved=len(docs),
            top_score=top[0][1][0],
            doc_ids=[did for did, _ in top],
        )
        return docs, evidence

    def _ensure_bm25_index(self) -> None:
        """Build BM25 index from vector store documents (lazy)."""
        if self._bm25_index is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("retriever.bm25_unavailable: rank-bm25 not installed")
            return

        raw_docs = getattr(self._store, "_documents", None)
        if raw_docs is None:
            logger.warning("retriever.bm25_no_docs")
            return

        docs = []
        tokenized = []
        for doc in raw_docs:
            docs.append((doc.doc_id, doc.content))
            tokenized.append(_tokenize_bm25(doc.content))

        if tokenized:
            self._bm25_index = BM25Okapi(tokenized)
            self._bm25_docs = docs


def load_retrieval_config(
    config_path: str | Path | None = None,
) -> RetrievalConfig:
    """Load retrieval configuration from YAML.

    Looks for config in this order:
    1. Explicit path argument
    2. configs/retrieval/retrieval_config.yaml
    """
    from llm_judge.paths import config_root

    if config_path is not None:
        path = Path(config_path)
    else:
        path = config_root() / "retrieval" / "retrieval_config.yaml"

    if not path.exists():
        logger.info("retrieval_config.not_found, using defaults")
        return RetrievalConfig(enabled=False)

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    return RetrievalConfig(
        enabled=data.get("enabled", True),
        method=data.get("method", "cosine_similarity"),
        top_k=data.get("top_k", 3),
        similarity_threshold=data.get("similarity_threshold", 0.2),
        hybrid_alpha=data.get("hybrid_alpha", 0.7),
        embedding_model=data.get("embedding_model", "all-MiniLM-L6-v2"),
        knowledge_base_path=data.get("knowledge_base_path"),
        vector_store_backend=data.get("vector_store_backend", "auto"),
    )
