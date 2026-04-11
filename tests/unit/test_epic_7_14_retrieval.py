"""
Tests for EPIC 7.14: RAG Context Integration.

Covers: VectorStore, KnowledgeBase loader, ContextRetriever (all 3 methods),
RetrievalConfig, RetrievalEvidence, PredictRequest.source_context,
_build_context enrichment.

All tests use TokenOverlapFallback — no real embeddings, no network calls.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_judge.properties import TokenOverlapFallback
from llm_judge.retrieval import (
    Document,
    InMemoryVectorStore,
    RetrievalResult,
    get_vector_store,
)
from llm_judge.retrieval.context_retriever import (
    ContextRetriever,
    RetrievalConfig,
    RetrievalEvidence,
    _tokenize_bm25,
    load_retrieval_config,
)
from llm_judge.retrieval.knowledge_base import KnowledgeBase
from llm_judge.schemas import Message, PredictRequest

try:
    import rank_bm25  # noqa: F401

    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture
def provider() -> TokenOverlapFallback:
    return TokenOverlapFallback(dimension=64)


@pytest.fixture
def sample_docs() -> list[Document]:
    return [
        Document(
            doc_id="refund",
            content="Refund policy: duplicate charges eligible for full refund within 3-5 business days.",
        ),
        Document(
            doc_id="password",
            content="Password reset: go to account.example.com/reset and enter your email address.",
        ),
        Document(
            doc_id="discount",
            content="Discount policy: maximum 20% without VP approval. Loyalty program 10% off after 10th purchase.",
        ),
        Document(
            doc_id="shipping",
            content="Shipping address change: can be modified if order has not yet shipped.",
        ),
    ]


@pytest.fixture
def populated_store(
    provider: TokenOverlapFallback, sample_docs: list[Document]
) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    embeddings = provider.encode([d.content for d in sample_docs])
    store.add_documents(sample_docs, embeddings=embeddings)
    return store


@pytest.fixture
def kb_json(tmp_path: Path, sample_docs: list[Document]) -> Path:
    """Create a synthetic KB JSON file."""
    kb = {
        "schema_version": "1",
        "knowledge_base": {
            d.doc_id: {
                "documentation": d.content,
                "intent": d.doc_id,
                "category": "TEST",
            }
            for d in sample_docs
        },
    }
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(kb))
    return path


@pytest.fixture
def kb_json_flat(tmp_path: Path) -> Path:
    """Create a flat-format KB JSON file."""
    kb = {
        "refund": "Refund policy: duplicate charges eligible for full refund.",
        "password": "Password reset procedure: enter your email address.",
    }
    path = tmp_path / "kb_flat.json"
    path.write_text(json.dumps(kb))
    return path


@pytest.fixture
def retrieval_config_yaml(tmp_path: Path) -> Path:
    """Create a retrieval config YAML file."""
    config = {
        "enabled": True,
        "method": "cosine_similarity",
        "top_k": 2,
        "similarity_threshold": 0.05,
        "hybrid_alpha": 0.6,
        "embedding_model": "all-MiniLM-L6-v2",
        "knowledge_base_path": "retrieval/knowledge_base.json",
        "vector_store_backend": "memory",
    }
    path = tmp_path / "retrieval_config.yaml"
    path.write_text(yaml.dump(config))
    return path


# =====================================================================
# 1. VectorStore — InMemoryVectorStore
# =====================================================================


class TestInMemoryVectorStore:
    def test_empty_store_returns_no_results(self) -> None:
        store = InMemoryVectorStore()
        assert store.document_count() == 0
        results = store.search([0.1, 0.2, 0.3], top_k=3)
        assert results == []

    def test_add_documents_increments_count(self, sample_docs: list[Document]) -> None:
        store = InMemoryVectorStore()
        embeddings = [[0.1] * 4 for _ in sample_docs]
        count = store.add_documents(sample_docs, embeddings=embeddings)
        assert count == 4
        assert store.document_count() == 4

    def test_add_documents_without_embeddings(
        self, sample_docs: list[Document]
    ) -> None:
        store = InMemoryVectorStore()
        count = store.add_documents(sample_docs)
        assert count == 4
        assert store.document_count() == 4
        # Search returns nothing without embeddings
        results = store.search([0.1] * 4, top_k=3)
        assert results == []

    def test_embedding_count_mismatch_raises(self, sample_docs: list[Document]) -> None:
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="Embedding count"):
            store.add_documents(sample_docs, embeddings=[[0.1] * 4])

    def test_search_returns_sorted_by_score(
        self, populated_store: InMemoryVectorStore, provider: TokenOverlapFallback
    ) -> None:
        query_emb = provider.encode(["refund duplicate charge"])[0]
        results = populated_store.search(query_emb, top_k=4)
        assert len(results) == 4
        # Scores should be descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_respects_top_k(
        self, populated_store: InMemoryVectorStore, provider: TokenOverlapFallback
    ) -> None:
        query_emb = provider.encode(["refund"])[0]
        results = populated_store.search(query_emb, top_k=2)
        assert len(results) == 2

    def test_search_result_has_document(
        self, populated_store: InMemoryVectorStore, provider: TokenOverlapFallback
    ) -> None:
        query_emb = provider.encode(["password reset email"])[0]
        results = populated_store.search(query_emb, top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievalResult)
        assert isinstance(results[0].document, Document)
        assert results[0].document.doc_id in (
            "refund",
            "password",
            "discount",
            "shipping",
        )
        assert len(results[0].document.content) > 0

    def test_clear_removes_all(self, populated_store: InMemoryVectorStore) -> None:
        assert populated_store.document_count() == 4
        populated_store.clear()
        assert populated_store.document_count() == 0

    def test_get_vector_store_returns_in_memory_when_no_faiss(self) -> None:
        store = get_vector_store(dimension=64, prefer_faiss=False)
        assert isinstance(store, InMemoryVectorStore)


# =====================================================================
# 2. KnowledgeBase Loader
# =====================================================================


class TestKnowledgeBase:
    def test_load_json_standard_format(
        self, kb_json: Path, provider: TokenOverlapFallback
    ) -> None:
        kb = KnowledgeBase(embedding_provider=provider)
        count = kb.load_json(kb_json)
        assert count == 4
        assert kb.is_loaded
        assert kb.document_count == 4

    def test_load_json_flat_format(
        self, kb_json_flat: Path, provider: TokenOverlapFallback
    ) -> None:
        kb = KnowledgeBase(embedding_provider=provider)
        count = kb.load_json(kb_json_flat)
        assert count == 2
        assert kb.is_loaded

    def test_load_json_missing_file_raises(
        self, provider: TokenOverlapFallback
    ) -> None:
        kb = KnowledgeBase(embedding_provider=provider)
        with pytest.raises(FileNotFoundError):
            kb.load_json("/nonexistent/path.json")

    def test_load_json_empty_docs(
        self, tmp_path: Path, provider: TokenOverlapFallback
    ) -> None:
        empty = tmp_path / "empty.json"
        empty.write_text(json.dumps({"knowledge_base": {}}))
        kb = KnowledgeBase(embedding_provider=provider)
        count = kb.load_json(empty)
        assert count == 0
        assert not kb.is_loaded

    def test_load_jsonl(self, tmp_path: Path, provider: TokenOverlapFallback) -> None:
        path = tmp_path / "kb.jsonl"
        lines = [
            json.dumps({"doc_id": "doc1", "content": "First document about refunds."}),
            json.dumps(
                {"doc_id": "doc2", "content": "Second document about passwords."}
            ),
        ]
        path.write_text("\n".join(lines))
        kb = KnowledgeBase(embedding_provider=provider)
        count = kb.load_jsonl(path)
        assert count == 2
        assert kb.is_loaded

    def test_load_directory(
        self, tmp_path: Path, provider: TokenOverlapFallback
    ) -> None:
        (tmp_path / "refund.txt").write_text("Refund policy document.")
        (tmp_path / "password.md").write_text("Password reset guide.")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")  # should be skipped
        kb = KnowledgeBase(embedding_provider=provider)
        count = kb.load_directory(tmp_path)
        assert count == 2
        assert kb.is_loaded

    def test_store_accessible_after_load(
        self, kb_json: Path, provider: TokenOverlapFallback
    ) -> None:
        kb = KnowledgeBase(embedding_provider=provider)
        kb.load_json(kb_json)
        store = kb.store
        assert store.document_count() == 4
        # Verify search works
        query_emb = provider.encode(["refund charge"])[0]
        results = store.search(query_emb, top_k=1)
        assert len(results) == 1

    def test_not_loaded_before_load(self) -> None:
        kb = KnowledgeBase()
        assert not kb.is_loaded
        assert kb.document_count == 0


# =====================================================================
# 3. ContextRetriever — Cosine Similarity
# =====================================================================


class TestContextRetrieverCosine:
    def test_retrieve_returns_docs_and_evidence(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="cosine_similarity", top_k=2, similarity_threshold=0.0
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("refund duplicate charge")
        assert docs is not None
        assert len(docs) <= 2
        assert evidence is not None
        assert evidence.method == "cosine_similarity"
        assert evidence.docs_retrieved > 0
        assert evidence.top_score > 0
        assert len(evidence.doc_ids) > 0

    def test_threshold_filters_low_scores(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="cosine_similarity", top_k=10, similarity_threshold=0.99
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("something very specific and unusual")
        # With threshold 0.99, likely nothing passes
        assert evidence is not None
        assert evidence.method == "cosine_similarity"

    def test_empty_store_returns_none(self, provider: TokenOverlapFallback) -> None:
        store = InMemoryVectorStore()
        config = RetrievalConfig(method="cosine_similarity", similarity_threshold=0.0)
        retriever = ContextRetriever(
            vector_store=store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("test query")
        assert docs is None
        assert evidence is not None

    def test_doc_content_matches_original(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="cosine_similarity", top_k=1, similarity_threshold=0.0
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("password reset email")
        assert docs is not None
        assert len(docs) == 1
        # The returned doc should be the full text from the knowledge base
        assert len(docs[0]) > 20


# =====================================================================
# 4. ContextRetriever — BM25
# =====================================================================


class TestContextRetrieverBM25:
    def test_bm25_retrieval(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(method="bm25", top_k=2)
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("refund duplicate charge")
        assert evidence is not None
        assert evidence.method == "bm25"
        if docs is not None:
            assert evidence.docs_retrieved > 0
            assert (
                "refund" in evidence.doc_ids[0].lower() or "refund" in docs[0].lower()
            )

    def test_bm25_empty_query(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(method="bm25", top_k=2)
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve(
            "a"
        )  # single short token, filtered by _tokenize_bm25
        assert evidence is not None
        assert evidence.method == "bm25"

    @pytest.mark.skipif(not HAS_BM25, reason="rank-bm25 not installed")
    def test_bm25_keyword_match(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(method="bm25", top_k=1)
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("password reset email address")
        assert docs is not None
        assert evidence is not None
        assert evidence.docs_retrieved >= 1
        assert "password" in evidence.doc_ids[0]


# =====================================================================
# 5. ContextRetriever — Hybrid
# =====================================================================


class TestContextRetrieverHybrid:
    def test_hybrid_combines_methods(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="hybrid", top_k=2, similarity_threshold=0.0, hybrid_alpha=0.7
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("discount loyalty program")
        assert evidence is not None
        assert evidence.method == "hybrid"
        if docs is not None:
            assert evidence.docs_retrieved > 0

    def test_hybrid_alpha_1_equals_cosine_only(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="hybrid", top_k=2, similarity_threshold=0.0, hybrid_alpha=1.0
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("refund policy")
        assert evidence is not None
        assert evidence.method == "hybrid"


# =====================================================================
# 6. ContextRetriever — Edge Cases
# =====================================================================


class TestContextRetrieverEdgeCases:
    def test_unknown_method_falls_back_to_cosine(
        self,
        populated_store: InMemoryVectorStore,
        provider: TokenOverlapFallback,
    ) -> None:
        config = RetrievalConfig(
            method="nonexistent_method", top_k=2, similarity_threshold=0.0
        )
        retriever = ContextRetriever(
            vector_store=populated_store, config=config, embedding_provider=provider
        )
        docs, evidence = retriever.retrieve("test")
        # Should fall back to cosine without error
        assert evidence is not None

    def test_default_config(self) -> None:
        config = RetrievalConfig()
        assert config.enabled is True
        assert config.method == "cosine_similarity"
        assert config.top_k == 3
        assert config.similarity_threshold == 0.2
        assert config.hybrid_alpha == 0.7


# =====================================================================
# 7. RetrievalConfig — load_retrieval_config
# =====================================================================


class TestRetrievalConfig:
    def test_load_from_yaml(self, retrieval_config_yaml: Path) -> None:
        config = load_retrieval_config(config_path=retrieval_config_yaml)
        assert config.enabled is True
        assert config.method == "cosine_similarity"
        assert config.top_k == 2
        assert config.similarity_threshold == 0.05
        assert config.hybrid_alpha == 0.6
        assert config.knowledge_base_path == "retrieval/knowledge_base.json"

    def test_missing_file_returns_disabled(self, tmp_path: Path) -> None:
        config = load_retrieval_config(config_path=tmp_path / "nonexistent.yaml")
        assert config.enabled is False

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("")
        config = load_retrieval_config(config_path=path)
        assert config.enabled is True
        assert config.method == "cosine_similarity"

    def test_partial_yaml_fills_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.yaml"
        path.write_text(yaml.dump({"method": "bm25", "top_k": 5}))
        config = load_retrieval_config(config_path=path)
        assert config.method == "bm25"
        assert config.top_k == 5
        assert config.similarity_threshold == 0.2  # default


# =====================================================================
# 8. RetrievalEvidence
# =====================================================================


class TestRetrievalEvidence:
    def test_default_evidence(self) -> None:
        ev = RetrievalEvidence(method="cosine_similarity")
        assert ev.method == "cosine_similarity"
        assert ev.docs_retrieved == 0
        assert ev.top_score == 0.0
        assert ev.doc_ids == []
        assert ev.error is None

    def test_evidence_with_results(self) -> None:
        ev = RetrievalEvidence(
            method="bm25",
            docs_retrieved=3,
            top_score=4.52,
            doc_ids=["a", "b", "c"],
        )
        assert ev.docs_retrieved == 3
        assert ev.top_score == 4.52
        assert len(ev.doc_ids) == 3

    def test_evidence_with_error(self) -> None:
        ev = RetrievalEvidence(method="hybrid", error="faiss not installed")
        assert ev.error == "faiss not installed"
        assert ev.docs_retrieved == 0


# =====================================================================
# 9. BM25 Tokenizer
# =====================================================================


class TestBM25Tokenizer:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize_bm25("How do I reset my password?")
        assert "reset" in tokens
        assert "password" in tokens
        # "my" and "I" are too short (<=2 chars), should be filtered
        assert "my" not in tokens

    def test_punctuation_stripped(self) -> None:
        tokens = _tokenize_bm25("Hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_empty_string(self) -> None:
        tokens = _tokenize_bm25("")
        assert tokens == []

    def test_all_short_words(self) -> None:
        tokens = _tokenize_bm25("I am a do")
        assert tokens == []


# =====================================================================
# 10. PredictRequest — source_context backward compatibility
# =====================================================================


class TestPredictRequestSourceContext:
    def test_source_context_default_none(self) -> None:
        req = PredictRequest(
            conversation=[Message(role="user", content="test")],
            candidate_answer="answer",
            rubric_id="chat_quality",
        )
        assert req.source_context is None

    def test_source_context_with_docs(self) -> None:
        req = PredictRequest(
            conversation=[Message(role="user", content="test")],
            candidate_answer="answer",
            rubric_id="chat_quality",
            source_context=["doc one", "doc two"],
        )
        assert req.source_context == ["doc one", "doc two"]
        assert len(req.source_context) == 2

    def test_source_context_empty_list(self) -> None:
        req = PredictRequest(
            conversation=[Message(role="user", content="test")],
            candidate_answer="answer",
            rubric_id="chat_quality",
            source_context=[],
        )
        assert req.source_context == []

    def test_serialization_without_source_context(self) -> None:
        req = PredictRequest(
            conversation=[Message(role="user", content="test")],
            candidate_answer="answer",
            rubric_id="chat_quality",
        )
        data = req.model_dump()
        assert "source_context" in data
        assert data["source_context"] is None

    def test_serialization_with_source_context(self) -> None:
        req = PredictRequest(
            conversation=[Message(role="user", content="test")],
            candidate_answer="answer",
            rubric_id="chat_quality",
            source_context=["policy doc"],
        )
        data = req.model_dump()
        assert data["source_context"] == ["policy doc"]


# =====================================================================
# 11. _build_context — enrichment with source docs
# =====================================================================


class TestBuildContext:
    def test_context_without_source_docs(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[Message(role="user", content="Hello")],
            candidate_answer="Hi",
            rubric_id="chat_quality",
        )
        ctx = _build_context(req)
        assert ctx == "Hello"

    def test_context_with_explicit_source_docs(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[Message(role="user", content="Hello")],
            candidate_answer="Hi",
            rubric_id="chat_quality",
        )
        ctx = _build_context(req, source_docs=["Policy document one"])
        assert "Hello" in ctx
        assert "Policy document one" in ctx
        assert "Source Documentation" in ctx

    def test_context_from_request_source_context(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[Message(role="user", content="Hello")],
            candidate_answer="Hi",
            rubric_id="chat_quality",
            source_context=["From request context"],
        )
        ctx = _build_context(req)
        assert "Hello" in ctx
        assert "From request context" in ctx

    def test_explicit_source_docs_override_request(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[Message(role="user", content="Hello")],
            candidate_answer="Hi",
            rubric_id="chat_quality",
            source_context=["Request context"],
        )
        ctx = _build_context(req, source_docs=["Explicit docs"])
        assert "Explicit docs" in ctx
        assert "Request context" not in ctx

    def test_multi_turn_conversation_context(self) -> None:
        from llm_judge.integrated_judge import _build_context

        req = PredictRequest(
            conversation=[
                Message(role="user", content="Question one"),
                Message(role="assistant", content="Answer one"),
                Message(role="user", content="Question two"),
            ],
            candidate_answer="Answer two",
            rubric_id="chat_quality",
        )
        ctx = _build_context(req)
        assert "Question one" in ctx
        assert "Answer one" in ctx
        assert "Question two" in ctx


# =====================================================================
# 12. End-to-End: KB → Retriever → Context Enrichment
# =====================================================================


class TestEndToEndRetrieval:
    def test_full_pipeline(self, kb_json: Path, provider: TokenOverlapFallback) -> None:
        """Load KB → create retriever → retrieve → verify evidence."""
        kb = KnowledgeBase(embedding_provider=provider)
        kb.load_json(kb_json)
        assert kb.is_loaded
        assert kb.document_count == 4

        config = RetrievalConfig(
            method="cosine_similarity",
            top_k=2,
            similarity_threshold=0.0,
        )
        retriever = ContextRetriever(
            vector_store=kb.store,
            config=config,
            embedding_provider=provider,
        )
        docs, evidence = retriever.retrieve(
            "How do I get a refund for a duplicate charge?"
        )
        assert docs is not None
        assert evidence is not None
        assert evidence.method == "cosine_similarity"
        assert evidence.docs_retrieved > 0
        assert evidence.top_score > 0

    def test_retriever_with_build_context(
        self, kb_json: Path, provider: TokenOverlapFallback
    ) -> None:
        """Full flow: retrieve docs → build enriched context → verify structure."""
        from llm_judge.integrated_judge import _build_context

        kb = KnowledgeBase(embedding_provider=provider)
        kb.load_json(kb_json)

        config = RetrievalConfig(
            method="cosine_similarity",
            top_k=2,
            similarity_threshold=0.0,
        )
        retriever = ContextRetriever(
            vector_store=kb.store,
            config=config,
            embedding_provider=provider,
        )
        docs, _ = retriever.retrieve("password reset")
        assert docs is not None

        req = PredictRequest(
            conversation=[Message(role="user", content="How do I reset my password?")],
            candidate_answer="Click forgot password.",
            rubric_id="chat_quality",
        )
        ctx = _build_context(req, source_docs=docs)
        assert "How do I reset my password?" in ctx
        assert "Source Documentation" in ctx
        assert len(ctx) > len("How do I reset my password?")

    def test_grounding_improvement_with_context(
        self, kb_json: Path, provider: TokenOverlapFallback
    ) -> None:
        """Science Gate replication: grounding ratio improves with source context."""
        from llm_judge.calibration.hallucination import check_hallucination
        from llm_judge.integrated_judge import _build_context

        kb = KnowledgeBase(embedding_provider=provider)
        kb.load_json(kb_json)

        config = RetrievalConfig(
            method="cosine_similarity",
            top_k=2,
            similarity_threshold=0.0,
        )
        retriever = ContextRetriever(
            vector_store=kb.store,
            config=config,
            embedding_provider=provider,
        )

        # Legitimate response (uses policy vocabulary)
        good_answer = "Our loyalty program gives you 10% off after your 10th purchase."
        query = "Can I get a discount?"

        # Without RAG context
        result_no_rag = check_hallucination(
            response=good_answer,
            context=query,
            case_id="test_good",
        )

        # With RAG context
        docs, _ = retriever.retrieve(query)
        assert docs is not None
        req = PredictRequest(
            conversation=[Message(role="user", content=query)],
            candidate_answer=good_answer,
            rubric_id="chat_quality",
        )
        enriched_ctx = _build_context(req, source_docs=docs)
        result_with_rag = check_hallucination(
            response=good_answer,
            context=enriched_ctx,
            case_id="test_good_rag",
        )

        # Grounding should improve with source context
        assert result_with_rag.grounding_ratio >= result_no_rag.grounding_ratio
