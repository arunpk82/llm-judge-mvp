"""
Knowledge Base Loader — loads and indexes source documentation.

Supports loading from:
  - JSON knowledge base files (synthetic or exported)
  - Directory of text/markdown files
  - JSONL files (one document per line)

Indexes documents into a VectorStore using the configured
EmbeddingProvider for retrieval-ready storage.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llm_judge.retrieval import (
    Document,
    VectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Loads, indexes, and serves a document knowledge base.

    The knowledge base is the source documentation that agents
    should be drawing from. Grounding properties compare agent
    responses against these documents.
    """

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        embedding_provider: Any | None = None,
    ) -> None:
        self._store = vector_store
        self._embedding_provider = embedding_provider
        self._loaded = False
        self._source_path: str | None = None

    def _ensure_store(self) -> None:
        if self._store is None:
            dim = 384
            if self._embedding_provider is not None:
                try:
                    dim = self._embedding_provider.dimension()
                except Exception:
                    pass
            self._store = get_vector_store(dimension=dim)

    def _ensure_embedding_provider(self) -> None:
        if self._embedding_provider is None:
            from llm_judge.properties import get_embedding_provider
            self._embedding_provider = get_embedding_provider()

    @property
    def store(self) -> VectorStore:
        """Access the underlying vector store."""
        self._ensure_store()
        assert self._store is not None
        return self._store

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.document_count()

    def load_json(self, path: str | Path) -> int:
        """
        Load knowledge base from a JSON file.

        Expected format (matches synthetic_knowledge_base.json):
        {
            "knowledge_base": {
                "intent_name": {
                    "documentation": "...",
                    "intent": "...",
                    "category": "..."
                }
            }
        }

        Also supports flat format:
        {
            "intent_name": "documentation text",
            ...
        }
        """
        self._ensure_store()
        self._ensure_embedding_provider()
        assert self._store is not None
        assert self._embedding_provider is not None

        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        documents: list[Document] = []

        # Detect format
        if "knowledge_base" in data:
            kb = data["knowledge_base"]
            for intent, entry in kb.items():
                if isinstance(entry, dict):
                    content = entry.get("documentation", "")
                    metadata = {
                        k: v for k, v in entry.items()
                        if k != "documentation"
                    }
                else:
                    content = str(entry)
                    metadata = {}

                if content.strip():
                    documents.append(Document(
                        doc_id=intent,
                        content=content.strip(),
                        metadata={"intent": intent, **metadata},
                    ))
        else:
            # Flat format: {"intent_name": "text"} or {"intent_name": {"documentation": "...", ...}}
            for key, value in data.items():
                if key.startswith("_") or key in (
                    "schema_version", "description", "retrieval_method",
                    "note", "case_intent_map",
                ):
                    continue

                if isinstance(value, dict):
                    # Dict entry — extract documentation field
                    content = value.get("documentation", "")
                    metadata = {
                        k: v for k, v in value.items()
                        if k != "documentation" and isinstance(v, (str, int, float, bool))
                    }
                    metadata["source_key"] = key
                elif isinstance(value, str):
                    content = value
                    metadata = {"source_key": key}
                else:
                    content = str(value)
                    metadata = {"source_key": key}

                if content.strip():
                    documents.append(Document(
                        doc_id=key,
                        content=content.strip(),
                        metadata=metadata,
                    ))

        return self._index_documents(documents, str(path))

    def load_jsonl(self, path: str | Path) -> int:
        """
        Load knowledge base from a JSONL file.

        Each line: {"doc_id": "...", "content": "...", "metadata": {...}}
        """
        self._ensure_store()
        self._ensure_embedding_provider()
        assert self._store is not None

        path = Path(path)
        documents: list[Document] = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                doc_id = entry.get("doc_id", f"doc_{len(documents)}")
                content = entry.get("content", "")
                metadata = entry.get("metadata", {})
                if content.strip():
                    documents.append(Document(
                        doc_id=doc_id,
                        content=content.strip(),
                        metadata=metadata,
                    ))

        return self._index_documents(documents, str(path))

    def load_directory(
        self, path: str | Path, extensions: tuple[str, ...] = (".txt", ".md"),
    ) -> int:
        """
        Load knowledge base from a directory of text files.

        Each file becomes one document. File name (without extension)
        becomes the doc_id.
        """
        self._ensure_store()
        self._ensure_embedding_provider()
        assert self._store is not None

        path = Path(path)
        documents: list[Document] = []

        for fpath in sorted(path.iterdir()):
            if fpath.suffix.lower() not in extensions:
                continue
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if content.strip():
                documents.append(Document(
                    doc_id=fpath.stem,
                    content=content.strip(),
                    metadata={
                        "filename": fpath.name,
                        "extension": fpath.suffix,
                    },
                ))

        return self._index_documents(documents, str(path))

    def _index_documents(self, documents: list[Document], source: str) -> int:
        """Embed and add documents to the vector store."""
        assert self._store is not None
        assert self._embedding_provider is not None

        if not documents:
            logger.warning("knowledge_base.empty", extra={"source": source})
            return 0

        # Embed all documents
        texts = [doc.content for doc in documents]
        embeddings = self._embedding_provider.encode(texts)

        count = self._store.add_documents(documents, embeddings=embeddings)
        self._loaded = True
        self._source_path = source

        logger.info(
            "knowledge_base.loaded",
            extra={
                "source": source,
                "documents": count,
                "store_type": type(self._store).__name__,
            },
        )

        return count


def load_knowledge_base(
    path: str | Path | None = None,
    embedding_provider: Any | None = None,
) -> KnowledgeBase:
    """
    Load a knowledge base from the configured path.

    Looks for knowledge base in this order:
    1. Explicit path argument
    2. configs/retrieval/knowledge_base.json
    3. datasets/knowledge_base/ directory

    Returns an empty KnowledgeBase if no source is found.
    """
    from llm_judge.paths import config_root, datasets_root

    kb = KnowledgeBase(embedding_provider=embedding_provider)

    if path is not None:
        p = Path(path)
        if p.is_file():
            if p.suffix == ".jsonl":
                kb.load_jsonl(p)
            else:
                kb.load_json(p)
        elif p.is_dir():
            kb.load_directory(p)
        return kb

    # Try default locations
    json_path = config_root() / "retrieval" / "knowledge_base.json"
    if json_path.exists():
        kb.load_json(json_path)
        return kb

    dir_path = datasets_root() / "knowledge_base"
    if dir_path.is_dir():
        kb.load_directory(dir_path)
        return kb

    logger.info("knowledge_base.no_source_found")
    return kb
