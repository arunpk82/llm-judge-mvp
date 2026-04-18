"""
RAGTruth Benchmark Adapter (EPIC 7.16, Phase 1).

Loads RAGTruth dataset in its native format (response.jsonl + source_info.jsonl)
without modification. Maps to platform evaluation format at runtime.

RAGTruth: Niu et al., ACL 2024
  - 17,790 responses (2,700 test) with word-level hallucination annotations
  - 4 types: Evident Baseless Info, Evident Conflict, Subtle Baseless Info, Subtle Conflict
  - 3 task types: Summary, QA, Data2txt
  - Published baseline: GPT-4 response-level F1 = 63-64%
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from llm_judge.benchmarks import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkMetadata,
    GroundTruth,
    PublishedBaseline,
    SpanAnnotation,
)
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path("datasets/benchmarks/ragtruth")


class RAGTruthAdapter(BenchmarkAdapter):
    """Loads RAGTruth in native format — files stay unmodified on disk."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH
        self._source_cache: dict[str, dict[str, Any]] | None = None
        self._benchmark_ids: set[str] | None = None

    def set_benchmark_filter(self, benchmark_path: str | Path) -> None:
        """Load a benchmark definition and restrict load_cases to those IDs.

        Args:
            benchmark_path: Path to benchmark JSON with ``response_ids`` list.
        """
        with open(benchmark_path, encoding="utf-8") as f:
            bm = json.load(f)
        self._benchmark_ids = set(bm["response_ids"])
        logger.info(
            "ragtruth.benchmark_filter_set",
            extra={"count": len(self._benchmark_ids), "path": str(benchmark_path)},
        )

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="RAGTruth",
            version="1.0",
            citation="Niu et al., RAGTruth: A Hallucination Corpus for Developing "
            "Trustworthy Retrieval-Augmented Language Models, ACL 2024",
            license="Apache 2.0",
            total_cases=17790,
            test_cases=2700,
            supported_properties=["1.1", "1.2", "1.3", "1.5"],
            description="Word-level hallucination corpus for RAG evaluation. "
            "Covers Summary, QA, and Data-to-text tasks.",
            published_baselines=[
                PublishedBaseline(
                    method="GPT-4-turbo (prompt-based)",
                    metric="response_level_f1",
                    value=0.635,
                    source="Niu et al., ACL 2024, Table 5",
                ),
                PublishedBaseline(
                    method="GPT-3.5-turbo (prompt-based)",
                    metric="response_level_f1",
                    value=0.543,
                    source="Niu et al., ACL 2024, Table 5",
                ),
                PublishedBaseline(
                    method="Llama-2-13B (fine-tuned on RAGTruth train)",
                    metric="response_level_f1",
                    value=0.622,
                    source="Niu et al., ACL 2024, Table 5",
                ),
            ],
        )

    def _load_sources(self) -> dict[str, dict[str, Any]]:
        """Load and cache source_info.jsonl keyed by source_id."""
        if self._source_cache is not None:
            return self._source_cache

        source_path = self._data_dir / "source_info.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(
                f"RAGTruth source_info.jsonl not found: {source_path}"
            )

        self._source_cache = {}
        with source_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self._source_cache[row["source_id"]] = row

        logger.info(
            "ragtruth.sources_loaded",
            extra={"count": len(self._source_cache)},
        )
        return self._source_cache

    def _extract_context(self, source: dict[str, Any]) -> str:
        """Extract source context from RAGTruth source_info."""
        source_info = source.get("source_info", "")
        if isinstance(source_info, str):
            return source_info
        if isinstance(source_info, dict):
            # QA format: {"question": ..., "passages": ...}
            parts = []
            if "passages" in source_info:
                parts.append(str(source_info["passages"]))
            if "question" in source_info:
                parts.append(str(source_info["question"]))
            # Data2txt format: structured business data
            for key, val in source_info.items():
                if key not in ("question", "passages"):
                    parts.append(f"{key}: {val}")
            return "\n".join(parts)
        return str(source_info)

    def _extract_query(self, source: dict[str, Any]) -> str:
        """Extract the user query/prompt from RAGTruth source."""
        prompt = source.get("prompt", "")
        if prompt:
            return prompt
        source_info = source.get("source_info", {})
        if isinstance(source_info, dict) and "question" in source_info:
            return str(source_info["question"])
        return "Summarize the following content."

    def _build_ground_truth(self, row: dict[str, Any]) -> GroundTruth:
        """Convert RAGTruth labels to normalized GroundTruth."""
        labels = row.get("labels", [])
        has_hallucination = len(labels) > 0

        # Count hallucination types
        type_counts: dict[str, int] = {}
        span_annotations: list[SpanAnnotation] = []
        for label in labels:
            label_type = label.get("label_type", "unknown")
            type_counts[label_type] = type_counts.get(label_type, 0) + 1
            span_annotations.append(
                SpanAnnotation(
                    start=label.get("start", 0),
                    end=label.get("end", 0),
                    text=label.get("text", ""),
                    label_type=label_type,
                    meta=label.get("meta", ""),
                )
            )

        # Map to property-level labels
        baseless_count = type_counts.get("Evident Baseless Info", 0) + type_counts.get(
            "Subtle Baseless Info", 0
        )
        conflict_count = type_counts.get("Evident Conflict", 0) + type_counts.get(
            "Subtle Conflict", 0
        )

        property_labels: dict[str, Any] = {
            # 1.1 Groundedness: response-level — is it grounded?
            "1.1": "fail" if has_hallucination else "pass",
            # 1.2 Ungrounded claims: count of baseless info spans
            "1.2": baseless_count,
            # 1.3 Citation verification: count of conflict spans
            #      (claims that contradict source — misstated facts)
            "1.3": conflict_count,
            # 1.5 Fabrication detection: any evident baseless info
            "1.5": (
                "fail" if type_counts.get("Evident Baseless Info", 0) > 0 else "pass"
            ),
        }

        return GroundTruth(
            response_level="fail" if has_hallucination else "pass",
            property_labels=property_labels,
            span_annotations=span_annotations,
            hallucination_types=type_counts,
        )

    def load_cases(
        self,
        *,
        split: str = "test",
        max_cases: int | None = None,
        response_ids: set[str] | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield RAGTruth cases in platform-native format.

        Args:
            split: Dataset split to load (default: "test").
            max_cases: Maximum number of cases to yield.
            response_ids: If provided, only yield cases whose response ID
                is in this set. Falls back to IDs set via
                ``set_benchmark_filter()``. Used by RAGTruth-50
                benchmark to fix the evaluation set.
        """
        # Use explicit arg, then instance filter, then None (all cases)
        effective_ids = response_ids or self._benchmark_ids
        sources = self._load_sources()
        response_path = self._data_dir / "response.jsonl"
        if not response_path.exists():
            raise FileNotFoundError(
                f"RAGTruth response.jsonl not found: {response_path}"
            )

        count = 0
        with response_path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                # Filter by split (skip when benchmark filter is active —
                # the benchmark definition is the authoritative filter)
                if effective_ids is None and row.get("split", "") != split:
                    continue

                # Filter by benchmark response IDs (fixed set)
                row_id = str(row.get("id", ""))
                if effective_ids is not None and row_id not in effective_ids:
                    continue

                # Skip quality issues
                quality = row.get("quality", "good")
                if quality in ("incorrect_refusal", "truncated"):
                    continue

                source_id = row.get("source_id", "")
                source = sources.get(source_id)
                if source is None:
                    logger.warning(
                        "ragtruth.missing_source",
                        extra={"response_id": row.get("id"), "source_id": source_id},
                    )
                    continue

                # Build PredictRequest from native format
                query = self._extract_query(source)
                context = self._extract_context(source)
                response_text = row.get("response", "")

                request = PredictRequest(
                    conversation=[Message(role="user", content=query)],
                    candidate_answer=response_text,
                    rubric_id="chat_quality",
                    source_context=[context] if context else None,
                )

                ground_truth = self._build_ground_truth(row)

                yield BenchmarkCase(
                    case_id=f"ragtruth_{row.get('id', count)}",
                    request=request,
                    ground_truth=ground_truth,
                    metadata={
                        "model": row.get("model", ""),
                        "task_type": source.get("task_type", ""),
                        "source_id": source_id,
                        "temperature": row.get("temperature", 0.0),
                    },
                )

                count += 1
                if max_cases is not None and count >= max_cases:
                    return

        logger.info(
            "ragtruth.cases_loaded",
            extra={"split": split, "count": count},
        )
