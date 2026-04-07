"""
Industry Benchmark Validation Framework (EPIC 7.16).

Loads external benchmark datasets in their native format (unmodified),
runs platform evaluation, and computes standard metrics against
published ground truth.

Design decisions:
  - Parallel to DatasetRegistry, not an extension (different trust model)
  - Files stay untouched on disk — adaptation happens in-memory at runtime
  - Each adapter declares which properties it can validate
  - Published baselines carried as metadata for comparison
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from llm_judge.schemas import PredictRequest


@dataclass
class SpanAnnotation:
    """Word-level hallucination annotation from benchmark ground truth."""
    start: int
    end: int
    text: str
    label_type: str  # e.g., "Evident Baseless Info", "Evident Conflict"
    meta: str = ""


@dataclass
class GroundTruth:
    """Normalized ground truth that all adapters produce.

    Rich enough for span-level evaluation (Cat 1.2–1.5),
    works with response-level binary for simpler benchmarks.
    """
    response_level: Literal["pass", "fail"]
    property_labels: dict[str, Any] = field(default_factory=dict)
    span_annotations: list[SpanAnnotation] = field(default_factory=list)
    hallucination_types: dict[str, int] = field(default_factory=dict)


@dataclass
class BenchmarkCase:
    """A single test case from a benchmark dataset."""
    case_id: str
    request: PredictRequest
    ground_truth: GroundTruth
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PublishedBaseline:
    """Published baseline result for comparison."""
    method: str
    metric: str
    value: float
    source: str  # paper citation


@dataclass
class BenchmarkMetadata:
    """Metadata about a benchmark dataset."""
    name: str
    version: str
    citation: str
    license: str
    total_cases: int
    test_cases: int
    supported_properties: list[str]
    published_baselines: list[PublishedBaseline] = field(default_factory=list)
    description: str = ""


class BenchmarkAdapter(ABC):
    """Interface for loading external benchmark datasets.

    Each benchmark gets a concrete implementation. This is the ONLY
    place that understands the benchmark's native file format.
    """

    @abstractmethod
    def metadata(self) -> BenchmarkMetadata:
        """Return benchmark metadata including supported properties."""
        raise NotImplementedError

    @abstractmethod
    def load_cases(
        self, *, split: str = "test", max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield benchmark cases in platform-native format.

        Args:
            split: Dataset split to load ("train" or "test").
            max_cases: Optional limit on number of cases to load.

        Yields:
            BenchmarkCase with PredictRequest + GroundTruth.
        """
        raise NotImplementedError

    def supported_properties(self) -> list[str]:
        """Shortcut to metadata().supported_properties."""
        return self.metadata().supported_properties
