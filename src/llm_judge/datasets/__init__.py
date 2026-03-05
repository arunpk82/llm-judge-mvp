"""Dataset governance primitives.

Level-3 (Governed Evaluation Platform) introduces dataset governance:
- dataset metadata contracts
- versioned dataset resolution (dataset_id + version)
- schema validation of dataset rows

This package is intentionally small and deterministic-first.
"""

from .models import DatasetMetadata, EvalCaseRow
from .registry import DatasetRegistry

__all__ = ["DatasetMetadata", "EvalCaseRow", "DatasetRegistry"]
