"""
Master Ground Truth Adapter (EPIC 7.16).

Loads the internally generated ground truth dataset covering
all deterministic properties (1.1, 1.5, 3.1, 3.2, 3.3, 4.1, 4.2)
with 3000 cases from Bitext CS data + synthetic injections.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Literal, cast

from llm_judge.benchmarks import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkMetadata,
    GroundTruth,
)
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path("datasets/benchmarks/master_ground_truth")


class MasterGroundTruthAdapter(BenchmarkAdapter):
    """Loads master ground truth dataset covering all deterministic properties."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="MasterGroundTruth",
            version="1.0",
            citation="Internal — Bitext CS dataset with synthetic injections",
            license="Internal",
            total_cases=3000,
            test_cases=3000,
            supported_properties=[
                "1.1",
                "1.2",
                "1.3",
                "1.4",
                "1.5",
                "3.1",
                "3.2",
                "3.3",
                "4.1",
                "4.2",
            ],
            description="3000 CS cases with known ground truth labels for all "
            "deterministic properties. Includes clean cases, PII injections, "
            "boundary violations, toxicity injections, fabrications, and "
            "instruction violations.",
            published_baselines=[],  # Internal — no published baseline
        )

    def load_cases(
        self,
        *,
        split: str = "test",
        max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield ground truth cases."""
        filepath = self._data_dir / "ground_truth.jsonl"
        if not filepath.exists():
            raise FileNotFoundError(
                f"Ground truth data not found: {filepath}\n"
                f"Generate it first: poetry run python -m llm_judge.benchmarks.generate_ground_truth"
            )

        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                if max_cases is not None and count >= max_cases:
                    return

                entry = json.loads(line)
                labels = entry.get("labels", {})

                # Determine response-level decision
                response_level = "pass"
                for pid, label in labels.items():
                    if label == "fail":
                        response_level = "fail"
                        break

                yield BenchmarkCase(
                    case_id=entry["case_id"],
                    request=PredictRequest(
                        conversation=[Message(role="user", content=entry["query"])],
                        candidate_answer=entry["response"],
                        rubric_id="chat_quality",
                        source_context=(
                            [entry["source_context"]]
                            if entry.get("source_context")
                            else None
                        ),
                    ),
                    ground_truth=GroundTruth(
                        response_level=cast(Literal["pass", "fail"], response_level),
                        property_labels=labels,
                    ),
                    metadata={
                        "intent": entry.get("intent", ""),
                        "injection_type": entry.get("injection_type", ""),
                        "injection_detail": entry.get("injection_detail", ""),
                    },
                )
                count += 1
