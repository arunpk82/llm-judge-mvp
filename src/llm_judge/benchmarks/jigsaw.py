"""
Jigsaw / Civil Comments Benchmark Adapter (EPIC 7.16).

Jigsaw Unintended Bias in Toxicity Classification
  - ~160K online comments with toxicity labels
  - From Wikipedia talk pages
  - License: CC0

Dataset format (from Kaggle):
  CSV with columns: id, comment_text, toxicity (float 0-1),
  severe_toxicity, obscene, identity_attack, insult, threat,
  plus identity mention columns
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterator

from llm_judge.benchmarks import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkMetadata,
    GroundTruth,
    PublishedBaseline,
)
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path("datasets/benchmarks/jigsaw")


class JigsawAdapter(BenchmarkAdapter):
    """Loads Jigsaw / Civil Comments in native format."""

    def __init__(
        self, data_dir: str | Path | None = None,
        toxicity_threshold: float = 0.5,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH
        self._threshold = toxicity_threshold

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="Jigsaw/CivilComments",
            version="1.0",
            citation="Jigsaw, Civil Comments dataset, "
                     "Unintended Bias in Toxicity Classification, Kaggle 2019",
            license="CC0",
            total_cases=160000,
            test_cases=160000,
            supported_properties=["3.1"],
            description="Online comments labeled for toxicity, severe toxicity, "
                       "obscenity, identity attack, insult, and threat.",
            published_baselines=[
                PublishedBaseline(
                    method="BERT (fine-tuned)",
                    metric="auc_roc",
                    value=0.946,
                    source="Kaggle competition leaderboard",
                ),
            ],
        )

    def load_cases(
        self, *, split: str = "test", max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield Jigsaw cases from CSV."""
        possible_files = [
            "train.csv", "all_data.csv", "test.csv",
            "jigsaw.csv", "civil_comments.csv",
        ]
        filepath = None
        for fname in possible_files:
            candidate = self._data_dir / fname
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            raise FileNotFoundError(
                f"Jigsaw data not found in {self._data_dir}. "
                f"Expected one of: {possible_files}"
            )

        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_cases is not None and count >= max_cases:
                    return

                comment = row.get("comment_text", row.get("text", "")).strip()
                if not comment or len(comment) < 10:
                    continue

                # Get toxicity score
                toxicity_str = row.get("toxicity", row.get("target", "0"))
                try:
                    toxicity = float(toxicity_str)
                except (ValueError, TypeError):
                    continue

                is_toxic = toxicity >= self._threshold

                # Collect subcategory scores
                subcategories: dict[str, float] = {}
                for col in ["severe_toxicity", "obscene", "identity_attack",
                           "insult", "threat"]:
                    val = row.get(col, "0")
                    try:
                        subcategories[col] = float(val)
                    except (ValueError, TypeError):
                        pass

                yield BenchmarkCase(
                    case_id=f"jigsaw_{row.get('id', count)}",
                    request=PredictRequest(
                        conversation=[Message(role="user", content="Help me with my issue.")],
                        candidate_answer=comment,
                        rubric_id="chat_quality",
                    ),
                    ground_truth=GroundTruth(
                        response_level="fail" if is_toxic else "pass",
                        property_labels={
                            "3.1": "fail" if is_toxic else "pass",
                        },
                    ),
                    metadata={
                        "toxicity_score": toxicity,
                        "subcategories": subcategories,
                    },
                )
                count += 1
