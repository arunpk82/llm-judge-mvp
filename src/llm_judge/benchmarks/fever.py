"""
FEVER Benchmark Adapter (EPIC 7.16).

FEVER: Thorne et al., 2018
  - 185K claims labeled SUPPORTS, REFUTES, NOT ENOUGH INFO
  - Each claim has Wikipedia evidence
  - License: Open

Dataset format:
  JSONL with: {"id": int, "claim": "...", "label": "SUPPORTS|REFUTES|NOT ENOUGH INFO",
               "evidence": [[annotation_id, evidence_id, wiki_title, sentence_idx]], ...}
"""

from __future__ import annotations

import json
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
_DEFAULT_PATH = Path("datasets/benchmarks/fever")


class FEVERAdapter(BenchmarkAdapter):
    """Loads FEVER in native format."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="FEVER",
            version="1.0",
            citation="Thorne et al., FEVER: a Large-scale Dataset for Fact "
            "Extraction and VERification, NAACL 2018",
            license="Open",
            total_cases=185445,
            test_cases=19998,
            supported_properties=["1.2", "1.3"],
            description="Fact verification benchmark. Claims labeled as "
            "SUPPORTS, REFUTES, or NOT ENOUGH INFO against Wikipedia evidence.",
            published_baselines=[
                PublishedBaseline(
                    method="DREAM (decompose-and-verify)",
                    metric="label_accuracy",
                    value=0.769,
                    source="FEVER shared task leaderboard",
                ),
                PublishedBaseline(
                    method="KGAT (kernel graph attention)",
                    metric="label_accuracy",
                    value=0.786,
                    source="FEVER shared task leaderboard",
                ),
            ],
        )

    def load_cases(
        self,
        *,
        split: str = "test",
        max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield FEVER cases."""
        possible_files = [
            "paper_test.jsonl",
            "test.jsonl",
            f"{split}.jsonl",
            "fever_test.jsonl",
            "shared_task_test.jsonl",
            "paper_dev.jsonl",
            "dev.jsonl",
        ]
        filepath = None
        for fname in possible_files:
            candidate = self._data_dir / fname
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            raise FileNotFoundError(
                f"FEVER data not found in {self._data_dir}. "
                f"Expected one of: {possible_files}"
            )

        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                if max_cases is not None and count >= max_cases:
                    return

                entry = json.loads(line)
                claim = entry.get("claim", "").strip()
                label = entry.get("label", "NOT ENOUGH INFO")
                fever_id = entry.get("id", count)

                if not claim:
                    continue

                # Extract evidence text if available
                evidence_texts: list[str] = []
                evidence_data = entry.get("evidence", [])
                for evidence_set in evidence_data:
                    if isinstance(evidence_set, list):
                        for ev in evidence_set:
                            if isinstance(ev, list) and len(ev) >= 4:
                                wiki_title = ev[2] if ev[2] else ""
                                if wiki_title:
                                    evidence_texts.append(wiki_title)

                # Map FEVER labels to our property framework
                # SUPPORTS = claim is grounded (pass)
                # REFUTES = claim contradicts source (fail — ungrounded)
                # NOT ENOUGH INFO = claim has no source support (fail — unverifiable)
                if label == "SUPPORTS":
                    is_grounded = True
                else:
                    is_grounded = False

                yield BenchmarkCase(
                    case_id=f"fever_{fever_id}",
                    request=PredictRequest(
                        conversation=[
                            Message(role="user", content="Is this claim accurate?")
                        ],
                        candidate_answer=claim,
                        rubric_id="chat_quality",
                        source_context=evidence_texts if evidence_texts else None,
                    ),
                    ground_truth=GroundTruth(
                        response_level="pass" if is_grounded else "fail",
                        property_labels={
                            "1.2": 0 if is_grounded else 1,
                            "1.3": 0 if is_grounded else 1,
                        },
                    ),
                    metadata={
                        "fever_label": label,
                        "evidence_count": len(evidence_texts),
                    },
                )
                count += 1
