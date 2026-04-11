"""
ToxiGen Benchmark Adapter (EPIC 7.16).

ToxiGen: Hartvigsen et al., 2022
  - 274K machine-generated toxic and benign statements about 13 minority groups
  - Adversarial: designed to bypass simple keyword filters
  - License: MIT

Dataset format (from HuggingFace toxigen/toxigen-data):
  {"text": "...", "target_group": "...", "toxicity_ai": float, "toxicity_human": float,
   "prompt_label": int (0=benign, 1=toxic), "generation_method": "..."}

Or from GitHub TOXIGEN:
  CSV/JSONL with text, target_group, label columns
"""

from __future__ import annotations

import csv
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
)
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path("datasets/benchmarks/toxigen")


class ToxiGenAdapter(BenchmarkAdapter):
    """Loads ToxiGen in native format."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="ToxiGen",
            version="1.0",
            citation="Hartvigsen et al., ToxiGen: A Large-Scale Machine-Generated "
            "Dataset for Adversarial and Implicit Hate Speech Detection, ACL 2022",
            license="MIT",
            total_cases=274000,
            test_cases=274000,
            supported_properties=["3.1"],
            description="Implicit toxicity and bias statements about 13 minority groups. "
            "Designed to be adversarially challenging for keyword-based detectors.",
            published_baselines=[
                PublishedBaseline(
                    method="RoBERTa (fine-tuned on ToxiGen)",
                    metric="f1",
                    value=0.817,
                    source="Hartvigsen et al., 2022",
                ),
                PublishedBaseline(
                    method="HateBERT",
                    metric="f1",
                    value=0.782,
                    source="Hartvigsen et al., 2022",
                ),
            ],
        )

    def _load_jsonl(
        self, filepath: Path, max_cases: int | None
    ) -> Iterator[BenchmarkCase]:
        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                if max_cases is not None and count >= max_cases:
                    return
                entry = json.loads(line)
                case = self._entry_to_case(entry, count)
                if case:
                    yield case
                    count += 1

    def _load_json(
        self, filepath: Path, max_cases: int | None
    ) -> Iterator[BenchmarkCase]:
        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        items = (
            data if isinstance(data, list) else data.get("data", data.get("rows", []))
        )
        for entry in items:
            if max_cases is not None and count >= max_cases:
                return
            case = self._entry_to_case(entry, count)
            if case:
                yield case
                count += 1

    def _load_csv(
        self, filepath: Path, max_cases: int | None
    ) -> Iterator[BenchmarkCase]:
        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if max_cases is not None and count >= max_cases:
                    return
                case = self._entry_to_case(row, count)
                if case:
                    yield case
                    count += 1

    def _entry_to_case(self, entry: dict[str, Any], index: int) -> BenchmarkCase | None:
        text = entry.get("text", entry.get("generation", "")).strip()
        if not text:
            return None

        # Determine toxicity label — multiple possible field names
        is_toxic = False
        if "prompt_label" in entry:
            is_toxic = int(entry["prompt_label"]) == 1
        elif "label" in entry:
            is_toxic = int(entry["label"]) == 1
        elif "toxicity_human" in entry:
            is_toxic = float(entry["toxicity_human"]) >= 0.5
        elif "toxicity_ai" in entry:
            is_toxic = float(entry["toxicity_ai"]) >= 0.5

        target_group = entry.get("target_group", "unknown")

        return BenchmarkCase(
            case_id=f"toxigen_{index}",
            request=PredictRequest(
                conversation=[Message(role="user", content="Respond to the customer.")],
                candidate_answer=text,
                rubric_id="chat_quality",
            ),
            ground_truth=GroundTruth(
                response_level="fail" if is_toxic else "pass",
                property_labels={
                    "3.1": "fail" if is_toxic else "pass",
                },
            ),
            metadata={
                "target_group": target_group,
                "is_toxic": is_toxic,
            },
        )

    def load_cases(
        self,
        *,
        split: str = "test",
        max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield ToxiGen cases from available data files."""
        # Try multiple formats
        for fname in ["toxigen_data.jsonl", "toxigen.jsonl", "data.jsonl"]:
            filepath = self._data_dir / fname
            if filepath.exists():
                yield from self._load_jsonl(filepath, max_cases)
                return

        for fname in ["toxigen_data.json", "toxigen.json", "data.json"]:
            filepath = self._data_dir / fname
            if filepath.exists():
                yield from self._load_json(filepath, max_cases)
                return

        for fname in ["toxigen_data.csv", "toxigen.csv", "data.csv", "train.csv"]:
            filepath = self._data_dir / fname
            if filepath.exists():
                yield from self._load_csv(filepath, max_cases)
                return

        raise FileNotFoundError(
            f"ToxiGen data not found in {self._data_dir}. "
            f"Expected .jsonl, .json, or .csv file."
        )
