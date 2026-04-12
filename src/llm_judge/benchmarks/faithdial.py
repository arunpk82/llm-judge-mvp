"""
FaithDial Benchmark Adapter (EPIC 7.16).

FaithDial: Dziri et al., 2022
  - Dialogue faithfulness with background knowledge
  - Each turn: knowledge + dialogue history + response + BEGIN tag (Faithful/Hallucination)
  - Closest to customer support domain (dialogue-based)
  - License: MIT

Dataset format (from HuggingFace McGill-NLP/FaithDial):
  {"knowledge": "...", "dialog_history": [...], "response": "...",
   "original_response": "...", "BEGIN": ["Hallucination"|"Faithful"|...],
   "VRM": [...]}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Literal, cast

from llm_judge.benchmarks import (
    BenchmarkAdapter,
    BenchmarkCase,
    BenchmarkMetadata,
    GroundTruth,
    PublishedBaseline,
)
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)
_DEFAULT_PATH = Path("datasets/benchmarks/faithdial")


class FaithDialAdapter(BenchmarkAdapter):
    """Loads FaithDial in native format."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="FaithDial",
            version="1.0",
            citation="Dziri et al., FaithDial: A Faithful Benchmark for "
            "Information-Seeking Dialogue, TACL 2022",
            license="MIT",
            total_cases=18000,
            test_cases=3500,
            supported_properties=["1.1", "1.2", "1.3"],
            description="Dialogue faithfulness benchmark with background knowledge. "
            "Closest to customer support domain among faithfulness benchmarks.",
            published_baselines=[
                PublishedBaseline(
                    method="Q2 (question-based faithfulness)",
                    metric="f1",
                    value=0.72,
                    source="Dziri et al., 2022",
                ),
            ],
        )

    def load_cases(
        self,
        *,
        split: str = "test",
        max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield FaithDial cases."""
        # Try multiple formats
        possible_files = [
            f"{split}.json",
            f"{split}.jsonl",
            "faithdial_test.json",
            "faithdial_test.jsonl",
            "test.json",
            "data.json",
        ]

        filepath = None
        for fname in possible_files:
            candidate = self._data_dir / fname
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            raise FileNotFoundError(
                f"FaithDial data not found in {self._data_dir}. "
                f"Expected one of: {possible_files}"
            )

        count = 0

        if filepath.suffix == ".jsonl":
            with filepath.open("r", encoding="utf-8") as f:
                for line in f:
                    if max_cases is not None and count >= max_cases:
                        return
                    entry = json.loads(line)
                    case = self._entry_to_case(entry, count)
                    if case:
                        yield case
                        count += 1
        else:
            with filepath.open("r", encoding="utf-8") as f:
                data = json.load(f)

            # FaithDial can be structured as list of dialogues or flat list of turns
            items = data if isinstance(data, list) else data.get("data", [])

            for item in items:
                # Handle nested dialogue structure
                if "utterances" in item:
                    knowledge = item.get("knowledge", "")
                    for turn in item["utterances"]:
                        if max_cases is not None and count >= max_cases:
                            return
                        turn["knowledge"] = turn.get("knowledge", knowledge)
                        case = self._entry_to_case(turn, count)
                        if case:
                            yield case
                            count += 1
                else:
                    if max_cases is not None and count >= max_cases:
                        return
                    case = self._entry_to_case(item, count)
                    if case:
                        yield case
                        count += 1

    def _entry_to_case(self, entry: dict[str, Any], index: int) -> BenchmarkCase | None:
        knowledge = entry.get("knowledge", "")
        response = entry.get("response", entry.get("original_response", ""))
        if not response:
            return None

        # Build conversation from dialogue history
        dialog_history = entry.get("dialog_history", entry.get("history", []))
        messages: list[Message] = []
        if isinstance(dialog_history, list):
            for i, turn in enumerate(dialog_history):
                if isinstance(turn, str):
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append(
                        Message(
                            role=cast(Literal["user", "assistant"], role),
                            content=str(turn),
                        )
                    )
                elif isinstance(turn, dict):
                    messages.append(
                        Message(
                            role=cast(
                                Literal["user", "assistant"], turn.get("role", "user")
                            ),
                            content=str(turn.get("content", turn.get("text", ""))),
                        )
                    )
        elif isinstance(dialog_history, str):
            messages.append(Message(role="user", content=dialog_history))

        if not messages:
            messages.append(Message(role="user", content="Tell me about this topic."))

        # Determine faithfulness from BEGIN tags
        begin_tags = entry.get("BEGIN", entry.get("begin", []))
        if isinstance(begin_tags, str):
            begin_tags = [begin_tags]

        is_hallucination = "Hallucination" in begin_tags

        return BenchmarkCase(
            case_id=f"faithdial_{index}",
            request=PredictRequest(
                conversation=messages,
                candidate_answer=response,
                rubric_id="chat_quality",
                source_context=[knowledge] if knowledge else None,
            ),
            ground_truth=GroundTruth(
                response_level="fail" if is_hallucination else "pass",
                property_labels={
                    "1.1": "fail" if is_hallucination else "pass",
                    "1.2": 1 if is_hallucination else 0,
                    "1.3": 1 if is_hallucination else 0,
                },
            ),
            metadata={
                "begin_tags": begin_tags,
                "has_knowledge": bool(knowledge),
            },
        )
