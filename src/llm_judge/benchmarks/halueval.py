"""
HaluEval Benchmark Adapter (EPIC 7.16).

HaluEval: Li et al., EMNLP 2023
  - 35K examples: 5K general + 10K QA + 10K dialogue + 10K summarization
  - Each has factual + hallucinated answer pairs
  - License: MIT

Dataset format (qa_data.json):
  {"knowledge": "...", "question": "...", "right_answer": "...", "hallucinated_answer": "..."}
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
_DEFAULT_PATH = Path("datasets/benchmarks/halueval")


class HaluEvalAdapter(BenchmarkAdapter):
    """Loads HaluEval in native format."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="HaluEval",
            version="1.0",
            citation="Li et al., HaluEval: A Large-Scale Hallucination Evaluation "
                     "Benchmark for Large Language Models, EMNLP 2023",
            license="MIT",
            total_cases=35000,
            test_cases=35000,
            supported_properties=["1.1", "1.5"],
            description="Hallucination evaluation with factual/hallucinated answer pairs "
                       "across QA, dialogue, and summarization tasks.",
            published_baselines=[
                PublishedBaseline(
                    method="ChatGPT (zero-shot)",
                    metric="recognition_accuracy",
                    value=0.625,
                    source="Li et al., EMNLP 2023, Table 4",
                ),
            ],
        )

    def _load_task_file(
        self, filename: str, task_type: str, *,
        split: str, max_cases: int | None, offset: int,
    ) -> Iterator[BenchmarkCase]:
        """Load a single HaluEval task file, yielding two cases per entry."""
        filepath = self._data_dir / filename
        if not filepath.exists():
            logger.info(f"halueval.file_not_found: {filepath}")
            return

        with filepath.open("r", encoding="utf-8") as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                # JSON array format
                data = json.load(f)
            else:
                # JSONL format (one JSON object per line)
                data = [json.loads(line) for line in f if line.strip()]

        count = 0
        for i, entry in enumerate(data):
            if max_cases is not None and count >= max_cases:
                return

            knowledge = entry.get("knowledge", "")
            question = entry.get("question", "")
            right_answer = entry.get("right_answer", "")
            hallucinated_answer = entry.get("hallucinated_answer", "")

            if not question and task_type == "dialogue":
                # Dialogue format uses "dialog_history" and "knowledge"
                question = entry.get("dialog_history", "")

            if not question and task_type == "summarization":
                question = "Summarize the following document."
                knowledge = entry.get("document", knowledge)
                right_answer = entry.get("right_summary", right_answer)
                hallucinated_answer = entry.get("hallucinated_summary", hallucinated_answer)

            context = knowledge if knowledge else question

            # Case 1: correct answer (pass)
            if right_answer:
                yield BenchmarkCase(
                    case_id=f"halueval_{task_type}_{offset + i}_correct",
                    request=PredictRequest(
                        conversation=[Message(role="user", content=question)],
                        candidate_answer=right_answer,
                        rubric_id="chat_quality",
                        source_context=[context] if context else None,
                    ),
                    ground_truth=GroundTruth(
                        response_level="pass",
                        property_labels={"1.1": "pass", "1.5": "pass"},
                    ),
                    metadata={"task_type": task_type, "variant": "correct"},
                )
                count += 1

            if max_cases is not None and count >= max_cases:
                return

            # Case 2: hallucinated answer (fail)
            if hallucinated_answer:
                yield BenchmarkCase(
                    case_id=f"halueval_{task_type}_{offset + i}_hallucinated",
                    request=PredictRequest(
                        conversation=[Message(role="user", content=question)],
                        candidate_answer=hallucinated_answer,
                        rubric_id="chat_quality",
                        source_context=[context] if context else None,
                    ),
                    ground_truth=GroundTruth(
                        response_level="fail",
                        property_labels={"1.1": "fail", "1.5": "fail"},
                    ),
                    metadata={"task_type": task_type, "variant": "hallucinated"},
                )
                count += 1

    def load_cases(
        self, *, split: str = "test", max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield HaluEval cases from all available task files."""
        per_task_limit = max_cases // 3 if max_cases else None
        task_files = [
            ("qa_data.json", "qa"),
            ("dialogue_data.json", "dialogue"),
            ("summarization_data.json", "summarization"),
        ]
        offset = 0
        for filename, task_type in task_files:
            count = 0
            for case in self._load_task_file(
                filename, task_type, split=split,
                max_cases=per_task_limit, offset=offset,
            ):
                yield case
                count += 1
            offset += count

        # General data (separate format)
        general_path = self._data_dir / "general_data.json"
        if general_path.exists():
            with general_path.open("r", encoding="utf-8") as f:
                first_char = f.read(1)
                f.seek(0)
                if first_char == "[":
                    data = json.load(f)
                else:
                    data = [json.loads(line) for line in f if line.strip()]
            for i, entry in enumerate(data):
                if max_cases and i >= (max_cases - offset):
                    break
                user_query = entry.get("user_query", "")
                response = entry.get("chatgpt_response", "")
                hallucination = entry.get("hallucination", "no")

                yield BenchmarkCase(
                    case_id=f"halueval_general_{i}",
                    request=PredictRequest(
                        conversation=[Message(role="user", content=user_query)],
                        candidate_answer=response,
                        rubric_id="chat_quality",
                    ),
                    ground_truth=GroundTruth(
                        response_level="fail" if hallucination == "yes" else "pass",
                        property_labels={
                            "1.1": "fail" if hallucination == "yes" else "pass",
                            "1.5": "fail" if hallucination == "yes" else "pass",
                        },
                    ),
                    metadata={"task_type": "general"},
                )
