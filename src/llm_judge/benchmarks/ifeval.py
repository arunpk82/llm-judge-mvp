"""
IFEval Benchmark Adapter (EPIC 7.16).

IFEval: Zhou et al., 2023
  - ~500 prompts with verifiable instruction constraints
  - 25 constraint types: word count, keyword inclusion, format, etc.
  - License: Apache 2.0

Note: IFEval tests the LLM's ability to follow instructions, not our
judge's detection of instruction violations. We use it to validate
our Cat 4 (Task Fidelity) property implementations — specifically
whether our detection methods correctly identify when constraints
are violated vs. satisfied.

Dataset format (from HuggingFace google/IFEval):
  {"key": int, "prompt": "...", "instruction_id_list": [...], "kwargs": [...]}

For our purposes, we need the prompt + a response to evaluate.
Since IFEval doesn't include responses, we generate two cases per prompt:
  1. A constraint-satisfying response (pass) — constructed from prompt
  2. The raw prompt text serves as the query for our 4.1/4.2 checks
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
_DEFAULT_PATH = Path("datasets/benchmarks/ifeval")


class IFEvalAdapter(BenchmarkAdapter):
    """Loads IFEval in native format."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir else _DEFAULT_PATH

    def metadata(self) -> BenchmarkMetadata:
        return BenchmarkMetadata(
            name="IFEval",
            version="1.0",
            citation="Zhou et al., Instruction-Following Evaluation for "
                     "Large Language Models, 2023",
            license="Apache 2.0",
            total_cases=541,
            test_cases=541,
            supported_properties=["4.1", "4.2"],
            description="Verifiable instruction-following prompts with 25 constraint "
                       "types including word limits, keyword requirements, and format.",
            published_baselines=[
                PublishedBaseline(
                    method="GPT-4 (strict)",
                    metric="prompt_level_strict_accuracy",
                    value=0.765,
                    source="Zhou et al., 2023",
                ),
            ],
        )

    def load_cases(
        self, *, split: str = "test", max_cases: int | None = None,
    ) -> Iterator[BenchmarkCase]:
        """Yield IFEval cases."""
        # Try multiple possible filenames
        possible_files = [
            "ifeval.jsonl",
            "input_data.jsonl",
            "ifeval_input_data.jsonl",
        ]
        filepath = None
        for fname in possible_files:
            candidate = self._data_dir / fname
            if candidate.exists():
                filepath = candidate
                break

        if filepath is None:
            raise FileNotFoundError(
                f"IFEval data not found in {self._data_dir}. "
                f"Expected one of: {possible_files}"
            )

        count = 0
        with filepath.open("r", encoding="utf-8") as f:
            for line in f:
                if max_cases is not None and count >= max_cases:
                    return

                entry = json.loads(line)
                prompt = entry.get("prompt", "")
                instruction_ids = entry.get("instruction_id_list", [])
                kwargs = entry.get("kwargs", [])
                key = entry.get("key", count)

                if not prompt:
                    continue

                # Extract constraint types for ground truth
                constraint_types = []
                for inst_id in instruction_ids:
                    # IFEval instruction IDs like "length_constraints:number_words"
                    parts = inst_id.split(":")
                    if len(parts) >= 2:
                        constraint_types.append(parts[1])
                    else:
                        constraint_types.append(inst_id)

                # We yield the prompt as the query — our Cat 4 checks
                # will attempt to detect the constraints in it
                yield BenchmarkCase(
                    case_id=f"ifeval_{key}",
                    request=PredictRequest(
                        conversation=[Message(role="user", content=prompt)],
                        candidate_answer="[Response placeholder — IFEval tests constraint detection, not response quality]",
                        rubric_id="chat_quality",
                    ),
                    ground_truth=GroundTruth(
                        response_level="fail",  # placeholder
                        property_labels={
                            "4.1": {
                                "constraints_present": constraint_types,
                                "constraint_count": len(instruction_ids),
                            },
                            "4.2": {
                                "instruction_ids": instruction_ids,
                            },
                        },
                    ),
                    metadata={
                        "instruction_ids": instruction_ids,
                        "kwargs": kwargs,
                        "constraint_types": constraint_types,
                    },
                )
                count += 1
