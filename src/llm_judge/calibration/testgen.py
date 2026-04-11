"""
Test Case Generation (L4 Step 9).

Generate evaluation test cases from context and knowledge base.
Supports happy path, edge case, and adversarial case generation.

For RAG: generates verifiable questions from source documents.
Uses deterministic templates when no LLM is available,
LLM-based generation when API key is configured.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.paths import datasets_root

logger = logging.getLogger(__name__)

GENERATED_DIR = datasets_root() / "generated"


@dataclass
class GeneratedCase:
    """A generated evaluation test case."""

    case_id: str
    category: str  # happy_path | edge_case | adversarial
    conversation: list[dict[str, str]]
    candidate_answer: str
    expected_decision: str  # pass | fail
    generation_method: str  # template | llm
    source: str  # what generated this case
    rubric_id: str = "chat_quality"


def _make_case_id(content: str, prefix: str = "gen") -> str:
    h = hashlib.sha256(content.encode()).hexdigest()[:8]
    return f"{prefix}-{h}"


# =====================================================================
# Template-based generation (no LLM needed)
# =====================================================================

# Happy path templates — clear questions with good answers
_HAPPY_PATH_TEMPLATES: list[dict[str, Any]] = [
    {
        "question": "What is the capital of {country}?",
        "answer": "The capital of {country} is {capital}.",
        "params": [
            {"country": "France", "capital": "Paris"},
            {"country": "Japan", "capital": "Tokyo"},
            {"country": "Brazil", "capital": "Brasília"},
            {"country": "Australia", "capital": "Canberra"},
            {"country": "Egypt", "capital": "Cairo"},
        ],
    },
    {
        "question": "Explain what {concept} means in simple terms.",
        "answer": "{concept} refers to {explanation}.",
        "params": [
            {
                "concept": "photosynthesis",
                "explanation": "the process by which plants convert sunlight into energy using carbon dioxide and water",
            },
            {
                "concept": "gravity",
                "explanation": "the force that pulls objects toward each other, keeping us on the ground",
            },
            {
                "concept": "democracy",
                "explanation": "a system of government where citizens vote to choose their leaders",
            },
        ],
    },
]

# Edge case templates — boundary conditions
_EDGE_CASE_TEMPLATES: list[dict[str, Any]] = [
    {
        "question": "What is {value}?",
        "answer": "{answer}",
        "expected_decision": "pass",
        "params": [
            {"value": "0 divided by 1", "answer": "0"},
            {"value": "the square root of 0", "answer": "0"},
            {"value": "-1 times -1", "answer": "1"},
        ],
    },
    {
        "question": "{question}",
        "answer": "",  # empty answer — should fail
        "expected_decision": "fail",
        "params": [
            {"question": "What is machine learning?"},
            {"question": "Explain quantum computing."},
        ],
    },
]

# Adversarial templates — attempts to trick the evaluator
_ADVERSARIAL_TEMPLATES: list[dict[str, Any]] = [
    {
        "question": "What is 2+2?",
        "answer": "{answer}",
        "expected_decision": "fail",
        "params": [
            {"answer": "5"},
            {"answer": "The answer depends on your perspective."},
            {"answer": "As an AI, I cannot perform mathematical calculations."},
        ],
    },
    {
        "question": "Is the Earth flat?",
        "answer": "{answer}",
        "expected_decision": "fail",
        "params": [
            {
                "answer": "Yes, many people believe the Earth is flat and there is evidence to support this."
            },
            {"answer": "There are compelling arguments on both sides of this debate."},
        ],
    },
]


def generate_template_cases(
    *,
    categories: list[str] | None = None,
    seed: int = 42,
    max_per_category: int = 10,
    rubric_id: str = "chat_quality",
) -> list[GeneratedCase]:
    """
    Generate test cases from deterministic templates.

    No LLM needed — uses predefined question/answer patterns.
    """
    if categories is None:
        categories = ["happy_path", "edge_case", "adversarial"]

    rng = random.Random(seed)
    cases: list[GeneratedCase] = []

    template_map = {
        "happy_path": _HAPPY_PATH_TEMPLATES,
        "edge_case": _EDGE_CASE_TEMPLATES,
        "adversarial": _ADVERSARIAL_TEMPLATES,
    }

    for category in categories:
        templates = template_map.get(category, [])
        category_cases: list[GeneratedCase] = []

        for tmpl in templates:
            expected = str(tmpl.get("expected_decision", "pass"))
            for params in tmpl["params"]:
                question = str(tmpl["question"]).format(**params)
                answer_tmpl = str(tmpl["answer"])
                answer = answer_tmpl.format(**params) if answer_tmpl else ""

                case_id = _make_case_id(f"{question}{answer}{category}")
                category_cases.append(
                    GeneratedCase(
                        case_id=case_id,
                        category=category,
                        conversation=[{"role": "user", "content": question}],
                        candidate_answer=answer,
                        expected_decision=expected,
                        generation_method="template",
                        source=f"template:{category}",
                        rubric_id=rubric_id,
                    )
                )

        rng.shuffle(category_cases)
        cases.extend(category_cases[:max_per_category])

    return cases


# =====================================================================
# RAG-style generation from source documents
# =====================================================================


def generate_from_document(
    *,
    document_text: str,
    source_name: str = "document",
    max_cases: int = 5,
    seed: int = 42,
) -> list[GeneratedCase]:
    """
    Generate verifiable question-answer pairs from a source document.

    Extracts sentences and creates questions that can be verified
    against the source. No LLM needed — uses sentence extraction.
    """
    rng = random.Random(seed)

    # Split into sentences
    sentences = [
        s.strip()
        for s in document_text.replace("\n", " ").split(".")
        if len(s.strip()) > 20
    ]

    if not sentences:
        return []

    rng.shuffle(sentences)
    cases: list[GeneratedCase] = []

    for sentence in sentences[:max_cases]:
        # Create a question from the sentence
        # Simple heuristic: "What is true about [topic]?"
        words = sentence.split()
        if len(words) < 5:
            continue

        topic = " ".join(words[:4])
        question = f"Based on the document, what is true about {topic}?"

        case_id = _make_case_id(f"{question}{sentence}")
        cases.append(
            GeneratedCase(
                case_id=case_id,
                category="rag_verification",
                conversation=[
                    {"role": "user", "content": f"Context: {sentence}"},
                    {"role": "user", "content": question},
                ],
                candidate_answer=sentence,
                expected_decision="pass",
                generation_method="document_extraction",
                source=f"document:{source_name}",
            )
        )

    return cases


# =====================================================================
# Export generated cases to dataset format
# =====================================================================


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def export_generated_dataset(
    cases: list[GeneratedCase],
    *,
    dataset_id: str = "generated_test",
    version: str = "v1",
    output_dir: Path = GENERATED_DIR,
) -> Path:
    """
    Export generated cases as a governed dataset (JSONL + dataset.yaml).

    The output can be used directly with the evaluation pipeline.
    """
    ds_dir = output_dir / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Write JSONL
    data_path = ds_dir / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for case in cases:
            row = {
                "case_id": case.case_id,
                "conversation": case.conversation,
                "candidate_answer": case.candidate_answer,
                "rubric_id": case.rubric_id,
                "human_decision": case.expected_decision,
                "meta": {
                    "category": case.category,
                    "generation_method": case.generation_method,
                    "source": case.source,
                },
            }
            f.write(json.dumps(row) + "\n")

    # Write dataset.yaml
    import hashlib

    h = hashlib.sha256()
    with data_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    content_hash = f"sha256:{h.hexdigest()}"

    meta = {
        "dataset_id": dataset_id,
        "version": version,
        "data_file": "data.jsonl",
        "owner": "test-generator",
        "task_type": "generated",
        "content_hash": content_hash,
    }
    (ds_dir / "dataset.yaml").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    logger.info(
        "testgen.exported",
        extra={"dataset_id": dataset_id, "cases": len(cases), "path": str(ds_dir)},
    )

    return ds_dir
