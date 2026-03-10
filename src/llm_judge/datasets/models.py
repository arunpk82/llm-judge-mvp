from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetMetadata(BaseModel):
    """Governed dataset metadata contract.

    Stored as datasets/<dataset_id>/dataset.yaml.

    Minimal by design: we can extend later without breaking the core.
    """

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    schema_version: int = Field(ge=1, default=1)

    # Path to JSONL relative to the dataset directory.
    data_file: str = Field(min_length=1)

    # Governance metadata
    owner: str | None = None
    license: str | None = None
    task_type: str | None = None


class MessageRow(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class EvalCaseRow(BaseModel):
    """Row-level schema for evaluation datasets.

    This matches the current runner expectations:
      - conversation: list[{role, content}]
      - candidate_answer: str
    Optional fields are allowed for future enrichment.
    """

    conversation: list[MessageRow]
    candidate_answer: str

    # Optional governance/labeling fields.
    rubric_id: str | None = None
    human_decision: Any | None = None
    human_scores: Any | None = None
