from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class DatasetCase(BaseModel):
    id: Optional[str] = None
    conversation: list[ConversationMessage] = Field(..., min_length=1)
    candidate_answer: str = Field(..., min_length=1)
    rubric_id: str = Field(..., min_length=1)
    human_decision: Literal["pass", "fail"]
    human_scores: Optional[dict[str, int]] = None
    rationale: Optional[str] = None


class DatasetManifest(BaseModel):
    schema_version: str = Field(default="1")
    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    cases_file: str = Field(..., min_length=1)
    format: Literal["jsonl", "yaml"] = "jsonl"
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    content_hash: Optional[str] = None
