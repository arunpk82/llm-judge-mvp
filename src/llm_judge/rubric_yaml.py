from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RuleConfig(BaseModel):
    id: str
    description: Optional[str] = None
    severity: int = Field(ge=1, le=5)


class RubricConfig(BaseModel):
    rubric_id: str
    version: str
    rules: List[RuleConfig]


class RubricRegistryConfig(BaseModel):
    """
    Schema for rubric registry/index YAMLs.

    Example:
      latest:
        chat_quality: v1
        correctness_basic: v2
    """

    latest: dict[str, str] = Field(default_factory=dict)
