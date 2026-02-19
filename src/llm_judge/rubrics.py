"""Lightweight rubric registry.

This module exists primarily to expose a small, import-friendly registry for
tests and basic introspection.

Source of truth for rubrics is the YAML files under `rubrics/`, loaded via
`llm_judge.rubric_store`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from llm_judge.rubric_store import get_rubric


@dataclass(frozen=True)
class RubricSummary:
    id: str
    version: str
    dimensions: List[str]


def _summarize(rubric_id: str) -> RubricSummary:
    r = get_rubric(rubric_id)
    return RubricSummary(id=r.rubric_id, version=r.version, dimensions=list(r.dimensions))


# Minimal registry. Extend as additional rubrics are introduced.
RUBRICS: Dict[str, RubricSummary] = {
    "chat_quality": _summarize("chat_quality"),
}
