from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from llm_judge.schemas import PredictRequest

Severity = Literal["weak", "strong"]


@dataclass(frozen=True)
class Flag:
    id: str
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RuleResult:
    flags: list[Flag] = field(default_factory=list)


@dataclass
class RuleContext:
    """
    Canonical context passed to deterministic rules.

    Important:
    - Must support ctx.rubric (for rubric-aware rules)
    - Must support ctx.flags (legacy pattern: rules can attach string flags)
    - Must support ctx.candidate (rules frequently access candidate text)
    """

    request: PredictRequest
    rubric: Any | None = None

    # Optional override. If None, derived from request.candidate_answer
    candidate_answer: str | None = None

    # Legacy/simple flags channel (string flags)
    flags: list[str] = field(default_factory=list)

    @property
    def candidate(self) -> str:
        """Candidate answer text used by rules."""
        if self.candidate_answer is not None:
            return self.candidate_answer
        # Fallback to request if present
        return self.request.candidate_answer or ""

    @property
    def user_text(self) -> str:
        """
        Last user message content (deterministic heuristic).
        Falls back to empty string if conversation is missing.
        """
        conv = getattr(self.request, "conversation", None)
        if not conv:
            return ""
        last = conv[-1]
        # Support both dict and object message shapes
        if isinstance(last, dict):
            c = last.get("content")
            return c if isinstance(c, str) else ""
        c = getattr(last, "content", None)
        return c if isinstance(c, str) else ""
