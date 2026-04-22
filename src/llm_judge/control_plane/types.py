"""Types for the Control Plane: request/result envelopes and errors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_judge.control_plane.envelope import ProvenanceEnvelope


class MissingProvenanceError(Exception):
    """Raised when a wrapper's pre-check finds a required upstream
    stamp absent from the envelope. The message names the missing
    stamp so bypass attempts fail loudly at the wrapper boundary."""


class Integrity(BaseModel):
    """Result integrity record — did every required sibling succeed?"""

    model_config = ConfigDict(frozen=True)

    complete: bool
    missing_capabilities: list[str] = Field(default_factory=list)
    reason: str | None = None


class SingleEvaluationRequest(BaseModel):
    """Minimum request shape for one-instance evaluation."""

    model_config = ConfigDict(frozen=True)

    response: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    caller_id: str | None = None
    request_id: str | None = None


class SingleEvaluationResult(BaseModel):
    """Runner output: verdict + manifest pointer + stamped envelope."""

    model_config = ConfigDict(frozen=True)

    verdict: dict[str, Any]
    manifest_id: str
    envelope: ProvenanceEnvelope
    integrity: Integrity
