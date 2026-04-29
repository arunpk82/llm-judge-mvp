"""Types for the Control Plane: request/result envelopes and errors."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
    ProvenanceEnvelope,
)

__all__ = [
    "CapabilityIntegrityRecord",
    "ConfigurationError",
    "Integrity",
    "MissingProvenanceError",
    "RubricNotInRegistryError",
    "SingleEvaluationRequest",
    "SingleEvaluationResult",
]


class MissingProvenanceError(Exception):
    """Raised when a wrapper's pre-check finds a required upstream
    stamp absent from the envelope. The message names the missing
    stamp so bypass attempts fail loudly at the wrapper boundary."""


class ConfigurationError(Exception):
    """Raised at platform startup when a required configuration value
    is missing or invalid for the current mode. Production mode without
    an HMAC key is the first instance; subsequent packets extend
    ``validate_configuration()`` to cover layer vocabulary alignment,
    artifact root validation, and governance preflight reachability."""


class RubricNotInRegistryError(ValueError):
    """Raised when ``rubric_store._resolve_version`` is asked for a
    rubric_id that is absent from ``rubrics/registry.yaml``'s
    ``latest:`` map.

    Subclasses :class:`ValueError` so existing callers that
    ``except ValueError:`` around rubric resolution continue to work
    unchanged; new callers can catch this class specifically when they
    want to distinguish "unknown rubric" from other ValueError causes
    without inspecting the message string.

    Scope note: the sibling raise site at
    ``rubric_store._resolve_version`` line 106-109 ("invalid latest
    version for rubric '<id>': <value>") describes a known rubric
    whose ``latest:`` pointer is empty/whitespace — registry
    malformation rather than a registry-membership gap. That case
    remains a plain :class:`ValueError`; it is closer to
    :class:`llm_judge.rubric_store.RubricSchemaError`'s territory than
    to :class:`RubricNotInRegistryError`'s and may be migrated
    separately in a future packet."""


class Integrity(BaseModel):
    """Legacy run-level integrity summary (CP-1 shape).

    Retained alongside the new per-capability
    :class:`llm_judge.control_plane.envelope.CapabilityIntegrityRecord`
    list on the envelope so ``SingleEvaluationResult`` remains a
    drop-in shape for CP-1 callers. CP-1c may consolidate.
    """

    model_config = ConfigDict(frozen=True)

    complete: bool
    missing_capabilities: list[str] = Field(default_factory=list)
    reason: str | None = None


class SingleEvaluationRequest(BaseModel):
    """Minimum request shape for one-instance evaluation."""

    model_config = ConfigDict(frozen=True)

    response: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    rubric_id: str = Field(
        ...,
        min_length=1,
        description="Governed rubric identifier",
    )
    rubric_version: str = Field(
        default="latest",
        description="Rubric version, or 'latest' for current",
    )
    caller_id: str | None = None
    request_id: str | None = None


class SingleEvaluationResult(BaseModel):
    """Runner output: verdict + manifest pointer + stamped envelope."""

    model_config = ConfigDict(frozen=True)

    verdict: dict[str, Any]
    manifest_id: str
    envelope: ProvenanceEnvelope
    integrity: Integrity
