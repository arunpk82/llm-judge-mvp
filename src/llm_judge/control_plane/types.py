"""Types for the Control Plane: request/result envelopes and errors."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
    ProvenanceEnvelope,
)

__all__ = [
    "BenchmarkContentCollisionError",
    "BenchmarkFileNotFoundError",
    "BenchmarkReference",
    "CapabilityIntegrityRecord",
    "ConfigurationError",
    "FieldOwnershipViolationError",
    "GuardrailDeniedError",
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


class GuardrailDeniedError(ConfigurationError):
    """Raised by the guardrail substrate when a registered guardrail's
    pre_call or post_call hook returns a Deny decision (CP-F8 substrate
    + CP-F12 timeout closure, L1-Pkt-B).

    Subclasses :class:`ConfigurationError` so existing
    ``except ConfigurationError:`` handlers continue to catch denial
    events — the runner's per-capability ``except Exception`` arms
    pick up these denials as capability failures alongside other
    operational errors (for CAP-1/CAP-2/CAP-7) or propagate them
    (for CAP-5, per the D5 contract). New callers can ``except
    GuardrailDeniedError:`` specifically to distinguish guardrail-
    initiated denials from other configuration errors.

    The message names the offending guardrail, capability_id, and
    decision reason so the failure is self-describing in logs and
    integrity records.

    Reserved namespace: future guardrails (rate limit, circuit
    breaker, kill switch) raise more specific subclasses of this
    class so handlers can discriminate by guardrail family without
    parsing message strings."""


class BenchmarkFileNotFoundError(ValueError):
    """Raised by
    :func:`llm_judge.datasets.benchmark_registry.register_benchmark`
    when the benchmark JSON definition file does not exist on disk
    (CP-F1 closure).

    Subclasses :class:`ValueError` so callers that already
    ``except ValueError:`` around adapter setup continue to work
    unchanged. The message names the missing path so the operator can
    fix the input quickly."""


class BenchmarkContentCollisionError(ValueError):
    """Raised by
    :func:`llm_judge.datasets.benchmark_registry.register_benchmark`
    when an existing sidecar registration record's content hash does
    not match the SHA-256 of the benchmark JSON definition file
    currently on disk (CP-F1 closure).

    Subclasses :class:`ValueError` so callers that already
    ``except ValueError:`` around adapter setup continue to work
    unchanged. The message names the benchmark id, the recorded
    hash, and the observed hash so the divergence is obvious from
    the failure alone."""


class BenchmarkReference(BaseModel):
    """Typed reference to a registered benchmark (CP-F1 closure).

    Returned by
    :func:`llm_judge.datasets.benchmark_registry.register_benchmark`
    and carried on
    :attr:`SingleEvaluationRequest.benchmark_reference`. The four
    fields are the same shape that ``_cap1_lineage_tracking`` stamps
    onto the envelope under CAP-1's allowlist
    (``benchmark_id``/``benchmark_version``/
    ``benchmark_content_hash``/``benchmark_registration_timestamp``)
    so that envelope provenance and the in-memory request agree by
    construction.

    Lives in ``types.py`` rather than the registry module per
    Pre-flight 6 recon (cleanest direction; avoids a registry → types
    circular import). The registry module imports this class plus
    the two exception classes above and re-exports nothing."""

    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., min_length=1)
    benchmark_version: str = Field(..., min_length=1)
    benchmark_content_hash: str = Field(..., min_length=1)
    benchmark_registration_timestamp: datetime


class FieldOwnershipViolationError(ValueError):
    """Raised by :meth:`ProvenanceEnvelope.stamped` when a capability
    attempts to stamp a field that is not in
    :data:`llm_judge.control_plane.field_ownership.FIELD_OWNERSHIP`
    for its capability id (CP-F3 closure of End-State property A3.3).

    Subclasses :class:`ValueError` so callers that already
    ``except ValueError:`` around envelope construction continue to
    work unchanged; new callers can catch this class specifically when
    they want to distinguish "stamped a field outside its territory"
    from other ValueError causes without inspecting the message string.

    The message names the offending capability id and the disallowed
    field key so the bypass attempt fails loudly at the wrapper
    boundary."""


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

    # CP-F1: when populated by the batch adapter, ``_cap1_lineage_tracking``
    # stamps the four benchmark provenance fields onto the envelope under
    # CAP-1's allowlist. Optional + default-None so existing per-case
    # callers (single-eval, demo, manually-constructed requests) are
    # unaffected.
    benchmark_reference: BenchmarkReference | None = None


class SingleEvaluationResult(BaseModel):
    """Runner output: verdict + manifest pointer + stamped envelope."""

    model_config = ConfigDict(frozen=True)

    verdict: dict[str, Any]
    manifest_id: str
    envelope: ProvenanceEnvelope
    integrity: Integrity
