"""ProvenanceEnvelope — immutable, signed provenance for one request.

Stamped as each capability completes. HMAC-SHA256 signature is
recomputed on every ``stamped(...)`` call; verifying the signature
matches a canonicalised serialisation detects tampering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = structlog.get_logger()

_DEFAULT_DEV_KEY = "dev-key-not-for-prod"
_HMAC_ENV_VAR = "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY"
_DEFAULT_KEY_WARNED = False


def _resolve_hmac_key() -> bytes:
    global _DEFAULT_KEY_WARNED
    key = os.environ.get(_HMAC_ENV_VAR)
    if key is None or not key.strip():
        if not _DEFAULT_KEY_WARNED:
            logger.warning(
                "control_plane.hmac.default_key_in_use",
                env_var=_HMAC_ENV_VAR,
                hint="set env var for non-development use",
            )
            _DEFAULT_KEY_WARNED = True
        key = _DEFAULT_DEV_KEY
    return key.encode("utf-8")


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise to a stable byte string for HMAC input.

    datetimes serialise to ISO strings; the envelope excludes its own
    ``signature`` field before calling this.
    """

    def _default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"non-serialisable field: {type(obj).__name__}")

    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_default,
    ).encode("utf-8")


def compute_signature(payload: dict[str, Any]) -> str:
    """HMAC-SHA256 hex digest over the canonical payload."""
    return hmac.new(
        _resolve_hmac_key(),
        _canonical_bytes(payload),
        hashlib.sha256,
    ).hexdigest()


class CapabilityIntegrityRecord(BaseModel):
    """Per-capability outcome record carried on the envelope (CP-1b).

    ``status`` values:
      - ``success``: the capability ran and returned normally.
      - ``failure``: the capability raised; ``error_type`` /
        ``error_message`` describe what happened.
      - ``skipped_upstream_failure``: the capability did not run
        because an earlier capability in the chain failed.
    """

    model_config = ConfigDict(frozen=True)

    capability_id: str = Field(..., min_length=1)
    status: Literal["success", "failure", "skipped_upstream_failure"]
    error_type: str | None = None
    error_message: str | None = None


class ProvenanceEnvelope(BaseModel):
    """Immutable provenance envelope. Stamp via ``stamped(...)``.

    CP-1b additions:
      - ``schema_version``: ``1`` for CP-1-shape envelopes (parsed
        from dicts that omit the field); ``2`` for CP-1b-shape
        envelopes constructed via :func:`new_envelope` or
        :meth:`stamped`.
      - ``integrity``: list of :class:`CapabilityIntegrityRecord`,
        appended via :meth:`with_integrity` as each capability's
        outcome is known. Empty by default.

    Design note on helpers: ``stamped(capability=..., ...)`` remains
    the CP-1 contract (capability chain + field updates). A separate
    :meth:`with_integrity` method appends an outcome record without
    touching the chain. Two focused helpers keep CP-1 tests
    unmodified and make Runner code explicit about *what* it's
    recording (a stamp vs. an outcome).
    """

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(..., min_length=1)
    caller_id: str = Field(..., min_length=1)
    arrived_at: datetime
    parent_attestation_id: str | None = None

    dataset_registry_id: str | None = None
    input_hash: str | None = None
    rule_set_version: str | None = None
    rules_fired: list[str] | None = None

    platform_version: str = Field(..., min_length=1)
    capability_chain: list[str] = Field(default_factory=list)
    schema_version: int = 2
    integrity: list[CapabilityIntegrityRecord] = Field(default_factory=list)
    signature: str = ""

    @model_validator(mode="before")
    @classmethod
    def _backfill_legacy_schema(cls, data: Any) -> Any:
        """CP-1 envelopes (schema_version absent) parse as v1.

        Only applies to dict-shaped inputs (e.g. ``model_validate`` of
        a persisted CP-1 manifest). Direct ``ProvenanceEnvelope(...)``
        kwargs flow through here too — new code must pass
        ``schema_version=2`` explicitly; :func:`new_envelope` and
        :meth:`stamped` do that.
        """
        if isinstance(data, dict) and "schema_version" not in data:
            data["schema_version"] = 1
        return data

    def _payload_without_signature(self) -> dict[str, Any]:
        data = self.model_dump()
        data.pop("signature", None)
        return data

    def verify_signature(self) -> bool:
        if not self.signature:
            return False
        expected = compute_signature(self._payload_without_signature())
        return hmac.compare_digest(expected, self.signature)

    def stamped(
        self,
        *,
        capability: str,
        **fields: Any,
    ) -> "ProvenanceEnvelope":
        """Return a new envelope with ``fields`` set, ``capability``
        appended to the chain, signature recomputed.

        The original is not mutated (Pydantic frozen model raises on
        attribute assignment).
        """
        if not capability:
            raise ValueError("capability must be a non-empty string")

        base = self.model_dump()
        base.update(fields)
        base["capability_chain"] = list(self.capability_chain) + [capability]
        base["signature"] = ""
        base["signature"] = compute_signature(
            {k: v for k, v in base.items() if k != "signature"}
        )
        return ProvenanceEnvelope(**base)

    def with_integrity(
        self,
        record: CapabilityIntegrityRecord,
    ) -> "ProvenanceEnvelope":
        """Return a new envelope with ``record`` appended to
        ``integrity`` and signature recomputed. Does NOT append to
        ``capability_chain`` — use :meth:`stamped` for that.
        """
        base = self.model_dump()
        # ``base["integrity"]`` is already a list of dicts from model_dump;
        # append the new record's dict form so the list stays serialisable.
        base["integrity"] = list(base.get("integrity") or []) + [
            record.model_dump()
        ]
        base["signature"] = ""
        base["signature"] = compute_signature(
            {k: v for k, v in base.items() if k != "signature"}
        )
        return ProvenanceEnvelope(**base)


def new_envelope(
    *,
    request_id: str,
    caller_id: str,
    arrived_at: datetime,
    platform_version: str,
    parent_attestation_id: str | None = None,
) -> ProvenanceEnvelope:
    """Construct the initial envelope at Runner entry. Signs with an
    empty ``capability_chain``; the first ``stamped(...)`` call will
    append CAP-1 and resign.
    """
    seed = ProvenanceEnvelope(
        request_id=request_id,
        caller_id=caller_id,
        arrived_at=arrived_at,
        parent_attestation_id=parent_attestation_id,
        platform_version=platform_version,
        schema_version=2,
    )
    payload = seed._payload_without_signature()
    sig = compute_signature(payload)
    return seed.model_copy(update={"signature": sig})
