"""ProvenanceEnvelope — immutable, signed provenance for one request.

Stamped as each capability completes. HMAC-SHA256 signature is
recomputed on every ``stamped(...)`` call; verifying the signature
matches a canonicalised serialisation detects tampering.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

_DEFAULT_DEV_KEY = "dev-key-not-for-prod"
_HMAC_ENV_VAR = "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY"
_DEFAULT_KEY_WARNED = False


def _resolve_hmac_key() -> bytes:
    global _DEFAULT_KEY_WARNED
    key = os.environ.get(_HMAC_ENV_VAR)
    if key is None or not key.strip():
        if not _DEFAULT_KEY_WARNED:
            logger.warning(
                "control_plane.hmac.default_key_in_use — "
                "set %s for non-development use",
                _HMAC_ENV_VAR,
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


class ProvenanceEnvelope(BaseModel):
    """Immutable provenance envelope. Stamp via ``stamped(...)``."""

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
    signature: str = ""

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
    )
    payload = seed._payload_without_signature()
    sig = compute_signature(payload)
    return seed.model_copy(update={"signature": sig})
