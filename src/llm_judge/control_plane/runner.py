"""PlatformRunner — orchestrates CAP-1 → [CAP-2, CAP-7] → CAP-5.

Sibling phase (CAP-2, CAP-7) tolerates individual failures: the
Runner records missing capabilities in the Integrity record and
returns a partial verdict. CAP-1 and CAP-5 failures propagate —
they are not sibling-optional (D5).
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.control_plane.envelope import ProvenanceEnvelope, new_envelope
from llm_judge.control_plane.types import (
    Integrity,
    SingleEvaluationRequest,
    SingleEvaluationResult,
)
from llm_judge.control_plane.wrappers import (
    invoke_cap1,
    invoke_cap2,
    invoke_cap5,
    invoke_cap7,
)

logger = logging.getLogger(__name__)

_PLATFORM_VERSION_ENV_VAR = "LLM_JUDGE_PLATFORM_VERSION"


def _resolve_platform_version() -> str:
    env = os.environ.get(_PLATFORM_VERSION_ENV_VAR)
    if env and env.strip():
        return env.strip()
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[3],
        )
        return out.decode("utf-8").strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        logger.warning(
            "control_plane.platform_version.unknown — "
            "git rev-parse HEAD failed and %s not set",
            _PLATFORM_VERSION_ENV_VAR,
        )
        return "unknown"


class PlatformRunner:
    """Runs a single evaluation end-to-end through the Control Plane."""

    def __init__(
        self,
        *,
        platform_version: str | None = None,
        transient_root: Path | None = None,
        runs_root: Path | None = None,
    ) -> None:
        self._platform_version = platform_version or _resolve_platform_version()
        self._transient_root = transient_root
        self._runs_root = runs_root

    def run_single_evaluation(
        self,
        payload: SingleEvaluationRequest,
    ) -> SingleEvaluationResult:
        request_id = payload.request_id or f"se-{uuid.uuid4().hex[:16]}"
        caller_id = payload.caller_id or "internal"

        envelope = new_envelope(
            request_id=request_id,
            caller_id=caller_id,
            arrived_at=datetime.now(timezone.utc),
            platform_version=self._platform_version,
        )

        # --- CAP-1 (propagates) ---
        envelope, dataset_handle = invoke_cap1(
            envelope, payload, transient_root=self._transient_root
        )

        # --- Sibling phase: CAP-2 and CAP-7 (failures tolerated) ---
        missing: list[str] = []
        reasons: list[str] = []

        rule_hits: list[str] = []
        try:
            envelope, rule_hits = invoke_cap2(envelope, payload, dataset_handle)
        except Exception as exc:  # CAP-2 is sibling-optional
            missing.append("CAP-2")
            reasons.append(f"CAP-2: {type(exc).__name__}: {exc}")
            logger.warning(
                "control_plane.sibling.cap2_failed",
                extra={"request_id": request_id, "error": str(exc)[:200]},
            )

        verdict: dict[str, Any] = {}
        try:
            envelope, verdict = invoke_cap7(envelope, payload, dataset_handle)
        except Exception as exc:  # CAP-7 is sibling-optional
            missing.append("CAP-7")
            reasons.append(f"CAP-7: {type(exc).__name__}: {exc}")
            logger.warning(
                "control_plane.sibling.cap7_failed",
                extra={"request_id": request_id, "error": str(exc)[:200]},
            )

        # --- Aggregate: CAP-7 verdict + CAP-2 rule_hits as evidence ---
        aggregated = dict(verdict)
        aggregated["rule_evidence"] = list(rule_hits)

        integrity = Integrity(
            complete=not missing,
            missing_capabilities=missing,
            reason="; ".join(reasons) if reasons else None,
        )

        # --- CAP-5 (propagates) ---
        envelope, manifest_id = invoke_cap5(
            envelope,
            aggregated,
            integrity.model_dump(),
            runs_root=self._runs_root,
        )

        return SingleEvaluationResult(
            verdict=aggregated,
            manifest_id=manifest_id,
            envelope=envelope,
            integrity=integrity,
        )
