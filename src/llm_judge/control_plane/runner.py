"""PlatformRunner — orchestrates CAP-1 -> [CAP-2, CAP-7] -> CAP-5.

Error-handling contract (CP-1b D5):
  **Wrappers raise. Runner catches and records.** No wrapper catches
  any exception from its underlying capability call. The Runner
  catches from the three capabilities that are tolerable to fail
  (CAP-1, CAP-2, CAP-7) and records the outcome in the envelope's
  per-capability integrity list. CAP-5 is NOT wrapped — its failures
  still propagate.
"""

from __future__ import annotations

import logging
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
    new_envelope,
)
from llm_judge.control_plane.types import (
    Integrity,
    SingleEvaluationRequest,
    SingleEvaluationResult,
)
from llm_judge.control_plane.wrappers import (
    DEFAULT_LAYERS,
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


def _failure_record(capability_id: str, exc: BaseException) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=capability_id,
        status="failure",
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
    )


def _success_record(capability_id: str) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=capability_id, status="success"
    )


def _skipped_record(capability_id: str) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=capability_id, status="skipped_upstream_failure"
    )


class PlatformRunner:
    """Runs a single evaluation end-to-end through the Control Plane.

    Rubric handling (CP-1b D4): the Control Plane currently binds
    ``rubric_id`` to a default during CP-1b; CP-1c replaces this with
    explicit rubric governance.
    """

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
        *,
        layers: list[str] | None = None,
        output_dir: Path | None = None,
    ) -> SingleEvaluationResult:
        """Run the single-evaluation flow.

        ``layers`` controls which hallucination-pipeline layers CAP-7
        runs. When ``None``, the default is ``["L1"]`` — the cheapest,
        deterministic, dependency-free layer. Higher layers
        (L2 knowledge-graph, L3 classifiers, L4 LLM-as-judge) are
        opt-in and must be named explicitly; this keeps the default
        path cheap and reproducible (CP-1b A1).

        ``output_dir`` overrides the constructor-level ``runs_root``
        for this invocation, e.g. ``tmp_path`` in tests.
        """
        active_layers = list(layers) if layers else list(DEFAULT_LAYERS)
        runs_root = output_dir if output_dir is not None else self._runs_root

        request_id = payload.request_id or f"se-{uuid.uuid4().hex[:16]}"
        caller_id = payload.caller_id or "internal"

        envelope = new_envelope(
            request_id=request_id,
            caller_id=caller_id,
            arrived_at=datetime.now(timezone.utc),
            platform_version=self._platform_version,
        )

        # --- CAP-1 (wrapped; failure is captured, not propagated) ---
        dataset_handle: Any = None
        cap1_failed = False
        try:
            envelope, dataset_handle = invoke_cap1(
                envelope, payload, transient_root=self._transient_root
            )
            envelope = envelope.with_integrity(_success_record("CAP-1"))
        except Exception as exc:
            cap1_failed = True
            envelope = envelope.with_integrity(_failure_record("CAP-1", exc))
            envelope = envelope.with_integrity(_skipped_record("CAP-2"))
            envelope = envelope.with_integrity(_skipped_record("CAP-7"))
            logger.warning(
                "control_plane.cap1_failed",
                extra={"request_id": request_id, "error": str(exc)[:200]},
            )

        # --- Sibling phase: CAP-2 and CAP-7 (failures tolerated) ---
        missing: list[str] = []
        reasons: list[str] = []
        rule_hits: list[str] = []
        cap2_succeeded = False
        verdict_from_cap7: dict[str, Any] = {}
        cap7_succeeded = False

        if cap1_failed:
            missing.extend(["CAP-2", "CAP-7"])
            reasons.append("CAP-1 failed; siblings skipped")
        else:
            try:
                envelope, rule_hits = invoke_cap2(
                    envelope, payload, dataset_handle
                )
                envelope = envelope.with_integrity(_success_record("CAP-2"))
                cap2_succeeded = True
            except Exception as exc:
                missing.append("CAP-2")
                reasons.append(f"CAP-2: {type(exc).__name__}: {exc}")
                envelope = envelope.with_integrity(
                    _failure_record("CAP-2", exc)
                )
                logger.warning(
                    "control_plane.cap2_failed",
                    extra={"request_id": request_id, "error": str(exc)[:200]},
                )

            try:
                envelope, verdict_from_cap7 = invoke_cap7(
                    envelope, payload, dataset_handle, layers=active_layers
                )
                envelope = envelope.with_integrity(_success_record("CAP-7"))
                cap7_succeeded = True
            except Exception as exc:
                missing.append("CAP-7")
                reasons.append(f"CAP-7: {type(exc).__name__}: {exc}")
                envelope = envelope.with_integrity(
                    _failure_record("CAP-7", exc)
                )
                logger.warning(
                    "control_plane.cap7_failed",
                    extra={"request_id": request_id, "error": str(exc)[:200]},
                )

        # --- Aggregate: CAP-7 verdict (if any) + CAP-2 rule_hits ---
        aggregated: dict[str, Any] = {}
        if cap7_succeeded:
            aggregated.update(verdict_from_cap7)
        if cap2_succeeded:
            aggregated["rule_evidence"] = list(rule_hits)

        integrity = Integrity(
            complete=not missing and not cap1_failed,
            missing_capabilities=(
                ["CAP-1", "CAP-2", "CAP-7"] if cap1_failed else missing
            ),
            reason="; ".join(reasons) if reasons else None,
        )

        # --- CAP-5 (propagates; not wrapped per D5) ---
        envelope, manifest_id = invoke_cap5(
            envelope,
            aggregated,
            integrity.model_dump(),
            runs_root=runs_root,
        )
        envelope = envelope.with_integrity(_success_record("CAP-5"))

        return SingleEvaluationResult(
            verdict=aggregated,
            manifest_id=manifest_id,
            envelope=envelope,
            integrity=integrity,
        )
