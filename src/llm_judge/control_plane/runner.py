"""PlatformRunner — orchestrates CAP-1 -> [CAP-2, CAP-7] -> CAP-5.

Error-handling contract (CP-1b D5):
  **Wrappers raise. Runner catches and records.** No wrapper catches
  any exception from its underlying capability call. The Runner
  catches from the three capabilities that are tolerable to fail
  (CAP-1, CAP-2, CAP-7) and records the outcome in the envelope's
  per-capability integrity list. CAP-5 is NOT wrapped — its failures
  still propagate.

Registry-driven iteration (CP-F9 closure, L1-Pkt-B):
  :data:`CAPABILITY_REGISTRY` defines the capability sequence and
  per-capability operational metadata. ``run_single_evaluation``
  iterates the registry and dispatches by ``capability_id`` to the
  matching arm (Option A from L1-Pkt-B Layer 1 chat checkpoint).
  Per-arm logic is preserved verbatim — heterogeneous signatures
  and conditional dispatch on CAP-1 failure / aggregation between
  CAP-7 and CAP-5 make a uniform-signature refactor out of scope.
  Adding a fifth capability requires both a registry entry and a
  new dispatch arm here.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from llm_judge.control_plane.capability_registry import CAPABILITY_REGISTRY
from llm_judge.control_plane.configuration import validate_configuration
from llm_judge.control_plane.envelope import (
    CapabilityIntegrityRecord,
    new_envelope,
)
from llm_judge.control_plane.observability import Timer, emit_event
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

logger = structlog.get_logger()

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
            "control_plane.platform_version.unknown",
            reason="git rev-parse HEAD failed and env var not set",
            env_var=_PLATFORM_VERSION_ENV_VAR,
        )
        return "unknown"


def _failure_record(
    capability_id: str,
    exc: BaseException,
    duration_ms: float | None = None,
) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=capability_id,
        status="failure",
        error_type=type(exc).__name__,
        error_message=str(exc)[:500],
        duration_ms=duration_ms,
    )


def _success_record(
    capability_id: str,
    duration_ms: float | None = None,
) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=capability_id,
        status="success",
        duration_ms=duration_ms,
    )


def _skipped_record(capability_id: str) -> CapabilityIntegrityRecord:
    """Skipped capabilities never ran — ``duration_ms`` stays None."""
    return CapabilityIntegrityRecord(
        capability_id=capability_id,
        status="skipped_upstream_failure",
    )


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
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
        validate_configuration()
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

        Envelope field semantics — aggregated verdict tri-state
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        The ``verdict`` field on the returned
        :class:`SingleEvaluationResult` is a ``dict[str, Any]`` whose
        shape depends on which sibling capabilities (CAP-2, CAP-7)
        ran successfully. The ``rule_evidence`` key is deliberately
        **conditionally present**, not always-empty-when-unavailable:

        - ``CAP-2 succeeded`` → ``verdict["rule_evidence"]`` is set to
          ``list[str]`` (empty list when CAP-2 ran but fired zero
          rules; populated when rules fired). Callers can rely on the
          key's presence.
        - ``CAP-2 failed or was skipped`` → ``verdict["rule_evidence"]``
          is **absent** from the dict. Callers must use
          ``verdict.get("rule_evidence")`` or check membership with
          ``"rule_evidence" in verdict``; indexing raises
          ``KeyError``. Absence signals "CAP-2 did not run" — distinct
          from "CAP-2 ran and fired nothing" (empty list).
        - ``CAP-7 succeeded`` → CAP-7's full output keys are copied
          into the verdict (e.g. ``risk_score``, ``grounding_ratio``,
          ``layers_requested``).
        - ``CAP-7 failed or was skipped`` → those keys are absent.

        The authoritative record of which capabilities actually ran is
        :attr:`SingleEvaluationResult.envelope.integrity`
        (``list[CapabilityIntegrityRecord]``) — consult it before
        interpreting the verdict shape. The legacy
        :attr:`SingleEvaluationResult.integrity` carries the CP-1
        summary (``complete``, ``missing_capabilities``, ``reason``).

        Default bindings — rubric_id
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``rubric_id`` is now a required field on
        :class:`SingleEvaluationRequest` (CP-1c-b.1). Callers must
        specify the rubric_id explicitly::

            request = SingleEvaluationRequest(
                response="...",
                source="...",
                rubric_id="chat_quality",   # required
                rubric_version="v1",         # optional, defaults to "latest"
            )

        The ``"latest"`` sentinel is resolved against
        ``rubrics/registry.yaml`` at the wrapper boundary so
        downstream loaders see a concrete version.
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

        run_timer = Timer()
        emit_event(
            "run_started",
            request_id=request_id,
            timestamp=_utc_now_iso(),
            caller_id=caller_id,
            platform_version=self._platform_version,
        )

        # State accumulated across registry iterations. Declared above
        # the loop so dispatch arms can read prior-arm output (e.g.
        # CAP-2/CAP-7 read ``dataset_handle`` from CAP-1; CAP-5's
        # aggregation reads ``rule_hits``/``verdict_from_cap7``).
        # ``integrity`` and ``manifest_id`` are populated by CAP-5's
        # arm; the placeholders satisfy the type checker on the
        # never-reached path where CAP-5 is absent from the registry
        # (registry is strict-from-day-one — CAP-5 is always last).
        dataset_handle: Any = None
        cap1_failed = False
        missing: list[str] = []
        reasons: list[str] = []
        rule_hits: list[str] = []
        cap2_succeeded = False
        verdict_from_cap7: dict[str, Any] = {}
        cap7_succeeded = False
        aggregated: dict[str, Any] = {}
        integrity = Integrity(complete=False)
        manifest_id = ""

        with run_timer:
            for spec in CAPABILITY_REGISTRY:
                cap_id = spec.capability_id

                if cap_id == "CAP-1":
                    # CAP-1 (wrapped; failure is captured, not propagated)
                    emit_event(
                        "capability_started",
                        capability_id="CAP-1",
                        request_id=request_id,
                        timestamp=_utc_now_iso(),
                    )
                    cap1_timer = Timer()
                    try:
                        with cap1_timer:
                            envelope, dataset_handle = invoke_cap1(
                                envelope, payload, transient_root=self._transient_root
                            )
                        envelope = envelope.with_integrity(
                            _success_record("CAP-1", duration_ms=cap1_timer.duration_ms)
                        )
                        emit_event(
                            "capability_completed",
                            capability_id="CAP-1",
                            request_id=request_id,
                            duration_ms=cap1_timer.duration_ms,
                            status="success",
                        )
                    except Exception as exc:
                        cap1_failed = True
                        envelope = envelope.with_integrity(
                            _failure_record(
                                "CAP-1", exc, duration_ms=cap1_timer.duration_ms
                            )
                        )
                        # Sibling-skip reason recorded once at the
                        # CAP-1 failure boundary; CAP-2 and CAP-7 arms
                        # only stamp their own skipped records.
                        reasons.append("CAP-1 failed; siblings skipped")
                        emit_event(
                            "capability_failed",
                            capability_id="CAP-1",
                            request_id=request_id,
                            duration_ms=cap1_timer.duration_ms,
                            status="failure",
                            error_type=type(exc).__name__,
                            error_message=str(exc)[:500],
                        )
                        logger.warning(
                            "control_plane.cap1_failed",
                            request_id=request_id,
                            error=str(exc)[:200],
                        )

                elif cap_id == "CAP-2":
                    # CAP-2 (wrapped; failure is captured, not propagated)
                    if cap1_failed:
                        envelope = envelope.with_integrity(_skipped_record("CAP-2"))
                        missing.append("CAP-2")
                        continue
                    emit_event(
                        "capability_started",
                        capability_id="CAP-2",
                        request_id=request_id,
                        timestamp=_utc_now_iso(),
                    )
                    cap2_timer = Timer()
                    try:
                        with cap2_timer:
                            envelope, rule_hits = invoke_cap2(
                                envelope, payload, dataset_handle
                            )
                        envelope = envelope.with_integrity(
                            _success_record("CAP-2", duration_ms=cap2_timer.duration_ms)
                        )
                        cap2_succeeded = True
                        emit_event(
                            "capability_completed",
                            capability_id="CAP-2",
                            request_id=request_id,
                            duration_ms=cap2_timer.duration_ms,
                            status="success",
                        )
                    except Exception as exc:
                        missing.append("CAP-2")
                        reasons.append(f"CAP-2: {type(exc).__name__}: {exc}")
                        envelope = envelope.with_integrity(
                            _failure_record(
                                "CAP-2", exc, duration_ms=cap2_timer.duration_ms
                            )
                        )
                        emit_event(
                            "capability_failed",
                            capability_id="CAP-2",
                            request_id=request_id,
                            duration_ms=cap2_timer.duration_ms,
                            status="failure",
                            error_type=type(exc).__name__,
                            error_message=str(exc)[:500],
                        )
                        logger.warning(
                            "control_plane.cap2_failed",
                            request_id=request_id,
                            error=str(exc)[:200],
                        )

                elif cap_id == "CAP-7":
                    # CAP-7 (wrapped; failure is captured, not propagated)
                    if cap1_failed:
                        envelope = envelope.with_integrity(_skipped_record("CAP-7"))
                        missing.append("CAP-7")
                        continue
                    emit_event(
                        "capability_started",
                        capability_id="CAP-7",
                        request_id=request_id,
                        timestamp=_utc_now_iso(),
                    )
                    cap7_timer = Timer()
                    try:
                        with cap7_timer:
                            envelope, verdict_from_cap7 = invoke_cap7(
                                envelope, payload, dataset_handle, layers=active_layers
                            )
                        envelope = envelope.with_integrity(
                            _success_record("CAP-7", duration_ms=cap7_timer.duration_ms)
                        )
                        cap7_succeeded = True
                        emit_event(
                            "capability_completed",
                            capability_id="CAP-7",
                            request_id=request_id,
                            duration_ms=cap7_timer.duration_ms,
                            status="success",
                        )
                    except Exception as exc:
                        missing.append("CAP-7")
                        reasons.append(f"CAP-7: {type(exc).__name__}: {exc}")
                        envelope = envelope.with_integrity(
                            _failure_record(
                                "CAP-7", exc, duration_ms=cap7_timer.duration_ms
                            )
                        )
                        emit_event(
                            "capability_failed",
                            capability_id="CAP-7",
                            request_id=request_id,
                            duration_ms=cap7_timer.duration_ms,
                            status="failure",
                            error_type=type(exc).__name__,
                            error_message=str(exc)[:500],
                        )
                        logger.warning(
                            "control_plane.cap7_failed",
                            request_id=request_id,
                            error=str(exc)[:200],
                        )

                elif cap_id == "CAP-5":
                    # Aggregate CAP-7 verdict (if any) + CAP-2 rule_hits,
                    # then build the run-level Integrity summary that CAP-5
                    # consumes as input.
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

                    # CAP-5 (propagates; not wrapped per D5)
                    emit_event(
                        "capability_started",
                        capability_id="CAP-5",
                        request_id=request_id,
                        timestamp=_utc_now_iso(),
                    )
                    cap5_timer = Timer()
                    try:
                        with cap5_timer:
                            envelope, manifest_id = invoke_cap5(
                                envelope,
                                aggregated,
                                integrity.model_dump(),
                                rubric_id=payload.rubric_id,
                                rubric_version=payload.rubric_version,
                                runs_root=runs_root,
                            )
                        envelope = envelope.with_integrity(
                            _success_record("CAP-5", duration_ms=cap5_timer.duration_ms)
                        )
                        emit_event(
                            "capability_completed",
                            capability_id="CAP-5",
                            request_id=request_id,
                            duration_ms=cap5_timer.duration_ms,
                            status="success",
                        )
                    except Exception as exc:
                        emit_event(
                            "capability_failed",
                            capability_id="CAP-5",
                            request_id=request_id,
                            duration_ms=cap5_timer.duration_ms,
                            status="failure",
                            error_type=type(exc).__name__,
                            error_message=str(exc)[:500],
                        )
                        raise

                else:
                    raise RuntimeError(
                        f"PlatformRunner has no dispatch arm for "
                        f"capability_id={cap_id!r}; CAPABILITY_REGISTRY "
                        f"and runner.py disagree."
                    )

        emit_event(
            "run_completed",
            request_id=request_id,
            duration_ms=run_timer.duration_ms,
            status="success" if integrity.complete else "partial",
        )

        return SingleEvaluationResult(
            verdict=aggregated,
            manifest_id=manifest_id,
            envelope=envelope,
            integrity=integrity,
        )
