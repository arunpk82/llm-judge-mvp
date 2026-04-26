"""BatchRunner — sequential batch driver for the Control Plane.

Wraps :class:`~llm_judge.control_plane.runner.PlatformRunner` and runs
an iterable of :class:`SingleEvaluationRequest` cases one at a time.
Per-case failures are recorded but do not abort the batch.

Emits four batch lifecycle events through the shared event bus:

  * ``batch_started``        — fields: batch_id, total_cases, source
  * ``batch_case_started``   — fields: batch_id, case_id, case_index, total_cases
  * ``batch_case_completed`` — fields: batch_id, case_id, case_index, status, duration_ms
  * ``batch_completed``      — fields: batch_id, total_cases, successful, failed, error, duration_ms

Per-case artifacts:
  ``<output_dir>/cases/<case_id>/manifest.json`` — the stamped
  ProvenanceEnvelope as JSON (lightweight per-case audit trail).
  CAP-5 still writes its full manifest to its canonical
  ``state_root()/single_eval/<manifest_id>/`` location.

Batch-level artifact:
  ``<output_dir>/batch_manifest.json`` — the BatchResult dump.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from llm_judge.control_plane.envelope import CapabilityIntegrityRecord
from llm_judge.control_plane.observability import Timer, emit_event
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


class CaseResult(BaseModel):
    """Outcome of one case in a batch run."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    request_id: str
    manifest_path: Path
    integrity: list[CapabilityIntegrityRecord]
    duration_ms: float
    had_failure: bool
    had_error: bool
    error_type: str | None = None
    error_message: str | None = None


class BatchResult(BaseModel):
    """Outcome of an end-to-end batch run."""

    model_config = ConfigDict(frozen=True)

    batch_id: str
    source: str
    total_cases: int
    successful_cases: int
    failed_cases: int
    error_cases: int
    duration_ms: float
    case_results: list[CaseResult]


class BatchRunner:
    """Sequential batch driver.

    Constructed with an existing :class:`PlatformRunner`; the same
    Runner instance is invoked per case, so platform_version,
    transient_root, and runs_root are shared across the batch.

    Sequential execution only (D5). Parallel batching is future work.
    """

    def __init__(self, platform_runner: PlatformRunner) -> None:
        self._runner = platform_runner

    def run_batch(
        self,
        cases: Iterable[SingleEvaluationRequest],
        batch_id: str,
        output_dir: Path,
        source: str,
    ) -> BatchResult:
        """Run every case in ``cases`` through the PlatformRunner.

        Materialises ``cases`` into a list up front so ``total_cases``
        can be reported on ``batch_started``. Per-case failures inside
        the Runner (CAP-1/2/7) are absorbed by the Runner itself and
        surface as integrity records on the returned envelope. CAP-5
        propagation failures and any unexpected exception become a
        ``had_error=True`` CaseResult and the batch continues.

        ``source`` is a human-readable origin string, e.g.
        ``"benchmark:ragtruth_50"`` or ``"file:examples/cases.yaml"``.
        It is included in ``batch_started`` and on the returned
        BatchResult so downstream reports can show where the cases
        came from.
        """
        cases_list = list(cases)
        total = len(cases_list)

        output_dir.mkdir(parents=True, exist_ok=True)
        cases_root = output_dir / "cases"
        cases_root.mkdir(parents=True, exist_ok=True)

        emit_event(
            "batch_started",
            batch_id=batch_id,
            total_cases=total,
            source=source,
        )

        case_results: list[CaseResult] = []
        successful = 0
        failed = 0
        errored = 0

        with Timer() as batch_timer:
            for case_index, case in enumerate(cases_list):
                case_id = self._derive_case_id(case, case_index)
                emit_event(
                    "batch_case_started",
                    batch_id=batch_id,
                    case_id=case_id,
                    case_index=case_index,
                    total_cases=total,
                )

                case_dir = cases_root / case_id
                case_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = case_dir / "manifest.json"

                with Timer() as case_timer:
                    case_result, status = self._run_one_case(
                        case=case,
                        case_id=case_id,
                        manifest_path=manifest_path,
                    )

                # ``case_timer.duration_ms`` is the authoritative wall
                # clock for the case — populate the result + the event.
                case_result = case_result.model_copy(
                    update={"duration_ms": case_timer.duration_ms}
                )

                if case_result.had_error:
                    errored += 1
                elif case_result.had_failure:
                    failed += 1
                else:
                    successful += 1

                case_results.append(case_result)

                emit_event(
                    "batch_case_completed",
                    batch_id=batch_id,
                    case_id=case_id,
                    case_index=case_index,
                    status=status,
                    duration_ms=case_timer.duration_ms,
                )

        emit_event(
            "batch_completed",
            batch_id=batch_id,
            total_cases=total,
            successful=successful,
            failed=failed,
            error=errored,
            duration_ms=batch_timer.duration_ms,
        )

        batch_result = BatchResult(
            batch_id=batch_id,
            source=source,
            total_cases=total,
            successful_cases=successful,
            failed_cases=failed,
            error_cases=errored,
            duration_ms=batch_timer.duration_ms,
            case_results=case_results,
        )

        (output_dir / "batch_manifest.json").write_text(
            batch_result.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return batch_result

    def _derive_case_id(
        self, case: SingleEvaluationRequest, index: int
    ) -> str:
        """Use the request's request_id when set; otherwise fall back
        to a zero-padded positional id (``case_0000``)."""
        rid = case.request_id
        if rid:
            return rid
        return f"case_{index:04d}"

    def _run_one_case(
        self,
        *,
        case: SingleEvaluationRequest,
        case_id: str,
        manifest_path: Path,
    ) -> tuple[CaseResult, str]:
        """Run one case through the PlatformRunner.

        Returns ``(CaseResult, status)`` where ``status`` is the value
        emitted on ``batch_case_completed``: ``"success"`` (every
        capability success), ``"failure"`` (one or more tolerated
        capability failures), or ``"error"`` (CAP-5 propagated or an
        unexpected exception).
        """
        try:
            result = self._runner.run_single_evaluation(case)
        except Exception as exc:
            manifest_path.write_text("{}", encoding="utf-8")
            return (
                CaseResult(
                    case_id=case_id,
                    request_id=case.request_id or case_id,
                    manifest_path=manifest_path,
                    integrity=[],
                    duration_ms=0.0,
                    had_failure=False,
                    had_error=True,
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:500],
                ),
                "error",
            )

        had_failure = any(
            record.status == "failure" for record in result.envelope.integrity
        )
        manifest_path.write_text(
            result.envelope.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return (
            CaseResult(
                case_id=case_id,
                request_id=result.envelope.request_id,
                manifest_path=manifest_path,
                integrity=list(result.envelope.integrity),
                duration_ms=0.0,
                had_failure=had_failure,
                had_error=False,
            ),
            "failure" if had_failure else "success",
        )
