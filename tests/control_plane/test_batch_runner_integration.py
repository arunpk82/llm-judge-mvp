"""Integration tests for BatchRunner.

These tests exercise the full ``BatchRunner`` → ``PlatformRunner``
pipeline against synthetic ``SingleEvaluationRequest`` payloads
(no benchmark data on disk required). They cover:

  * happy path — every case succeeds, every per-case manifest is written
  * mixed outcomes — CAP-2 monkeypatched to fail on alternate cases,
    confirms ``had_failure`` propagates and the batch keeps running
  * unexpected exception in PlatformRunner — case is marked errored,
    batch continues
  * batch lifecycle event ordering — batch_started, then per-case
    started/completed pairs, then batch_completed

Tests are isolated by clearing the default event bus between cases
and by writing all artifacts under tmp_path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane.batch_runner import BatchResult, BatchRunner
from llm_judge.control_plane.event_bus import get_default_bus
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture(autouse=True)
def _clear_event_bus() -> None:
    """Ensure no leaked subscribers between integration tests."""
    get_default_bus().clear()
    yield
    get_default_bus().clear()


@pytest.fixture
def batch_runner(tmp_path: Path) -> BatchRunner:
    runner = PlatformRunner(
        platform_version="batch-integration-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )
    return BatchRunner(runner)


def _payload(i: int) -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response=f"Case {i} candidate response with a fact.",
        source=f"Case {i} source document containing the same fact.",
        caller_id="batch-integration-test",
        request_id=f"int-case-{i:03d}",
    )


def test_happy_path_five_cases_all_succeed(
    batch_runner: BatchRunner, tmp_path: Path
) -> None:
    output_dir = tmp_path / "batch_runs" / "happy"
    cases = [_payload(i) for i in range(5)]

    result = batch_runner.run_batch(
        cases=cases,
        batch_id="happy",
        output_dir=output_dir,
        source="benchmark:integration_synthetic",
    )

    assert isinstance(result, BatchResult)
    assert result.total_cases == 5
    assert result.successful_cases == 5
    assert result.failed_cases == 0
    assert result.error_cases == 0
    assert len(result.case_results) == 5

    # Per-case manifests written under cases/<case_id>/manifest.json.
    for case_result in result.case_results:
        assert case_result.manifest_path.is_file(), (
            f"missing manifest for {case_result.case_id}"
        )
        # Envelope JSON has a request_id field.
        contents = case_result.manifest_path.read_text(encoding="utf-8")
        assert case_result.request_id in contents

    # Batch manifest written at the top-level output_dir.
    assert (output_dir / "batch_manifest.json").is_file()


def test_mixed_outcomes_failure_does_not_abort_batch(
    batch_runner: BatchRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject a CAP-2 failure on every other case; the batch must
    still finish and the failure count must be exact."""
    from llm_judge.control_plane import runner as runner_module

    real_invoke_cap2 = runner_module.invoke_cap2
    call_counter = {"n": 0}

    def _maybe_boom(*args: Any, **kwargs: Any) -> Any:
        call_counter["n"] += 1
        if call_counter["n"] % 2 == 0:
            raise RuntimeError("integration test injected CAP-2 failure")
        return real_invoke_cap2(*args, **kwargs)

    monkeypatch.setattr(
        runner_module, "invoke_cap2", _maybe_boom
    )

    output_dir = tmp_path / "batch_runs" / "mixed"
    cases = [_payload(i) for i in range(4)]
    result = batch_runner.run_batch(
        cases=cases,
        batch_id="mixed",
        output_dir=output_dir,
        source="benchmark:integration_synthetic",
    )

    assert result.total_cases == 4
    # Cases 0, 2 succeed; 1, 3 fail (injected CAP-2 failure).
    assert result.successful_cases == 2
    assert result.failed_cases == 2
    assert result.error_cases == 0

    failed_results = [c for c in result.case_results if c.had_failure]
    assert len(failed_results) == 2
    for case_result in failed_results:
        # The injected failure should leave a CAP-2 failure record on the envelope.
        cap2_records = [
            r for r in case_result.integrity if r.capability_id == "CAP-2"
        ]
        assert cap2_records, f"missing CAP-2 record for {case_result.case_id}"
        assert cap2_records[0].status == "failure"


def test_runner_exception_marks_case_errored(
    batch_runner: BatchRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If PlatformRunner.run_single_evaluation itself raises (e.g.
    CAP-5 propagation), the case is recorded as errored and the
    batch continues to the next case."""

    def _boom(self: Any, payload: Any, **kwargs: Any) -> Any:
        raise RuntimeError("integration test injected PlatformRunner failure")

    monkeypatch.setattr(
        PlatformRunner, "run_single_evaluation", _boom
    )

    output_dir = tmp_path / "batch_runs" / "errored"
    cases = [_payload(i) for i in range(3)]
    result = batch_runner.run_batch(
        cases=cases,
        batch_id="errored",
        output_dir=output_dir,
        source="benchmark:integration_synthetic",
    )

    assert result.total_cases == 3
    assert result.successful_cases == 0
    assert result.failed_cases == 0
    assert result.error_cases == 3
    for case_result in result.case_results:
        assert case_result.had_error
        assert case_result.error_type == "RuntimeError"


def test_batch_lifecycle_event_ordering(
    batch_runner: BatchRunner, tmp_path: Path
) -> None:
    """batch_started → (case_started, case_completed)*N → batch_completed."""
    output_dir = tmp_path / "batch_runs" / "ordered"
    cases = [_payload(i) for i in range(3)]

    with capture_logs() as logs:
        batch_runner.run_batch(
            cases=cases,
            batch_id="ordered",
            output_dir=output_dir,
            source="benchmark:integration_synthetic",
        )

    seq = [log["event"] for log in logs if str(log.get("event", "")).startswith("batch_")]
    assert seq[0] == "batch_started", f"first batch_* event: {seq[0]}"
    assert seq[-1] == "batch_completed", f"last batch_* event: {seq[-1]}"

    # Per case: case_started followed by case_completed (ignoring any
    # intervening non-batch_ events filtered above).
    inner = seq[1:-1]
    assert len(inner) == 2 * len(cases)
    for i in range(0, len(inner), 2):
        assert inner[i] == "batch_case_started"
        assert inner[i + 1] == "batch_case_completed"
