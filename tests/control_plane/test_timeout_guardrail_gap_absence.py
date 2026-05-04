"""CP-F12 timeout-enforcement gap-absence test.

Per Brief Template v1.3, reframed for Option β post-completion
denial (L1-Pkt-B Layer 1 chat checkpoint disposition):

  Original Option α framing assumed the guardrail interrupted a
  running capability and raised within the timeout window. Option β
  semantics — the chosen strategy because the Layer 1 stack is
  fully synchronous (Pre-flight 4) — let the capability run to
  completion and raise post-call once elapsed > configured budget.

Test shape:

  * CAP-1 timeout pinned to 0.1 s via a monkeypatched
    CAPABILITY_REGISTRY in :mod:`runner`.
  * ``invoke_cap1`` monkeypatched to sleep 0.3 s and return a
    valid envelope/dataset_handle pair. The capability succeeds —
    no exception raised by the body itself.
  * Run; assert GuardrailDeniedError surfaces, CAP-1 is recorded
    as a failure with error_type=GuardrailDeniedError, CAP-2 and
    CAP-7 are skipped via the existing sibling-skip path, CAP-5
    still runs.
  * Assert wall-clock elapsed ≥ the sleep duration. Under Option α
    interruption, the capability would have been cut short at
    0.1 s and elapsed would be near the timeout. The ≥ 0.3 s
    assertion locks in the Option β trade-off (no interruption,
    only post-completion denial).

If L1-Pkt-B Commit 5's TimeoutGuardrail registration were reverted,
no guardrail would deny the slow capability, the body would
succeed, CAP-1 would be marked successful, and several assertions
below would fail.

Evidence D1.2 mechanism class transition is verified: timeout
budget overshoot is materialised as a runtime denial event.

CP-F12 closure scope note: Option β closes the observability
surface — overshoot is a structured GuardrailDeniedError with a
guardrail.timeout_exceeded telemetry event. The "actually bound
resource consumption" surface (interrupt a runaway capability mid-
flight) is filed as **CP-F23 candidate** for v1.5 gap analysis.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llm_judge.control_plane.capability_registry import CapabilitySpec
from llm_judge.control_plane.envelope import ProvenanceEnvelope
from llm_judge.control_plane.guardrails import substrate as _substrate
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest


@pytest.fixture(autouse=True)
def isolated_guardrail_registry() -> Iterator[None]:
    """Snapshot/restore so the production TimeoutGuardrail
    registration survives the test run while the test itself sees
    only what it explicitly opts into."""
    saved = list(_substrate._REGISTERED_GUARDRAILS)
    try:
        yield
    finally:
        _substrate._REGISTERED_GUARDRAILS.clear()
        _substrate._REGISTERED_GUARDRAILS.extend(saved)


_FAST_TIMEOUT_REGISTRY: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(capability_id="CAP-1", sequence_position=0, timeout_seconds=0.1),
    CapabilitySpec(capability_id="CAP-2", sequence_position=1, timeout_seconds=5.0),
    CapabilitySpec(capability_id="CAP-7", sequence_position=2, timeout_seconds=30.0),
    CapabilitySpec(capability_id="CAP-5", sequence_position=3, timeout_seconds=5.0),
)


def _payload() -> SingleEvaluationRequest:
    return SingleEvaluationRequest(
        response="Paris is the capital of France.",
        source="Paris is the capital of France and its largest city.",
        rubric_id="chat_quality",
        caller_id="cp-f12-gap-absence",
    )


def _slow_invoke_cap1(
    envelope: ProvenanceEnvelope,
    payload: SingleEvaluationRequest,
    *,
    transient_root: Path | None = None,
) -> tuple[ProvenanceEnvelope, Any]:
    """Mock CAP-1 wrapper: sleeps long enough to overshoot the
    test's tight timeout, then returns a no-op envelope plus a
    placeholder dataset handle. The handle is never read because
    CAP-1 failure causes CAP-2 and CAP-7 to skip via the existing
    sibling-skip path."""
    del payload, transient_root
    time.sleep(0.3)
    return envelope, None


def test_post_completion_denial_when_capability_exceeds_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "llm_judge.control_plane.runner.CAPABILITY_REGISTRY",
        _FAST_TIMEOUT_REGISTRY,
    )
    monkeypatch.setattr(
        "llm_judge.control_plane.runner.invoke_cap1",
        _slow_invoke_cap1,
    )

    runner = PlatformRunner(
        platform_version="cp-f12-gap-absence-sha",
        transient_root=tmp_path / "transient",
        runs_root=tmp_path / "runs",
    )

    started = time.perf_counter()
    result = runner.run_single_evaluation(_payload())
    elapsed = time.perf_counter() - started

    # Option β: capability ran to completion; wall-clock elapsed is
    # at least the sleep duration. Under hypothetical Option α
    # interruption, elapsed would be near the timeout (0.1 s) and
    # this assertion would fail.
    assert elapsed >= 0.25, (
        f"Expected wall-clock elapsed >= 0.25 s (Option β: capability "
        f"runs to completion); observed {elapsed:.3f} s"
    )

    # CAP-1 marked failed; error_type identifies the guardrail denial.
    cap1_records = [
        r for r in result.envelope.integrity if r.capability_id == "CAP-1"
    ]
    assert len(cap1_records) == 1
    assert cap1_records[0].status == "failure"
    assert cap1_records[0].error_type == "GuardrailDeniedError"
    # duration_ms reflects the actual capability execution (≥ 250 ms).
    assert cap1_records[0].duration_ms is not None
    assert cap1_records[0].duration_ms >= 250.0

    # CAP-2 and CAP-7 skipped via the sibling-skip path; CAP-5 ran.
    statuses = {r.capability_id: r.status for r in result.envelope.integrity}
    assert statuses["CAP-2"] == "skipped_upstream_failure"
    assert statuses["CAP-7"] == "skipped_upstream_failure"
    assert statuses["CAP-5"] == "success"

    # Run-level integrity records the CAP-1 failure.
    assert result.integrity.complete is False
    assert "CAP-1" in result.integrity.missing_capabilities
    assert result.integrity.reason is not None
    assert "siblings skipped" in result.integrity.reason
