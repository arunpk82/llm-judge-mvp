"""L1 completion integration tests (Phase 1 closure).

Asserts the four exit criteria for declaring L1 'done' through the
platform-as-harness flow — every assertion goes through the same
artifacts a human would inspect, no platform bypass:

  1. **Algorithm correctness** — L1 produces specific, deterministic
     outputs on a fixed synthetic corpus. Regression-detects any
     change to the substring-match or sentence-segmentation logic
     that would invalidate the empirical RAGTruth-50 result.

  2. **RAGTruth-50 verification PASSED** — the canonical pre-flight
     is captured as a verification_summary.json artifact written by
     ``make verify-l1`` and committed to the PR description; this
     test asserts the locally-runnable ragtruth_5 slice exhibits the
     same precision invariant (100%) the full run did.

  3. **Cascade wired** — ``DEFAULT_LAYERS`` from the wrappers module
     contains ``"L1"``, so the no-flag default path actually runs
     L1.

  4. **Sub-capability instrumentation** — running L1 emits all four
     boundary events (one CLEAN, three SOFT-with-reason) AND those
     events propagate through the platform into the per-case
     ``events.jsonl`` written by the batch runner.

The integration test uses RAGTruth-5 (5-case slice). Full RAGTruth-50
verification ran in pre-flight (``make verify-l1``); its output is
captured in the PR description, not re-run here.

Regression test #1 uses a fixed synthetic corpus — no benchmark data
needed, no spaCy variability to worry about beyond the model version
already pinned in CI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from llm_judge.calibration.hallucination import check_hallucination
from llm_judge.control_plane.event_bus import get_default_bus
from llm_judge.control_plane.wrappers import DEFAULT_LAYERS

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _clear_event_bus() -> None:
    get_default_bus().clear()
    yield
    get_default_bus().clear()


# Synthetic corpus pinned for the regression test. Sentence text is
# chosen so spaCy splits cleanly on '.' and the substring matcher's
# thresholds (exact, 0.85 SequenceMatcher, 0.80 Jaccard) produce
# unambiguous results — robust against minor spaCy / tokenizer drift.
_REGRESSION_RESPONSE = (
    "Paris is the capital of France. "
    "It is also the largest city in the country. "
    "The Sahara Desert is in fact a small lake near Berlin."
)
_REGRESSION_CONTEXT = (
    "Paris is the capital of France. "
    "Paris is the most populous city of France."
)
# Sentence 0 is a perfect substring match.
# Sentence 1 paraphrases context but doesn't substring/Jaccard-match.
# Sentence 2 is a fabrication (unrelated content).
_EXPECTED_L1_CLEARED = 1


def test_l1_algorithm_unchanged_on_regression_corpus() -> None:
    """Exit criterion 1 — algorithm correctness.

    On the fixed synthetic corpus, L1 clears exactly the sentences it
    should: the verbatim-match sentence, and only that one. Any change
    to L1's substring / Jaccard / SequenceMatcher thresholds, or to the
    spaCy splitter's behavior, will trip this test before it ships.

    No platform plumbing here — exercises ``check_hallucination``
    directly with L1-only knobs (matches the Commit 1 instrumentation
    test pattern). Frees the test from spaCy / Runner / CAP-5 cost on
    the regression path.
    """
    result = check_hallucination(
        response=_REGRESSION_RESPONSE,
        context=_REGRESSION_CONTEXT,
        case_id="regression-l1",
        skip_embeddings=True,
        l1_enabled=True,
        l2_enabled=False,
        l3_enabled=False,
        l4_enabled=False,
        gate2_routing="none",
    )
    assert result.layer_stats["L1"] == _EXPECTED_L1_CLEARED, (
        f"L1 should clear exactly {_EXPECTED_L1_CLEARED} sentence on the "
        f"regression corpus; got {result.layer_stats['L1']}. Stats: "
        f"{result.layer_stats}"
    )
    # total_sentences exposure (CAP-7 wrapper change in Commit 2).
    assert result.layer_stats["total_sentences"] == 3
    # The cleared sentence's resolved_by attribution is preserved.
    cleared = [sr for sr in result.sentence_results if sr.resolved_by == "L1"]
    assert len(cleared) == _EXPECTED_L1_CLEARED


def test_l1_in_default_layers() -> None:
    """Exit criterion 3 — cascade wired.

    Without ``--isolate-layer``, the platform routes through
    ``DEFAULT_LAYERS``. L1 must be in that tuple, otherwise the
    no-flag default path silently bypasses L1.
    """
    assert "L1" in DEFAULT_LAYERS, (
        f"L1 missing from DEFAULT_LAYERS={DEFAULT_LAYERS!r} — the "
        f"no-flag default path would skip L1."
    )


def test_l1_sub_capability_events_fire_via_check_hallucination() -> None:
    """Exit criterion 4a — instrumentation fires at the function level.

    Confirms the four L1 boundary events are emitted by
    ``check_hallucination`` regardless of whether anything wraps it.
    Complements the platform-level assertion below.
    """
    with capture_logs() as logs:
        check_hallucination(
            response=_REGRESSION_RESPONSE,
            context=_REGRESSION_CONTEXT,
            case_id="instr-l1",
            skip_embeddings=True,
            l1_enabled=True,
            l2_enabled=False,
            l3_enabled=False,
            l4_enabled=False,
            gate2_routing="none",
        )
    seen = {
        log["sub_capability_id"]
        for log in logs
        if log.get("capability_id") == "CAP-7"
        and isinstance(log.get("sub_capability_id"), str)
        and log["sub_capability_id"].startswith("l1_")
    }
    expected = {
        "l1_input_preparation",
        "l1_substring_matching",
        "l1_aggregation",
        "l1_result_emission",
    }
    assert seen == expected, f"missing L1 boundaries: {expected - seen}"


def test_l1_meets_all_four_exit_criteria_via_platform() -> None:
    """The canonical 'L1 is done' integration test.

    Runs --isolate-layer L1 on RAGTruth-5 through the platform, renders
    the verification report, then asserts each exit criterion against
    the rendered artifacts:

      1. Algorithm correctness (sentence-level): precision == 1.0 on
         the slice. The full RAGTruth-50 PASS verdict is captured in
         the PR description (pre-flight artifact); the slice
         re-confirms the precision invariant locally.
      2. Verification flow itself works: report files (HTML, MD, JSON
         summary) exist with expected structure.
      3. Cascade wired: ``verdict.layers_requested == ["L1"]`` in
         each per-case CAP-5 manifest.
      4. Instrumentation fires through the platform: per-case
         ``events.jsonl`` carries the four L1 boundary events.

    Slice strategy: Option 1 — registered ``ragtruth_5`` benchmark
    (5-id prefix of ragtruth_50). Bounded scope: one factory closure
    in the registry, one new JSON file. Option 2 (fixture batch_run)
    was the fallback and was not needed.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_batch_evaluation.py"),
        "--benchmark",
        "ragtruth_5",
        "--isolate-layer",
        "L1",
    ]
    completed = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode == 0, (
        f"batch run failed (rc={completed.returncode}):\n"
        f"STDERR: {completed.stderr[-500:]}"
    )

    runs_root = REPO_ROOT / "reports" / "batch_runs"
    batch_dir = max(
        (d for d in runs_root.iterdir() if d.name.startswith("batch-")),
        key=lambda p: p.stat().st_mtime,
    )

    # Render via the same module the Makefile invokes.
    from experiments.render_layer_verification_report import (
        render_verification_report,
    )

    output_dir = batch_dir / "verification_out"
    verdict = render_verification_report(
        batch_run_dir=batch_dir,
        single_eval_root=REPO_ROOT / "reports" / "single_eval",
        benchmark="ragtruth_5",
        layer="L1",
        target_detection=0.074,
        tolerance_detection=0.005,
        target_precision=1.0,
        output_dir=output_dir,
    )

    # ── Criterion 1 — algorithm correctness on the slice ─────────────
    summary = json.loads(
        (output_dir / "verification_summary.json").read_text(encoding="utf-8")
    )
    assert summary["metrics"]["precision"] == 1.0, (
        f"precision regressed: {summary['metrics']['precision']} "
        f"(verdict={verdict})"
    )

    # ── Criterion 2 — verification flow produced report artifacts ────
    assert (output_dir / "verification_report.html").is_file()
    assert (output_dir / "verification_report.md").is_file()
    assert (output_dir / "verification_summary.json").is_file()
    md = (output_dir / "verification_report.md").read_text(encoding="utf-8")
    for section in ("Verification summary", "Per-case attribution", "Findings"):
        assert section in md, f"verification report missing section: {section}"
    # No platform faults expected on ragtruth_5 (cases 0-4 don't trip the
    # security_warning path that previously crashed CAP-1).
    assert summary["case_inclusion"]["excluded"] == 0, (
        f"ragtruth_5 produced unexpected fault cases: "
        f"{summary['case_inclusion']['excluded_cases']}"
    )

    # ── Criterion 3 — cascade wired (verdict.layers_requested) ────────
    case_ids = sorted(d.name for d in (batch_dir / "cases").iterdir())
    assert case_ids, "no cases written"
    for cid in case_ids:
        cap5_manifest = json.loads(
            (REPO_ROOT / "reports" / "single_eval" / cid / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        layers_req = cap5_manifest["verdict"]["layers_requested"]
        assert layers_req == ["L1"], (
            f"{cid}: layers_requested={layers_req!r}, expected ['L1']"
        )

    # ── Criterion 4 — instrumentation fires via the platform ──────────
    expected_boundaries = {
        "l1_input_preparation",
        "l1_substring_matching",
        "l1_aggregation",
        "l1_result_emission",
    }
    sample_events_path = batch_dir / "cases" / case_ids[0] / "events.jsonl"
    with sample_events_path.open("r", encoding="utf-8") as f:
        sample_events = [json.loads(line) for line in f if line.strip()]
    seen_boundaries = {
        ev["sub_capability_id"]
        for ev in sample_events
        if ev.get("capability_id") == "CAP-7"
        and isinstance(ev.get("sub_capability_id"), str)
        and ev["sub_capability_id"].startswith("l1_")
    }
    assert seen_boundaries == expected_boundaries, (
        f"missing L1 boundaries in {sample_events_path}: "
        f"{expected_boundaries - seen_boundaries}"
    )

    # Closure assertion — verdict on the slice is PASS or PARTIAL,
    # never FAIL (FAIL would mean precision regression which already
    # failed Criterion 1 above; defensive belt-and-braces).
    assert verdict in ("PASS", "PARTIAL"), (
        f"slice verdict={verdict}, expected PASS or PARTIAL"
    )
