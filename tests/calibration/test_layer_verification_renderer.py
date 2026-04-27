"""Tests for the platform-as-harness verification flow (Commit 2).

Covers the three pieces that ship together:

  1. ``--isolate-layer`` argparse flag in ``tools/run_batch_evaluation.py``
     plumbs through to CAP-7 such that the layer constraint is observable
     in batch_run + single_eval artifacts.
  2. ``experiments/render_layer_verification_report.py`` reads only
     existing artifacts (no pipeline re-invocation), classifies each
     sentence into TP/FP/TN/FN, and renders HTML + Markdown.
  3. PASS / FAIL / PARTIAL verdict logic on synthetic metrics — no
     pipeline involvement.

The plumbing test uses the ``ragtruth_5`` slice (5-case prefix of
``ragtruth_50``) so it runs in well under a minute. Full RAGTruth-50
verification is the pre-flight artifact captured in the PR — not a
test-suite obligation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
# experiments/ isn't a Python package — make its modules importable for
# the test process the same way tools/run_batch_evaluation.py does for
# its sibling helpers (sys.path shim at script entry).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.render_layer_verification_report import (  # noqa: E402
    CaseClassification,
    CaseEvidence,
    VerificationMetrics,
    _aggregate_metrics,
    _classify_case,
    _compute_verdict,
    render_verification_report,
)
from llm_judge.control_plane.event_bus import get_default_bus  # noqa: E402


@pytest.fixture(autouse=True)
def _clear_event_bus() -> None:
    """Avoid event-bus subscriber leakage between tests."""
    get_default_bus().clear()
    yield
    get_default_bus().clear()


# ----------------------------------------------------------------------
# Verdict logic — synthetic metrics, no pipeline
# ----------------------------------------------------------------------


def _metrics(*, tp: int, fp: int, tn: int, fn: int) -> VerificationMetrics:
    total = tp + fp + tn + fn
    detection = (tp + fp) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return VerificationMetrics(
        total_sentences=total,
        total_responses=10,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        detection_rate=detection,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def test_verdict_pass_when_in_tolerance_and_at_target_precision() -> None:
    metrics = _metrics(tp=22, fp=0, tn=18, fn=263)
    verdict, findings = _compute_verdict(
        metrics,
        target_detection=0.074,
        tolerance_detection=0.005,
        target_precision=1.0,
    )
    assert verdict == "PASS", f"expected PASS, got {verdict} ({findings})"
    assert any("Precision" in line for line in findings)
    assert any("Detection rate" in line for line in findings)


def test_verdict_fail_when_precision_below_target() -> None:
    metrics = _metrics(tp=20, fp=2, tn=18, fn=263)
    verdict, findings = _compute_verdict(
        metrics,
        target_detection=0.074,
        tolerance_detection=0.005,
        target_precision=1.0,
    )
    assert verdict == "FAIL"
    assert any("below target" in line for line in findings)


def test_verdict_partial_when_precision_passes_but_detection_outside_tolerance() -> None:
    metrics = _metrics(tp=33, fp=0, tn=20, fn=250)
    verdict, findings = _compute_verdict(
        metrics,
        target_detection=0.074,
        tolerance_detection=0.005,
        target_precision=1.0,
    )
    assert verdict == "PARTIAL"
    assert any("OUTSIDE tolerance" in line for line in findings)


# ----------------------------------------------------------------------
# Renderer classification — synthetic CaseEvidence
# ----------------------------------------------------------------------


def _evidence_from_sentences(
    sentences: list[str],
    *,
    cleared_indices: set[int],
    span_texts: list[str],
    case_id: str = "synthetic_0",
) -> CaseEvidence:
    """Build a CaseEvidence whose response splits into the supplied sentences.

    The renderer re-invokes ``_split_sentences`` on the response, so the
    response must be a deterministic concatenation that yields exactly
    these sentences after spaCy splitting + the >10-char filter.
    """
    response = " ".join(s.strip() for s in sentences) + " "
    return CaseEvidence(
        case_id=case_id,
        response_text=response,
        sentence_count=len(sentences),
        layers_requested=["L1"],
        layer_stats={"L1": len(cleared_indices), "total_sentences": len(sentences)},
        cleared_indices=cleared_indices,
        sub_capability_events=[],
        span_annotations=[
            {"start": 0, "end": len(t), "text": t, "label_type": "Evident Baseless Info"}
            for t in span_texts
        ],
    )


def test_classify_case_tp_when_cleared_sentence_has_no_overlapping_span() -> None:
    sentences = [
        "The capital of France is Paris.",
        "It is the largest metropolitan area in the country.",
    ]
    evidence = _evidence_from_sentences(
        sentences, cleared_indices={0}, span_texts=[]
    )
    classification = _classify_case(evidence)
    # idx 0 cleared and grounded → TP; idx 1 uncleared and grounded → FN
    assert classification.tp == 1
    assert classification.fp == 0
    assert classification.tn == 0
    assert classification.fn == 1


def test_classify_case_fp_when_cleared_sentence_contains_a_span() -> None:
    sentences = [
        "The capital of France is Paris.",
        "Paris was founded in the year nineteen ninety nine.",
    ]
    # L1 incorrectly clears sentence 1 even though it contains a hallucinated
    # span ("nineteen ninety nine") — that's a false positive at L1.
    evidence = _evidence_from_sentences(
        sentences,
        cleared_indices={1},
        span_texts=["nineteen ninety nine"],
    )
    classification = _classify_case(evidence)
    assert classification.fp == 1, (
        f"expected FP=1, got {classification.fp}; tp={classification.tp}, "
        f"tn={classification.tn}, fn={classification.fn}"
    )
    # idx 0 was uncleared and grounded → FN
    assert classification.fn == 1


def test_aggregate_metrics_sums_across_cases_and_handles_zero_division() -> None:
    classifications = [
        CaseClassification(case_id="a", sentence_count=3, tp=2, fp=0, tn=0, fn=1),
        CaseClassification(case_id="b", sentence_count=3, tp=0, fp=0, tn=3, fn=0),
    ]
    metrics = _aggregate_metrics(classifications)
    assert metrics.tp == 2
    assert metrics.fp == 0
    assert metrics.tn == 3
    assert metrics.fn == 1
    # total = 6; detection = 2/6 = 0.333...
    assert metrics.detection_rate == pytest.approx(2 / 6)
    # precision = 2/2 = 1.0; recall = 2/3 = 0.667
    assert metrics.precision == pytest.approx(1.0)
    assert metrics.recall == pytest.approx(2 / 3)


def test_aggregate_metrics_zero_total_does_not_divide_by_zero() -> None:
    metrics = _aggregate_metrics([])
    assert metrics.detection_rate == 0.0
    assert metrics.precision == 0.0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


# ----------------------------------------------------------------------
# CLI plumbing — small ragtruth_5 slice through the platform
# ----------------------------------------------------------------------


def _run_batch_isolated(layer: str, batch_root: Path) -> Path:
    """Invoke run_batch_evaluation.py with --isolate-layer and return the run dir.

    The script writes its run directory to stdout in the form
    ``output: reports/batch_runs/<id>/``. The CWD-relative path is
    normalised to ``REPO_ROOT`` so the test passes regardless of the
    pytest invocation directory.
    """
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_batch_evaluation.py"),
        "--benchmark",
        "ragtruth_5",
        "--isolate-layer",
        layer,
    ]
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"run_batch_evaluation.py failed (rc={result.returncode}):\n"
        f"STDOUT: {result.stdout[-500:]}\n"
        f"STDERR: {result.stderr[-500:]}"
    )
    # Find the latest run dir under reports/batch_runs.
    runs_root = REPO_ROOT / "reports" / "batch_runs"
    candidates = sorted(
        (d for d in runs_root.iterdir() if d.name.startswith("batch-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    assert candidates, f"no batch_runs created under {runs_root}"
    return candidates[0]


def test_isolate_layer_l1_constrains_cap7_visible_in_artifacts(
    tmp_path: Path,
) -> None:
    """End-to-end: --isolate-layer L1 → events.jsonl shows L1 fired,
    CAP-5 manifest shows layers_requested == ['L1'] and L2/L3/L4 = 0.

    Uses ragtruth_5 (5-case slice). Tagged ``slow`` because it loads
    spaCy + writes platform artifacts; expect ~10-30s.
    """
    run_dir = _run_batch_isolated("L1", tmp_path)
    case_dirs = sorted((run_dir / "cases").iterdir())
    assert case_dirs, f"no per-case dirs under {run_dir / 'cases'}"

    case_id = case_dirs[0].name
    events_path = case_dirs[0] / "events.jsonl"
    cap5_manifest_path = REPO_ROOT / "reports" / "single_eval" / case_id / "manifest.json"

    # 1. events.jsonl shows L1 sub_capability events fired.
    with events_path.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    cap7_events = [e for e in events if e.get("capability_id") == "CAP-7"]
    assert any(
        e.get("event") == "sub_capability_started"
        and e.get("sub_capability_id") == "l1_substring_matching"
        for e in cap7_events
    ), "expected CAP-7 l1_substring_matching sub_capability_started event"

    # 2. CAP-5 manifest shows isolation: layers_requested == ['L1'], L2/L3/L4 zero.
    cap5_manifest = json.loads(cap5_manifest_path.read_text(encoding="utf-8"))
    verdict = cap5_manifest["verdict"]
    assert verdict["layers_requested"] == ["L1"], (
        f"expected layers_requested=['L1'], got {verdict['layers_requested']}"
    )
    layer_stats = verdict["layer_stats"]
    assert layer_stats["L2"] == 0
    assert layer_stats["L3_deberta"] == 0
    assert layer_stats["L4_supported"] == 0
    assert layer_stats["L4_unsupported"] == 0
    # 3. New verdict surfaces: total_sentences and sentence_results both present.
    assert "total_sentences" in layer_stats
    assert "sentence_results" in verdict
    assert isinstance(verdict["sentence_results"], list)


def test_isolate_layer_invalid_value_rejected_by_argparse() -> None:
    """Argparse rejects layer values outside the L1-L5 choice list."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_batch_evaluation.py"),
        "--benchmark",
        "ragtruth_5",
        "--isolate-layer",
        "L9",
    ]
    result = subprocess.run(
        cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=30
    )
    assert result.returncode != 0, "expected non-zero exit for invalid layer"
    # argparse writes the choice error to stderr.
    assert "L9" in result.stderr or "invalid choice" in result.stderr.lower()


# ----------------------------------------------------------------------
# Renderer end-to-end — runs only after _run_batch_isolated has produced
# artifacts. Co-located here (rather than its own test) because spinning
# up another full batch in a fresh test would double the wall time on a
# path that already covers the same plumbing.
# ----------------------------------------------------------------------


def test_render_verification_report_produces_html_md_and_summary(
    tmp_path: Path,
) -> None:
    """The renderer produces three on-disk artifacts and a non-FAIL verdict
    on the ragtruth_5 slice (expected: PARTIAL — precision 100%, detection
    out of tolerance because the slice is too small)."""
    run_dir = _run_batch_isolated("L1", tmp_path)

    output_dir = tmp_path / "verification_out"
    verdict = render_verification_report(
        batch_run_dir=run_dir,
        single_eval_root=REPO_ROOT / "reports" / "single_eval",
        benchmark="ragtruth_5",
        layer="L1",
        target_detection=0.074,
        tolerance_detection=0.005,
        target_precision=1.0,
        output_dir=output_dir,
    )
    # On the 5-case slice, precision must still be 100% — algorithm is
    # the same. Verdict is PASS or PARTIAL (not FAIL).
    assert verdict in ("PASS", "PARTIAL"), (
        f"unexpected verdict {verdict} on ragtruth_5 — precision should "
        f"hold at 100% even on the slice"
    )

    assert (output_dir / "verification_report.html").is_file()
    assert (output_dir / "verification_report.md").is_file()
    summary_path = output_dir / "verification_summary.json"
    assert summary_path.is_file()
    summary: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["layer"] == "L1"
    assert summary["benchmark"] == "ragtruth_5"
    assert summary["verdict"] == verdict
    assert summary["metrics"]["precision"] == 1.0
