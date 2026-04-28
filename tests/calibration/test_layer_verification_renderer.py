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
    CASE_STATE_L1_VERDICT_MISSING,
    CASE_STATE_PLATFORM_FAULT,
    CASE_STATE_SUCCESS,
    CaseClassification,
    CaseEvidence,
    VerificationMetrics,
    _aggregate_metrics,
    _classify_case,
    _classify_disposition,
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
    # Three-state inclusion ledger present.
    assert "case_inclusion" in summary
    assert summary["case_inclusion"]["total"] == summary["case_inclusion"]["included"]
    assert summary["case_inclusion"]["excluded"] == 0


# ----------------------------------------------------------------------
# Three-state classification — fixture-driven, no real batch run
# ----------------------------------------------------------------------
#
# These tests build minimal batch_run + single_eval directory layouts
# under tmp_path and run the renderer against them. They exercise the
# disposition logic directly without paying spaCy startup more than
# the renderer would on its successful-case path.


def _write_envelope(case_dir: Path, *, integrity: list[dict[str, Any]]) -> None:
    """Write a per-case batch_run envelope with the supplied integrity list."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(
        json.dumps(
            {
                "request_id": case_dir.name,
                "caller_id": "test",
                "integrity": integrity,
                "capability_chain": ["CAP-1", "CAP-2", "CAP-7", "CAP-5"],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "events.jsonl").write_text("", encoding="utf-8")


def _write_cap5_manifest(
    single_eval_root: Path,
    case_id: str,
    *,
    verdict: dict[str, Any] | None,
) -> None:
    """Write a CAP-5 manifest with the supplied verdict shape (or empty)."""
    case_dir = single_eval_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "manifest.json").write_text(
        json.dumps({"verdict": verdict or {}, "manifest_id": case_id}),
        encoding="utf-8",
    )


def _benchmark_case_for_id(case_id: str) -> Any:
    """Build the smallest BenchmarkCase the renderer needs for a fault path —
    only ``request.candidate_answer`` and ``ground_truth.span_annotations``
    are read by ``_classify_disposition``, both of which can be empty for
    fault cases since classification short-circuits before reading them."""
    from llm_judge.benchmarks import BenchmarkCase, GroundTruth
    from llm_judge.schemas import Message, PredictRequest

    return BenchmarkCase(
        case_id=case_id,
        request=PredictRequest(
            conversation=[Message(role="user", content="q")],
            candidate_answer="The capital of France is Paris.",
            rubric_id="chat_quality",
        ),
        ground_truth=GroundTruth(
            response_level="pass",
            property_labels={},
            span_annotations=[],
            hallucination_types={},
        ),
    )


def test_renderer_classifies_platform_fault_when_cap1_failed(tmp_path: Path) -> None:
    case_dir = tmp_path / "batch_run" / "cases" / "case_a"
    _write_envelope(
        case_dir,
        integrity=[
            {
                "capability_id": "CAP-1",
                "status": "failure",
                "error_type": "KeyError",
                "error_message": "boom",
                "duration_ms": 1.0,
            },
            {"capability_id": "CAP-2", "status": "skipped_upstream_failure"},
            {"capability_id": "CAP-7", "status": "skipped_upstream_failure"},
        ],
    )
    # CAP-5 manifest has empty verdict (matches what the platform writes
    # when CAP-1 fails) — disposition classifier should short-circuit on
    # the envelope and not depend on the CAP-5 file.
    _write_cap5_manifest(tmp_path / "single_eval", "case_a", verdict={})

    disposition = _classify_disposition(
        case_id="case_a",
        batch_run_dir=tmp_path / "batch_run",
        single_eval_root=tmp_path / "single_eval",
        benchmark_case=_benchmark_case_for_id("case_a"),
    )
    assert disposition.state == CASE_STATE_PLATFORM_FAULT
    assert "CAP-1" in disposition.reason
    assert "failure" in disposition.reason
    assert disposition.evidence is None


def test_renderer_classifies_l1_verdict_missing_when_cap5_lacks_verdict(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "batch_run" / "cases" / "case_b"
    # CAP-1/2/7 all green — no platform fault.
    _write_envelope(
        case_dir,
        integrity=[
            {"capability_id": "CAP-1", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-2", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-7", "status": "success", "duration_ms": 1.0},
        ],
    )
    # ...but the CAP-5 verdict is empty (e.g. shape regression / older
    # manifest predating the surfacing change).
    _write_cap5_manifest(tmp_path / "single_eval", "case_b", verdict={})

    disposition = _classify_disposition(
        case_id="case_b",
        batch_run_dir=tmp_path / "batch_run",
        single_eval_root=tmp_path / "single_eval",
        benchmark_case=_benchmark_case_for_id("case_b"),
    )
    assert disposition.state == CASE_STATE_L1_VERDICT_MISSING
    assert "verdict" in disposition.reason.lower()
    assert disposition.evidence is None


def test_renderer_classifies_l1_verdict_missing_when_total_sentences_absent(
    tmp_path: Path,
) -> None:
    """Older manifests can carry a ``verdict`` dict but lack
    ``layer_stats['total_sentences']`` — that's the canonical sign the
    manifest predates the CAP-7 wrapper change. Renderer should classify
    as l1_verdict_missing rather than crash."""
    case_dir = tmp_path / "batch_run" / "cases" / "case_c"
    _write_envelope(
        case_dir,
        integrity=[
            {"capability_id": "CAP-1", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-2", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-7", "status": "success", "duration_ms": 1.0},
        ],
    )
    _write_cap5_manifest(
        tmp_path / "single_eval",
        "case_c",
        verdict={
            "risk_score": 0.0,
            "layer_stats": {"L1": 0, "L2": 0},  # no 'total_sentences'
            "layers_requested": ["L1"],
        },
    )

    disposition = _classify_disposition(
        case_id="case_c",
        batch_run_dir=tmp_path / "batch_run",
        single_eval_root=tmp_path / "single_eval",
        benchmark_case=_benchmark_case_for_id("case_c"),
    )
    assert disposition.state == CASE_STATE_L1_VERDICT_MISSING
    assert "total_sentences" in disposition.reason


def test_renderer_metrics_only_aggregate_successful_dispositions(
    tmp_path: Path,
) -> None:
    """Build a 3-case batch where one case is a platform fault and two
    succeed. Render and verify (a) the Platform Faults panel surfaces
    the fault, (b) metrics are computed only on the two successful
    cases, (c) summary JSON's ``case_inclusion`` ledger reflects the
    split.
    """
    batch_dir = tmp_path / "batch_run"
    single_eval_root = tmp_path / "single_eval"

    # Case A: success — short, clean, all-grounded response.
    _write_envelope(
        batch_dir / "cases" / "case_a",
        integrity=[
            {"capability_id": "CAP-1", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-2", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-7", "status": "success", "duration_ms": 1.0},
        ],
    )
    _write_cap5_manifest(
        single_eval_root,
        "case_a",
        verdict={
            "risk_score": 0.0,
            "grounding_ratio": 1.0,
            "layers_requested": ["L1"],
            "layer_stats": {"L1": 1, "total_sentences": 1},
            "sentence_results": [
                {
                    "sentence_idx": 0,
                    "sentence": "The capital of France is Paris.",
                    "resolved_by": "L1",
                    "detail": "exact_match",
                }
            ],
        },
    )

    # Case B: same shape — ensures aggregation across multiple cases.
    _write_envelope(
        batch_dir / "cases" / "case_b",
        integrity=[
            {"capability_id": "CAP-1", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-2", "status": "success", "duration_ms": 1.0},
            {"capability_id": "CAP-7", "status": "success", "duration_ms": 1.0},
        ],
    )
    _write_cap5_manifest(
        single_eval_root,
        "case_b",
        verdict={
            "risk_score": 0.0,
            "grounding_ratio": 1.0,
            "layers_requested": ["L1"],
            "layer_stats": {"L1": 1, "total_sentences": 1},
            "sentence_results": [
                {
                    "sentence_idx": 0,
                    "sentence": "The capital of France is Paris.",
                    "resolved_by": "L1",
                    "detail": "exact_match",
                }
            ],
        },
    )

    # Case C: platform fault — CAP-1 failed, no verdict.
    _write_envelope(
        batch_dir / "cases" / "case_c",
        integrity=[
            {
                "capability_id": "CAP-1",
                "status": "failure",
                "error_type": "KeyError",
                "error_message": 'Attempt to overwrite "message" in LogRecord',
                "duration_ms": 1.0,
            },
            {"capability_id": "CAP-2", "status": "skipped_upstream_failure"},
            {"capability_id": "CAP-7", "status": "skipped_upstream_failure"},
        ],
    )
    _write_cap5_manifest(single_eval_root, "case_c", verdict={})

    # Patch the benchmark loader to return all three synthetic cases.
    import experiments.render_layer_verification_report as mod

    monkey_cases = {
        cid: _benchmark_case_for_id(cid) for cid in ("case_a", "case_b", "case_c")
    }
    original_loader = mod._load_benchmark_cases
    mod._load_benchmark_cases = lambda _b: monkey_cases  # type: ignore[assignment]
    try:
        output_dir = tmp_path / "verification_out"
        verdict = render_verification_report(
            batch_run_dir=batch_dir,
            single_eval_root=single_eval_root,
            benchmark="synthetic",
            layer="L1",
            target_detection=0.5,
            tolerance_detection=0.5,
            target_precision=1.0,
            output_dir=output_dir,
        )
    finally:
        mod._load_benchmark_cases = original_loader  # type: ignore[assignment]

    summary = json.loads(
        (output_dir / "verification_summary.json").read_text(encoding="utf-8")
    )
    # case_a + case_b included, case_c excluded.
    inclusion = summary["case_inclusion"]
    assert inclusion["total"] == 3
    assert inclusion["included"] == 2
    assert inclusion["excluded"] == 1
    excluded_ids = [e["case_id"] for e in inclusion["excluded_cases"]]
    assert excluded_ids == ["case_c"]
    assert inclusion["excluded_cases"][0]["state"] == CASE_STATE_PLATFORM_FAULT

    # Metrics computed on 2 cases × 1 sentence each = 2 sentences total.
    assert summary["metrics"]["total_sentences"] == 2
    assert summary["metrics"]["total_responses"] == 2
    # Both sentences cleared, both grounded → TP=2, FP=0, precision=1.0.
    assert summary["metrics"]["tp"] == 2
    assert summary["metrics"]["fp"] == 0
    assert summary["metrics"]["precision"] == 1.0

    # Verdict reflects metrics on included cases only — precision passes
    # (1.0), detection rate is 1.0 which lands inside the [0.0, 1.0]
    # tolerance window we set up. Either PASS or PARTIAL is acceptable
    # depending on tolerance bounds; FAIL is not.
    assert verdict in ("PASS", "PARTIAL")
    # Findings mention the excluded case.
    assert any("excluded" in f.lower() for f in summary["findings"])

    # Report markdown includes a Platform Faults section.
    md = (output_dir / "verification_report.md").read_text(encoding="utf-8")
    assert "Platform Faults" in md
    assert "case_c" in md

