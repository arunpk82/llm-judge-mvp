"""Tests for tools/_batch_html_report.py.

Exercises the report renderer against synthetic BatchResult /
BatchAggregation fixtures so we can pin the honest-gap labels and
section coverage without going through the full Runner.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# tools/ is not a package; expose its modules to the test module.
_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(_TOOLS_DIR))

from _batch_html_report import render_batch_report  # noqa: E402

from llm_judge.control_plane.batch_aggregation import (  # noqa: E402
    BatchAggregation,
    aggregate_batch,
)
from llm_judge.control_plane.batch_runner import (  # noqa: E402
    BatchResult,
    CaseResult,
)
from llm_judge.control_plane.envelope import (  # noqa: E402
    CapabilityIntegrityRecord,
)


def _success(cap: str, ms: float) -> CapabilityIntegrityRecord:
    return CapabilityIntegrityRecord(
        capability_id=cap, status="success", duration_ms=ms
    )


def _case(case_id: str, ms: float, *, fail: bool = False) -> CaseResult:
    integrity = [
        _success("CAP-1", 10.0),
        _success("CAP-2", 5.0)
        if not fail
        else CapabilityIntegrityRecord(
            capability_id="CAP-2",
            status="failure",
            error_type="Boom",
            error_message="injected",
            duration_ms=2.0,
        ),
        _success("CAP-7", 100.0),
        _success("CAP-5", 8.0),
    ]
    return CaseResult(
        case_id=case_id,
        request_id=case_id,
        manifest_path=Path("/tmp/manifest.json"),
        integrity=integrity,
        duration_ms=ms,
        had_failure=fail,
        had_error=False,
    )


@pytest.fixture
def synthetic_batch() -> BatchResult:
    cases = [
        _case("synthetic_001", 100.0),
        _case("synthetic_002", 200.0, fail=True),
        _case("synthetic_003", 300.0),
    ]
    return BatchResult(
        batch_id="batch-test",
        source="benchmark:test_synthetic",
        total_cases=len(cases),
        successful_cases=2,
        failed_cases=1,
        error_cases=0,
        duration_ms=600.0,
        case_results=cases,
    )


@pytest.fixture
def synthetic_aggregation(synthetic_batch: BatchResult) -> BatchAggregation:
    # Synthetic event log mirroring three runs through CAP-1/2/5 sub-caps.
    events = []
    for case_id in ("synthetic_001", "synthetic_002", "synthetic_003"):
        for sub in ("reception", "validation", "registration", "hashing", "lineage_tracking"):
            events.append(
                {
                    "event": "sub_capability_started",
                    "capability_id": "CAP-1",
                    "sub_capability_id": sub,
                    "request_id": case_id,
                }
            )
            events.append(
                {
                    "event": "sub_capability_completed",
                    "capability_id": "CAP-1",
                    "sub_capability_id": sub,
                    "request_id": case_id,
                    "duration_ms": 1.0,
                    "status": "success",
                }
            )
        events.append(
            {
                "event": "sub_capability_skipped",
                "capability_id": "CAP-1",
                "sub_capability_id": "discovery",
                "request_id": case_id,
                "reason": "single_eval_does_not_query_registry",
            }
        )
    return aggregate_batch(synthetic_batch, events=events)


def test_html_and_md_files_written_with_substantial_content(
    tmp_path: Path,
    synthetic_batch: BatchResult,
    synthetic_aggregation: BatchAggregation,
) -> None:
    out = tmp_path / "batch_runs" / synthetic_batch.batch_id
    out.mkdir(parents=True)
    html_path = out / "aggregated_report.html"
    md_path = out / "aggregated_report.md"

    render_batch_report(
        batch_result=synthetic_batch,
        aggregation=synthetic_aggregation,
        output_dir=out,
        html_path=html_path,
        md_path=md_path,
    )

    assert html_path.is_file()
    assert md_path.is_file()
    # Both forms should have substantial content.
    assert html_path.stat().st_size > 5_000
    assert md_path.stat().st_size > 1_000


def test_md_report_contains_all_section_headers(
    tmp_path: Path,
    synthetic_batch: BatchResult,
    synthetic_aggregation: BatchAggregation,
) -> None:
    out = tmp_path / "report_sections"
    out.mkdir()
    html_path = out / "r.html"
    md_path = out / "r.md"
    render_batch_report(
        batch_result=synthetic_batch,
        aggregation=synthetic_aggregation,
        output_dir=out,
        html_path=html_path,
        md_path=md_path,
    )
    md = md_path.read_text(encoding="utf-8")
    for section in (
        "Batch report",
        "Summary",
        "Per-capability rollups",
        "CAP-1 — Dataset Ingestion sub-capabilities",
        "CAP-2 — Rule Evaluation sub-capabilities",
        "CAP-5 — Artifact Governance sub-capabilities",
        "CAP-7 — Hallucination Detection",  # deferred panel
        "Per-case results",
        "Artifacts",
    ):
        assert section in md, f"missing section header: {section!r}"


def test_md_report_renders_honest_gap_labels(
    tmp_path: Path,
    synthetic_batch: BatchResult,
    synthetic_aggregation: BatchAggregation,
) -> None:
    out = tmp_path / "honest_gaps"
    out.mkdir()
    html_path = out / "r.html"
    md_path = out / "r.md"
    render_batch_report(
        batch_result=synthetic_batch,
        aggregation=synthetic_aggregation,
        output_dir=out,
        html_path=html_path,
        md_path=md_path,
    )
    md = md_path.read_text(encoding="utf-8")

    # CAP-7 deferred-instrumentation rationale must surface verbatim.
    assert "deferred to the" in md
    assert "CAP-7 completion packet" in md

    # CAP-1.discovery (skipped) should render with its reason note.
    assert "discovery" in md
    assert "not engaged on" in md or "single_eval_does_not_query_registry" in md

    # Horizontals should be 0/N or render the "not wired" tag in some form.
    assert "0/3" in md or "not wired" in md


def test_md_report_includes_every_case_row(
    tmp_path: Path,
    synthetic_batch: BatchResult,
    synthetic_aggregation: BatchAggregation,
) -> None:
    out = tmp_path / "case_rows"
    out.mkdir()
    html_path = out / "r.html"
    md_path = out / "r.md"
    render_batch_report(
        batch_result=synthetic_batch,
        aggregation=synthetic_aggregation,
        output_dir=out,
        html_path=html_path,
        md_path=md_path,
    )
    md = md_path.read_text(encoding="utf-8")

    for case in synthetic_batch.case_results:
        # case_id may render truncated by Rich's column wrapping —
        # check with the unique numeric suffix instead.
        suffix = case.case_id.split("_")[-1][:3]
        assert suffix in md, f"missing case row for {case.case_id}"
