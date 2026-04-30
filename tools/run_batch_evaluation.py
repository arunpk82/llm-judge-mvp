#!/usr/bin/env python3
"""CLI driver for batch evaluation through the Control Plane Runner.

Two input modes:
  --benchmark NAME : load via adapter registry (ragtruth_50, halueval, ...)
  --file PATH      : load via BatchInputFile schema (.yaml or .json)

Optional ``--limit N`` truncates to the first N cases. Useful for the
``make demo-batch-quick`` target.

Commit 9 ships a basic console summary; Commit 10 will plug in
rich-formatted live progress, Commit 11 will plug in the aggregated
HTML report.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from typing import Iterator

from llm_judge.benchmarks import BenchmarkCase
from llm_judge.benchmarks.registry import (
    RAGTRUTH_5_BENCHMARK_PATH,
    RAGTRUTH_50_BENCHMARK_PATH,
    BenchmarkNotFoundError,
    build,
)
from llm_judge.control_plane.batch_aggregation import aggregate_batch
from llm_judge.control_plane.batch_input import BatchCase, load_batch_file
from llm_judge.control_plane.batch_runner import BatchRunner
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import (
    BenchmarkReference,
    SingleEvaluationRequest,
)
from llm_judge.datasets.benchmark_registry import register_benchmark

# Map registered adapter names → benchmark JSON definition path.
# Only benchmarks with a JSON definition file participate in CAP-1
# benchmark registration (CP-F1). Other adapters (halueval, fever, ...)
# load directly from raw data files; their per-case requests carry
# ``benchmark_reference=None`` and the envelope's benchmark provenance
# fields stay empty.
_BENCHMARK_DEFINITION_PATHS: dict[str, Path] = {
    "ragtruth_50": RAGTRUTH_50_BENCHMARK_PATH,
    "ragtruth_5": RAGTRUTH_5_BENCHMARK_PATH,
}

# pylint: disable=wrong-import-position
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _batch_html_report import render_batch_report  # noqa: E402
from _batch_terminal import BatchTerminalRenderer  # noqa: E402


def _benchmark_case_to_request(
    case: BenchmarkCase,
    benchmark_reference: BenchmarkReference | None = None,
) -> SingleEvaluationRequest:
    """Map a BenchmarkCase onto SingleEvaluationRequest.

    ``response`` ← ``request.candidate_answer``.
    ``source``   ← joined ``request.source_context`` if present;
                   else the last user-message content.
    ``request_id`` ← case_id, so per-case manifests live under the
                     case_id directory in batch_runs/.
    ``benchmark_reference`` (CP-F1): when non-None, flows onto the
    request and is stamped into envelope provenance by
    ``_cap1_lineage_tracking`` under CAP-1's allowlist.
    """
    pred = case.request
    if pred.source_context:
        source = "\n\n".join(s for s in pred.source_context if s)
    else:
        user_msgs = [m.content for m in pred.conversation if m.role == "user"]
        source = (
            user_msgs[-1] if user_msgs else pred.conversation[0].content
        )
    if not source:
        source = pred.candidate_answer  # last-resort: avoid pydantic min_length=1
    return SingleEvaluationRequest(
        response=pred.candidate_answer,
        source=source,
        rubric_id=pred.rubric_id,
        caller_id="batch-driver",
        request_id=case.case_id,
        benchmark_reference=benchmark_reference,
    )


def _batch_case_to_request(case: BatchCase) -> SingleEvaluationRequest:
    """Map a BatchCase onto SingleEvaluationRequest.

    BatchCase.source is optional in the file schema, but the Runner
    requires it. We raise here with the offending case_id so users
    can fix their input file.
    """
    if not case.source:
        raise ValueError(
            f"batch case {case.case_id!r} has no source — "
            "SingleEvaluationRequest requires a non-empty source"
        )
    return SingleEvaluationRequest(
        response=case.response,
        source=case.source,
        rubric_id="chat_quality",
        caller_id="batch-driver",
        request_id=case.case_id,
    )


def _resolve_cases(
    args: argparse.Namespace,
) -> tuple[Iterator[SingleEvaluationRequest], str]:
    """Return ``(cases_iter, source_label)`` based on the parsed args."""
    if args.benchmark:
        adapter = build(args.benchmark)
        bench_iter = adapter.load_cases()
        # CP-F1: benchmarks with a JSON definition are registered once
        # per batch via the benchmark registry. Failures (missing file,
        # content collision) raise ValueError-subclass exceptions and
        # fail the adapter at entry — no partial-batch execution.
        json_path = _BENCHMARK_DEFINITION_PATHS.get(args.benchmark)
        benchmark_ref = (
            register_benchmark(json_path) if json_path is not None else None
        )
        cases_iter: Iterator[SingleEvaluationRequest] = (
            _benchmark_case_to_request(c, benchmark_ref) for c in bench_iter
        )
        source = f"benchmark:{args.benchmark}"
    else:
        input_file = load_batch_file(Path(args.file))
        cases_iter = (_batch_case_to_request(c) for c in input_file.cases)
        source = f"file:{args.file}"

    if args.limit is not None:
        cases_iter = islice(cases_iter, args.limit)

    return cases_iter, source


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a batch evaluation through the Control Plane Runner.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--benchmark",
        type=str,
        help="Registered benchmark name (e.g. ragtruth_50, halueval).",
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to a batch input YAML/JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Truncate to the first N cases. Default: no limit.",
    )
    parser.add_argument(
        "--isolate-layer",
        type=str,
        default=None,
        choices=["L1", "L2", "L3", "L4", "L5"],
        help=(
            "Constrain CAP-7 to a single hallucination-pipeline layer. "
            "Used by the layer-by-layer verification flow "
            "(make verify-l1, etc.). Default: None (uses CAP-7 default "
            "DEFAULT_LAYERS=['L1']). NOTE: L5 is in the choice list to "
            "keep argparse stable across the level-by-level arc — it "
            "currently raises ValueError from invoke_cap7 until "
            "VALID_LAYERS is extended in Phase 5."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    try:
        cases_iter, source = _resolve_cases(args)
    except BenchmarkNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    batch_id = (
        f"batch-{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H%M%S')}"
    )
    output_dir = Path("reports/batch_runs") / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = PlatformRunner()
    batch_runner = BatchRunner(runner)

    renderer = BatchTerminalRenderer(output_dir)
    renderer.attach()

    layers = [args.isolate_layer] if args.isolate_layer else None

    try:
        result = batch_runner.run_batch(
            cases=cases_iter,
            batch_id=batch_id,
            output_dir=output_dir,
            source=source,
            layers=layers,
        )
    except Exception as exc:
        print(f"batch driver failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        renderer.detach()

    aggregation = aggregate_batch(result, events=renderer.captured_events)
    # Use a fresh Console for the report so the live-progress / structlog
    # noise the terminal renderer captured during the run doesn't bleed
    # into the static report. The renderer's record buffer is throwaway.
    render_batch_report(
        batch_result=result,
        aggregation=aggregation,
        output_dir=output_dir,
        html_path=output_dir / "aggregated_report.html",
        md_path=output_dir / "aggregated_report.md",
    )

    print()
    print(f"output: {output_dir}/")
    print(f"report: {output_dir / 'aggregated_report.html'}")
    if result.error_cases > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
