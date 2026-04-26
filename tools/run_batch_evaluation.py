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

from llm_judge.benchmarks import BenchmarkAdapter, BenchmarkCase
from llm_judge.benchmarks.registry import BenchmarkNotFoundError
from llm_judge.benchmarks.registry import get as get_benchmark
from llm_judge.control_plane.batch_input import BatchCase, load_batch_file
from llm_judge.control_plane.batch_runner import BatchRunner
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import SingleEvaluationRequest

# Canonical RAGTruth-50 subset definition. When the user picks the
# ragtruth_50 benchmark, we narrow the adapter to this 50-case set
# rather than the full ~1170-case RAGTruth dump.
RAGTRUTH_50_BENCHMARK_PATH = Path(
    "datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json"
)


def _benchmark_case_to_request(
    case: BenchmarkCase,
) -> SingleEvaluationRequest:
    """Map a BenchmarkCase onto SingleEvaluationRequest.

    ``response`` ← ``request.candidate_answer``.
    ``source``   ← joined ``request.source_context`` if present;
                   else the last user-message content.
    ``request_id`` ← case_id, so per-case manifests live under the
                     case_id directory in batch_runs/.
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
        caller_id="batch-driver",
        request_id=case.case_id,
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
        caller_id="batch-driver",
        request_id=case.case_id,
    )


def _build_benchmark_adapter(name: str) -> BenchmarkAdapter:
    """Instantiate via the registry; apply ragtruth_50 filtering."""
    adapter_class = get_benchmark(name)
    adapter = adapter_class()
    if name == "ragtruth_50" and RAGTRUTH_50_BENCHMARK_PATH.exists():
        # The class is RAGTruthAdapter; the filter restricts load_cases
        # to the canonical 50-id subset. Without it the adapter would
        # iterate the full ~1170-case dump.
        set_filter = getattr(adapter, "set_benchmark_filter", None)
        if callable(set_filter):
            set_filter(RAGTRUTH_50_BENCHMARK_PATH)
    return adapter


def _resolve_cases(
    args: argparse.Namespace,
) -> tuple[Iterator[SingleEvaluationRequest], str]:
    """Return ``(cases_iter, source_label)`` based on the parsed args."""
    if args.benchmark:
        adapter = _build_benchmark_adapter(args.benchmark)
        bench_iter = adapter.load_cases()
        cases_iter: Iterator[SingleEvaluationRequest] = (
            _benchmark_case_to_request(c) for c in bench_iter
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

    runner = PlatformRunner()
    batch_runner = BatchRunner(runner)

    try:
        result = batch_runner.run_batch(
            cases=cases_iter,
            batch_id=batch_id,
            output_dir=output_dir,
            source=source,
        )
    except Exception as exc:
        print(f"batch driver failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    print()
    print(f"Batch {batch_id} complete:")
    print(f"  source:     {source}")
    print(f"  successful: {result.successful_cases}/{result.total_cases}")
    print(f"  failed:     {result.failed_cases}")
    print(f"  errored:    {result.error_cases}")
    print(f"  duration:   {result.duration_ms:.0f}ms")
    print(f"  output:     {output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
