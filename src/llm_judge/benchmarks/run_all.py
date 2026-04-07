#!/usr/bin/env python3
"""
Benchmark Runner CLI — runs all available benchmarks and produces reports.

Usage:
    python -m llm_judge.benchmarks.run_all                    # all available
    python -m llm_judge.benchmarks.run_all --benchmark ragtruth  # single benchmark
    python -m llm_judge.benchmarks.run_all --max-cases 100       # quick test
    python -m llm_judge.benchmarks.run_all --output-dir reports/benchmarks
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_judge.benchmarks import BenchmarkAdapter
from llm_judge.benchmarks.metrics import compute_metrics
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.benchmarks.report import format_report_text, generate_report
from llm_judge.benchmarks.runner import run_benchmark


def _get_available_adapters(data_root: Path) -> dict[str, BenchmarkAdapter]:
    """Discover which benchmarks have data available on disk."""
    adapters: dict[str, BenchmarkAdapter] = {}

    # RAGTruth
    ragtruth_dir = data_root / "ragtruth"
    if (ragtruth_dir / "response.jsonl").exists():
        adapters["ragtruth"] = RAGTruthAdapter(data_dir=ragtruth_dir)

    # HaluEval
    halueval_dir = data_root / "halueval"
    if any((halueval_dir / f).exists() for f in ["qa_data.json", "general_data.json"]):
        from llm_judge.benchmarks.halueval import HaluEvalAdapter
        adapters["halueval"] = HaluEvalAdapter(data_dir=halueval_dir)

    # IFEval
    ifeval_dir = data_root / "ifeval"
    if any((ifeval_dir / f).exists() for f in ["ifeval.jsonl", "input_data.jsonl"]):
        from llm_judge.benchmarks.ifeval import IFEvalAdapter
        adapters["ifeval"] = IFEvalAdapter(data_dir=ifeval_dir)

    # ToxiGen
    toxigen_dir = data_root / "toxigen"
    if any((toxigen_dir / f).exists() for f in
           ["toxigen_data.jsonl", "toxigen.jsonl", "data.jsonl",
            "toxigen_data.json", "toxigen.csv", "data.csv"]):
        from llm_judge.benchmarks.toxigen import ToxiGenAdapter
        adapters["toxigen"] = ToxiGenAdapter(data_dir=toxigen_dir)

    # FaithDial
    faithdial_dir = data_root / "faithdial"
    if any((faithdial_dir / f).exists() for f in
           ["test.json", "test.jsonl", "faithdial_test.json", "data.json"]):
        from llm_judge.benchmarks.faithdial import FaithDialAdapter
        adapters["faithdial"] = FaithDialAdapter(data_dir=faithdial_dir)

    # Jigsaw / Civil Comments
    jigsaw_dir = data_root / "jigsaw"
    if any((jigsaw_dir / f).exists() for f in
           ["train.csv", "all_data.csv", "test.csv", "civil_comments.csv"]):
        from llm_judge.benchmarks.jigsaw import JigsawAdapter
        adapters["jigsaw"] = JigsawAdapter(data_dir=jigsaw_dir)

    # FEVER
    fever_dir = data_root / "fever"
    if any((fever_dir / f).exists() for f in
           ["paper_test.jsonl", "test.jsonl", "dev.jsonl", "paper_dev.jsonl"]):
        from llm_judge.benchmarks.fever import FEVERAdapter
        adapters["fever"] = FEVERAdapter(data_dir=fever_dir)

    # Master Ground Truth (internal — all deterministic properties)
    master_dir = data_root / "master_ground_truth"
    if (master_dir / "ground_truth.jsonl").exists():
        from llm_judge.benchmarks.master_gt import MasterGroundTruthAdapter
        adapters["master"] = MasterGroundTruthAdapter(data_dir=master_dir)

    return adapters


def main() -> None:
    parser = argparse.ArgumentParser(description="Run industry benchmark validation")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Run a single benchmark (ragtruth, halueval, ifeval, toxigen)")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Limit cases per benchmark (for quick tests)")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to evaluate (default: test)")
    parser.add_argument("--data-dir", type=str, default="datasets/benchmarks",
                        help="Root directory for benchmark datasets")
    parser.add_argument("--output-dir", type=str, default="experiments",
                        help="Directory for output reports")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding-based checks (1.4, 1.5) for faster runs")
    parser.add_argument("--with-llm", action="store_true",
                        help="Include Cat 2 LLM-based evaluation (requires GEMINI_API_KEY)")
    parser.add_argument("--gate2", type=str, default="none",
                        choices=["none", "ambiguous", "fail", "ambiguous+fail", "all"],
                        help="Gate 2 routing for Property 1.1: which cases to send to Gemini")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapters = _get_available_adapters(data_root)

    if not adapters:
        print(f"No benchmark datasets found in {data_root}/")
        print("Download datasets into:")
        print(f"  {data_root}/ragtruth/   (response.jsonl + source_info.jsonl)")
        print(f"  {data_root}/halueval/   (qa_data.json)")
        print(f"  {data_root}/ifeval/     (ifeval.jsonl)")
        print(f"  {data_root}/toxigen/    (toxigen_data.jsonl)")
        sys.exit(1)

    if args.benchmark:
        if args.benchmark not in adapters:
            available = ", ".join(adapters.keys())
            print(f"Benchmark '{args.benchmark}' not available. Available: {available}")
            sys.exit(1)
        adapters = {args.benchmark: adapters[args.benchmark]}

    print(f"Available benchmarks: {', '.join(adapters.keys())}")
    print(f"Max cases: {args.max_cases or 'all'}")
    print(f"Split: {args.split}")
    print()

    all_reports: dict[str, dict] = {}

    for name, adapter in adapters.items():
        meta = adapter.metadata()
        print(f"{'=' * 70}")
        print(f"Running: {meta.name} ({meta.test_cases} test cases)")
        print(f"Properties: {meta.supported_properties}")
        print(f"{'=' * 70}")

        # Run
        result = run_benchmark(
            adapter, split=args.split, max_cases=args.max_cases,
            skip_embeddings=args.skip_embeddings, with_llm=args.with_llm,
            gate2_routing=args.gate2,
        )

        # Compute metrics
        metrics = compute_metrics(result)

        # Print report
        report_text = format_report_text(metrics, meta)
        print(report_text)

        # Save JSON report
        report_dict = generate_report(metrics, meta)
        report_path = output_dir / f"{name}_baseline_results.json"
        with report_path.open("w") as f:
            json.dump(report_dict, f, indent=2)
        print(f"Saved: {report_path}")
        print()

        all_reports[name] = report_dict

    # Summary across all benchmarks
    if len(all_reports) > 1:
        print(f"\n{'=' * 70}")
        print("CROSS-BENCHMARK SUMMARY")
        print(f"{'=' * 70}")
        for name, report in all_reports.items():
            rl = report["response_level"]
            print(f"  {name}: F1={rl['f1']:.3f} P={rl['precision']:.3f} "
                  f"R={rl['recall']:.3f} (n={rl['total']})")

        # Merge per-property results across all benchmarks
        print("\n  Per-property (merged):")
        all_properties: dict[str, dict] = {}
        for report in all_reports.values():
            for pid, pm in report.get("per_property", {}).items():
                if pid not in all_properties:
                    all_properties[pid] = pm
                else:
                    existing = all_properties[pid]
                    # Take the one with more data
                    if pm["total"] > existing["total"]:
                        all_properties[pid] = pm

        for pid in sorted(all_properties.keys()):
            pm = all_properties[pid]
            print(f"    Property {pid}: F1={pm['f1']:.3f} P={pm['precision']:.3f} "
                  f"R={pm['recall']:.3f}")

    # Save combined summary
    summary_path = output_dir / "benchmark_summary.json"
    with summary_path.open("w") as f:
        json.dump(all_reports, f, indent=2)
    print(f"\nCombined summary: {summary_path}")


if __name__ == "__main__":
    main()
