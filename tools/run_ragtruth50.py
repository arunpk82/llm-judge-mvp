#!/usr/bin/env python3
"""
RAGTruth-50 Operator CLI — preseed, benchmark, funnel.

Usage:
    python tools/run_ragtruth50.py preseed          # seed graph cache from Exp 31
    python tools/run_ragtruth50.py benchmark         # run RAGTruth-50 benchmark
    python tools/run_ragtruth50.py funnel            # print funnel from last run
    python tools/run_ragtruth50.py all               # preseed → benchmark → funnel

Environment:
    GEMINI_API_KEY  — required for benchmark (L3 fact-counting, L4 Gemini)
    BENCHMARK_MAX   — optional, limit cases (default: all 50)

Output:
    results/ragtruth50_results.json   — benchmark results
    results/ragtruth50_funnel.txt     — funnel report
    .cache/hallucination_graphs/      — graph cache directory
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logger = logging.getLogger("ragtruth50")


# =====================================================================
# Preseed
# =====================================================================


def cmd_preseed(args: argparse.Namespace) -> bool:
    """Seed graph cache from Exp 31 fact tables + RAGTruth sources."""
    from llm_judge.calibration.graph_cache import GraphCache, preseed_from_exp31

    exp31_path = REPO_ROOT / "experiments" / "exp31_multipass_fact_tables.json"
    if not exp31_path.exists():
        logger.error(f"Exp 31 data not found: {exp31_path}")
        return False

    # Load source texts from RAGTruth
    source_file = REPO_ROOT / "datasets" / "benchmarks" / "ragtruth" / "source_info.jsonl"
    response_file = REPO_ROOT / "datasets" / "benchmarks" / "ragtruth" / "response.jsonl"
    benchmark_file = (
        REPO_ROOT / "datasets" / "benchmarks" / "ragtruth" / "ragtruth_50_benchmark.json"
    )

    if not all(f.exists() for f in [source_file, response_file, benchmark_file]):
        logger.error("RAGTruth data files missing. Check datasets/benchmarks/ragtruth/")
        return False

    # Load benchmark definition
    with open(benchmark_file) as f:
        benchmark = json.load(f)
    valid_ids = {str(rid) for rid in benchmark["response_ids"]}

    # Load sources
    sources: dict[str, str] = {}
    with open(source_file) as f:
        for line in f:
            item = json.loads(line)
            sources[str(item["source_id"])] = item.get("source_info", "")

    # Build case_id → source_text mapping
    # Exp 31 uses "ragtruth_{response_id}" as case_id
    source_texts: dict[str, str] = {}
    with open(response_file) as f:
        for line in f:
            item = json.loads(line)
            rid = str(item["id"])
            if rid in valid_ids:
                sid = str(item["source_id"])
                case_id = f"ragtruth_{rid}"
                source_texts[case_id] = sources.get(sid, "")

    # Create cache and preseed
    cache_dir = args.cache_dir or str(REPO_ROOT / ".cache" / "hallucination_graphs")
    cache = GraphCache(Path(cache_dir), ttl_hours=args.ttl_hours)

    print(f"Pre-seeding graph cache from {exp31_path.name}...")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Source texts available: {len(source_texts)}")

    result = preseed_from_exp31(cache, exp31_path, source_texts)

    print(f"\n  Total cases in Exp 31:    {result.get('total_cases', '?')}")
    print(f"  Unique sources seeded:    {result.get('unique_sources_seeded', '?')}")
    print(f"  Dedup savings:            {result.get('dedup_savings', '?')}")
    print(f"  Skipped (no source text): {result.get('skipped_no_source_text', '?')}")
    print(f"\n  Cache stats: {cache.stats()}")
    return True


# =====================================================================
# Benchmark
# =====================================================================


def cmd_benchmark(args: argparse.Namespace) -> bool:
    """Run RAGTruth-50 through the benchmark runner."""
    import os

    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
    from llm_judge.benchmarks.runner import run_benchmark
    from llm_judge.calibration.pipeline_config import get_pipeline_config

    if not os.environ.get("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not set — L3 fact-counting and L4 will fail")

    max_cases = args.max_cases or int(os.environ.get("BENCHMARK_MAX", "0")) or None
    config = get_pipeline_config()

    adapter = RAGTruthAdapter()
    print("Running RAGTruth-50 benchmark...")
    print(f"  Config: l3_method={config.l3_method}")
    print(f"  Max cases: {max_cases or 'all'}")
    print(f"  gate2_routing: {args.gate2_routing}")

    start = time.time()
    result = run_benchmark(
        adapter,
        split="test",
        max_cases=max_cases,
        gate2_routing=args.gate2_routing,
        config=config,
    )
    elapsed = time.time() - start

    # Save results
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    results_path = results_dir / "ragtruth50_results.json"

    summary = {
        "benchmark": result.benchmark_name,
        "cases_evaluated": result.cases_evaluated,
        "elapsed_seconds": round(elapsed, 1),
        "properties_executed": result.properties_executed,
        "properties_skipped": result.properties_skipped,
        "errors_count": len(result.errors),
        "fire_rates": result.fire_rates,
        "diagnostic_results": result.diagnostic_results,
        "response_level_results": result.response_level_results,
    }

    results_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  Results saved: {results_path}")
    print(f"  Cases: {result.cases_evaluated}, Elapsed: {elapsed:.1f}s")
    print(f"  Errors: {len(result.errors)}")

    if result.errors:
        print("\n  First 3 errors:")
        for e in result.errors[:3]:
            print(f"    {e.get('case_id', '?')}: {e.get('error', '?')[:80]}")

    return True


# =====================================================================
# Funnel
# =====================================================================


def cmd_funnel(args: argparse.Namespace) -> bool:
    """Build and print funnel report from last benchmark results."""
    results_path = REPO_ROOT / "results" / "ragtruth50_results.json"
    if not results_path.exists():
        logger.error(
            f"No results found at {results_path}. Run 'benchmark' first."
        )
        return False

    with open(results_path) as f:
        data = json.load(f)

    response_results = data.get("response_level_results", [])
    if not response_results:
        print("No response-level results found. Run benchmark with gate2_routing=pass.")
        return False

    # Aggregate layer stats across all responses
    agg: dict[str, float] = {}
    total_sentences = 0
    total_responses = len(response_results)
    for rr in response_results:
        hal = rr.get("hallucination", {})
        ls = hal.get("layer_stats", {})
        for k, v in ls.items():
            if isinstance(v, (int, float)):
                agg[k] = agg.get(k, 0) + v
        sr = hal.get("sentence_results", [])
        total_sentences += len(sr) if sr else 0

    # Compute cascade counts
    l1 = int(agg.get("L1", 0))
    l2 = int(agg.get("L2", 0))
    l2_flagged = int(agg.get("L2_flagged", 0))
    l2_cache_hit = int(agg.get("L2_cache_hit", 0))
    l2_cache_miss = int(agg.get("L2_cache_miss", 0))
    l3_mc = int(agg.get("L3_minicheck", 0))
    l3_fc_clear = int(agg.get("L3_fact_counting_clear", 0))
    l3_fc_flag = int(agg.get("L3_fact_counting_flag", 0))
    l3_fc_error = int(agg.get("L3_fact_counting_error", 0))
    l4_sup = int(agg.get("L4_supported", 0))
    l4_unsup = int(agg.get("L4_unsupported", 0))
    minilm_flag = int(agg.get("minilm_flag", 0))

    # MiniLM averages
    avg_grounding = agg.get("minilm_grounding_ratio", 0) / max(total_responses, 1)
    avg_min_sim = agg.get("minilm_min_sentence_sim", 0) / max(total_responses, 1)

    after_l1 = total_sentences - l1
    after_l2 = after_l1 - l2  # l2_flagged NOT subtracted (they cascade)
    after_l3 = after_l2 - l3_mc - l3_fc_clear
    total_cleared = l1 + l2 + l3_mc + l3_fc_clear + l4_sup
    total_flagged = l4_unsup

    w = 62
    print()
    print("=" * w)
    print("  HALLUCINATION DETECTION — CASCADE FUNNEL")
    print("=" * w)

    print(f"\n  Responses: {total_responses}    Sentences: {total_sentences}")
    print(f"  {'─' * (w - 4)}")

    # L1
    pct_l1 = l1 * 100 / max(total_sentences, 1)
    print("\n  L1 — Rules (substring + Jaccard)")
    print(f"    Cleared:  {l1:>4d} / {total_sentences}  ({pct_l1:.1f}%)")
    print(f"    → {after_l1} sentences to L2")

    # L2
    pct_l2 = l2 * 100 / max(total_sentences, 1)
    print("\n  L2 — Knowledge Graph Ensemble")
    print(f"    Cache:    {l2_cache_hit} hit / {l2_cache_miss} miss")
    print(f"    Cleared:  {l2:>4d} / {after_l1}  ({pct_l2:.1f}% of total)")
    print(f"    Flagged:  {l2_flagged:>4d}  (cascade to L3 with evidence)")
    print(f"    → {after_l2} sentences to L3")

    # MiniLM (informational)
    print("\n  ┄┄ MiniLM (informational, not gating) ┄┄")
    print(f"    Avg grounding ratio:  {avg_grounding:.3f}")
    print(f"    Avg min sentence sim: {avg_min_sim:.3f}")
    print(f"    Responses flagged:    {minilm_flag} / {total_responses}")

    # L3
    pct_mc = l3_mc * 100 / max(total_sentences, 1)
    pct_fc = l3_fc_clear * 100 / max(total_sentences, 1)
    print("\n  L3a — MiniCheck (local, free)")
    print(f"    Cleared:  {l3_mc:>4d} / {after_l2}  ({pct_mc:.1f}% of total)")
    print("\n  L3b — Gemma Fact-Counting (API)")
    print(f"    Cleared:  {l3_fc_clear:>4d}  ({pct_fc:.1f}% of total)")
    print(f"    Flagged:  {l3_fc_flag:>4d}  (cascade to L4)")
    if l3_fc_error:
        print(f"    Errors:   {l3_fc_error:>4d}")
    print(f"    → {after_l3} sentences to L4")

    # L4
    print("\n  L4 — Gemini Per-Sentence LLM")
    print(f"    Supported:    {l4_sup:>4d}")
    print(f"    Unsupported:  {l4_unsup:>4d}")

    # Summary
    print(f"\n  {'─' * (w - 4)}")
    pct_cleared = total_cleared * 100 / max(total_sentences, 1)
    print(f"  TOTAL CLEARED:    {total_cleared:>4d} / {total_sentences}  ({pct_cleared:.1f}%)")
    print(f"  TOTAL FLAGGED:    {total_flagged:>4d} / {total_sentences}")
    unresolved = total_sentences - total_cleared - total_flagged
    if unresolved > 0:
        print(f"  UNRESOLVED:       {unresolved:>4d} / {total_sentences}")

    # L1+L2 combined (deterministic)
    det = l1 + l2
    pct_det = det * 100 / max(total_sentences, 1)
    print(f"\n  Deterministic (L1+L2): {det} ({pct_det:.1f}%)")
    print(f"  Classifier (L3):      {l3_mc + l3_fc_clear}")
    print(f"  LLM (L4):             {l4_sup + l4_unsup}")

    # Fire rates
    fire_rates = data.get("fire_rates", {})
    if "1.1" in fire_rates:
        fr = fire_rates["1.1"]
        print(f"\n  Property 1.1 fire rate: {fr.get('fail', 0)}/{fr.get('total', 0)}")

    print(f"\n  Full results: {results_path}")
    print("=" * w)
    return True


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="RAGTruth-50 operator CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # preseed
    p_pre = sub.add_parser("preseed", help="Seed graph cache from Exp 31")
    p_pre.add_argument("--cache-dir", default=None, help="Cache directory override")
    p_pre.add_argument("--ttl-hours", type=int, default=168, help="Cache TTL in hours")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run RAGTruth-50 benchmark")
    p_bench.add_argument("--max-cases", type=int, default=None, help="Limit cases")
    p_bench.add_argument(
        "--gate2-routing", default="pass",
        help="Gate 2 routing: none, pass, all (default: pass)",
    )

    # funnel
    sub.add_parser("funnel", help="Print funnel from last benchmark run")

    # all
    p_all = sub.add_parser("all", help="preseed → benchmark → funnel")
    p_all.add_argument("--cache-dir", default=None, help="Cache directory override")
    p_all.add_argument("--ttl-hours", type=int, default=168, help="Cache TTL in hours")
    p_all.add_argument("--max-cases", type=int, default=None, help="Limit cases")
    p_all.add_argument(
        "--gate2-routing", default="pass",
        help="Gate 2 routing: none, pass, all (default: pass)",
    )

    args = parser.parse_args()

    if args.command == "preseed":
        ok = cmd_preseed(args)
    elif args.command == "benchmark":
        ok = cmd_benchmark(args)
    elif args.command == "funnel":
        ok = cmd_funnel(args)
    elif args.command == "all":
        ok = cmd_preseed(args)
        if ok:
            ok = cmd_benchmark(args)
        if ok:
            ok = cmd_funnel(args)
    else:
        parser.print_help()
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
