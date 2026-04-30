#!/usr/bin/env python3
"""
RAGTruth-50 Operator CLI — preseed, funnel.

The ``benchmark`` subcommand was retired in L1-Pkt-A (CP-F1 closure):
it was the parallel orchestration entry point that bypassed the
Control Plane Runner. Running benchmarks now happens via
``tools/run_batch_evaluation.py`` (``make demo-batch-quick`` /
``make benchmark`` aliases). This script is preserved for the two
non-orchestration utilities it provides:

  * ``preseed`` — seed the L2 hallucination graph cache from Exp 31
    fact tables.
  * ``funnel`` — pretty-print a funnel report from the last
    benchmark results JSON in ``results/ragtruth50_results.json``.

Usage:
    python tools/run_ragtruth50.py preseed
    python tools/run_ragtruth50.py funnel

Output:
    results/ragtruth50_funnel.txt     — funnel report (read input)
    .cache/hallucination_graphs/      — graph cache directory (preseed)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

    # ---- 28-Metric Status Report ----
    fire_rates = data.get("fire_rates", {})

    print(f"\n{'=' * w}")
    print("  28-METRIC PROPERTY EVALUATION REPORT")
    print(f"{'=' * w}")

    for cat_name, props in PROPERTY_CATALOG_GROUPED.items():
        print(f"\n  {cat_name}")
        print(f"  {'─' * (w - 4)}")
        print(f"  {'ID':<6} {'Property':<28} {'Status':<14} {'Fire Rate'}")
        print(f"  {'─' * (w - 4)}")

        for pid, prop_name, impl_status in props:
            fr = fire_rates.get(pid, {})
            total = fr.get("total", 0)
            fail = fr.get("fail", 0)

            if total > 0:
                rate_str = f"{fail}/{total} ({fail * 100 / total:.0f}%)"
            elif impl_status == "Stub":
                rate_str = "—"
            else:
                rate_str = "0/0"

            print(f"  {pid:<6} {prop_name:<28} {impl_status:<14} {rate_str}")

    # Summary counts
    cats = {"Implemented": 0, "Partial": 0, "Stub": 0}
    for props in PROPERTY_CATALOG_GROUPED.values():
        for _, _, status in props:
            cats[status] = cats.get(status, 0) + 1

    print(f"\n  {'─' * (w - 4)}")
    print(f"  Implemented: {cats['Implemented']}   Partial: {cats['Partial']}   Stub: {cats['Stub']}   Total: 28")

    print(f"\n  Full results: {results_path}")
    print("=" * w)
    return True


# =====================================================================
# Property catalog — all 28 metrics
# =====================================================================

PROPERTY_CATALOG = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7",
    "3.1", "3.2", "3.3",
    "4.1", "4.2",
    "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7",
    "6.1", "6.2", "6.3", "6.4",
]

PROPERTY_CATALOG_GROUPED: dict[str, list[tuple[str, str, str]]] = {
    "Cat 1 — Faithfulness": [
        ("1.1", "Hallucination Detection", "Implemented"),
        ("1.2", "Ungrounded Claims", "Implemented"),
        ("1.3", "Unverifiable Citations", "Implemented"),
        ("1.4", "Attribution Accuracy", "Implemented"),
        ("1.5", "Fabrication Detection", "Implemented"),
    ],
    "Cat 2 — Semantic Quality": [
        ("2.1", "Relevance", "Implemented"),
        ("2.2", "Clarity", "Implemented"),
        ("2.3", "Correctness", "Implemented"),
        ("2.4", "Tone", "Implemented"),
        ("2.5", "Completeness", "Implemented"),
        ("2.6", "Coherence", "Implemented"),
        ("2.7", "Depth & Nuance", "Implemented"),
    ],
    "Cat 3 — Safety": [
        ("3.1", "Toxicity", "Implemented"),
        ("3.2", "Instruction Boundary", "Implemented"),
        ("3.3", "PII Leakage", "Implemented"),
    ],
    "Cat 4 — Task Fidelity": [
        ("4.1", "Instruction Following", "Implemented"),
        ("4.2", "Format & Structure", "Implemented"),
    ],
    "Cat 5 — Robustness": [
        ("5.1", "Position Bias", "Implemented"),
        ("5.2", "Length Bias", "Implemented"),
        ("5.3", "Self-Preference Bias", "Stub"),
        ("5.4", "Consistency", "Implemented"),
        ("5.5", "Discrimination", "Partial"),
        ("5.6", "Calibration", "Partial"),
        ("5.7", "Adversarial Robustness", "Stub"),
    ],
    "Cat 6 — Performance": [
        ("6.1", "Latency", "Implemented"),
        ("6.2", "Cost Estimation", "Implemented"),
        ("6.3", "Explainability", "Implemented"),
        ("6.4", "Reasoning Fidelity", "Implemented"),
    ],
}


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

    # funnel
    sub.add_parser("funnel", help="Print funnel from last benchmark run")

    args = parser.parse_args()

    if args.command == "preseed":
        ok = cmd_preseed(args)
    elif args.command == "funnel":
        ok = cmd_funnel(args)
    else:
        parser.print_help()
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
