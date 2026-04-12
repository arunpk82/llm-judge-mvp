"""
RAG System Verification — validate retrieval accuracy before
trusting it for groundedness evaluation.

Measures: Recall@1, Recall@3, MRR, Hit Rate across all 27 intents.
Uses Bitext queries with known intents as ground truth.

Usage:
    poetry run python -m llm_judge.benchmarks.verify_rag --max-queries 200
"""

from __future__ import annotations

import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path

BITEXT_PATH = Path(
    "datasets/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_Responses.csv"
)
OUTPUT_DIR = Path("experiments")


def load_test_queries(max_per_intent: int = 10) -> list[dict[str, str]]:
    """Load test queries from Bitext — sample per intent for balance."""
    by_intent: dict[str, list[dict[str, str]]] = defaultdict(list)

    with BITEXT_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row.get("intent", "").strip()
            query = row.get("instruction", "").strip()
            if intent and query:
                by_intent[intent].append({"query": query, "intent": intent})

    # Sample evenly across intents
    random.seed(42)
    queries = []
    for intent, cases in sorted(by_intent.items()):
        sampled = random.sample(cases, min(max_per_intent, len(cases)))
        queries.extend(sampled)

    random.shuffle(queries)
    return queries


def verify_rag(max_queries: int = 200, top_k: int = 3) -> None:
    """Run RAG verification."""
    from llm_judge.retrieval.context_retriever import ContextRetriever, RetrievalConfig
    from llm_judge.retrieval.knowledge_base import load_knowledge_base

    print("=" * 70)
    print("RAG SYSTEM VERIFICATION")
    print("=" * 70)

    # Step 1: Load KB
    print("\nStep 1: Loading knowledge base...")
    kb = load_knowledge_base()
    print(f"  Documents: {kb.document_count}")
    print(f"  Loaded: {kb.is_loaded}")

    if not kb.is_loaded or kb.document_count == 0:
        print("  ERROR: Knowledge base is empty. Cannot verify.")
        return

    # Step 2: Create retriever
    config = RetrievalConfig(
        method="cosine_similarity",
        top_k=top_k,
        similarity_threshold=0.0,  # Don't filter — we want to see all scores
    )
    retriever = ContextRetriever(vector_store=kb.store, config=config)

    # Step 3: Load test queries
    print("\nStep 2: Loading test queries from Bitext...")
    queries = load_test_queries(max_per_intent=max_queries // 27 + 1)
    queries = queries[:max_queries]
    intents_covered = set(q["intent"] for q in queries)
    print(f"  Queries: {len(queries)}")
    print(f"  Intents covered: {len(intents_covered)} / 27")

    # Step 4: Run retrieval
    print(f"\nStep 3: Running retrieval (top_k={top_k})...")
    t0 = time.time()

    recall_at_1 = 0
    recall_at_3 = 0
    hit_rate = 0
    reciprocal_ranks = []
    per_intent_results: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "hit_at_1": 0, "hit_at_3": 0}
    )
    score_by_rank: dict[str, list[float]] = {
        "correct_top1": [],
        "correct_any": [],
        "incorrect_top1": [],
    }
    failures: list[dict] = []

    for i, q in enumerate(queries):
        docs, evidence = retriever.retrieve(q["query"])
        expected_intent = q["intent"]

        if evidence is None or not evidence.doc_ids:
            failures.append(
                {"query": q["query"], "intent": expected_intent, "error": "no_results"}
            )
            per_intent_results[expected_intent]["total"] += 1
            reciprocal_ranks.append(0.0)
            continue

        doc_ids = evidence.doc_ids
        per_intent_results[expected_intent]["total"] += 1

        # Recall@1
        if doc_ids[0] == expected_intent:
            recall_at_1 += 1
            per_intent_results[expected_intent]["hit_at_1"] += 1
            score_by_rank["correct_top1"].append(evidence.top_score)
        else:
            score_by_rank["incorrect_top1"].append(evidence.top_score)

        # Recall@3
        if expected_intent in doc_ids[:3]:
            recall_at_3 += 1
            per_intent_results[expected_intent]["hit_at_3"] += 1

        # Hit rate (any position)
        if expected_intent in doc_ids:
            hit_rate += 1
            score_by_rank["correct_any"].append(evidence.top_score)

        # MRR
        try:
            rank = doc_ids.index(expected_intent) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
            failures.append(
                {
                    "query": q["query"][:80],
                    "intent": expected_intent,
                    "got": doc_ids,
                    "top_score": evidence.top_score,
                }
            )

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(queries)} queries processed")

    elapsed = time.time() - t0
    n = len(queries)

    # Step 5: Report
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    print(f"\nOVERALL METRICS ({n} queries, {elapsed:.1f}s)")
    print(f"  Recall@1:  {recall_at_1/n:.3f} ({recall_at_1}/{n})")
    print(f"  Recall@3:  {recall_at_3/n:.3f} ({recall_at_3}/{n})")
    print(f"  Hit Rate:  {hit_rate/n:.3f} ({hit_rate}/{n})")
    mrr = sum(reciprocal_ranks) / n if n > 0 else 0
    print(f"  MRR:       {mrr:.3f}")
    print(f"  Latency:   {elapsed/n*1000:.0f}ms per query")

    # Verdict
    print("\nVERDICT")
    if recall_at_1 / n >= 0.90:
        print(f"  Recall@1 = {recall_at_1/n:.1%} — EXCELLENT. RAG is production-ready.")
    elif recall_at_1 / n >= 0.75:
        print(
            f"  Recall@1 = {recall_at_1/n:.1%} — GOOD. RAG is usable, some intents need tuning."
        )
    elif recall_at_1 / n >= 0.50:
        print(
            f"  Recall@1 = {recall_at_1/n:.1%} — FAIR. RAG needs improvement before production use."
        )
    else:
        print(
            f"  Recall@1 = {recall_at_1/n:.1%} — POOR. RAG is not reliable for groundedness evaluation."
        )

    # Per-intent breakdown
    print("\nPER-INTENT BREAKDOWN")
    print(f"  {'Intent':<30} {'Total':>6} {'R@1':>6} {'R@3':>6} {'R@1%':>6}")
    print(f"  {'-'*30} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    for intent in sorted(per_intent_results.keys()):
        r = per_intent_results[intent]
        r1_pct = r["hit_at_1"] / r["total"] * 100 if r["total"] > 0 else 0
        mark = "***" if r1_pct < 50 else ""
        print(
            f"  {intent:<30} {r['total']:>6} {r['hit_at_1']:>6} {r['hit_at_3']:>6} {r1_pct:>5.0f}% {mark}"
        )

    # Score distributions
    if score_by_rank["correct_top1"]:
        correct_scores = score_by_rank["correct_top1"]
        print("\nSCORE ANALYSIS")
        print(
            f"  Correct top-1 scores:   min={min(correct_scores):.3f} mean={sum(correct_scores)/len(correct_scores):.3f} max={max(correct_scores):.3f} (n={len(correct_scores)})"
        )
    if score_by_rank["incorrect_top1"]:
        wrong_scores = score_by_rank["incorrect_top1"]
        print(
            f"  Incorrect top-1 scores: min={min(wrong_scores):.3f} mean={sum(wrong_scores)/len(wrong_scores):.3f} max={max(wrong_scores):.3f} (n={len(wrong_scores)})"
        )

    # Failures
    if failures:
        print(
            f"\nFAILURES ({len(failures)} queries where correct intent not in top-{top_k})"
        )
        for f in failures[:10]:
            print(f"  Query: {f.get('query', '')[:60]}")
            print(f"    Expected: {f['intent']}, Got: {f.get('got', 'none')}")

    print(f"\n{'=' * 70}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "rag_verification_results.json"
    save_data = {
        "queries": n,
        "recall_at_1": round(recall_at_1 / n, 4),
        "recall_at_3": round(recall_at_3 / n, 4),
        "hit_rate": round(hit_rate / n, 4),
        "mrr": round(mrr, 4),
        "per_intent": dict(per_intent_results),
        "failure_count": len(failures),
    }
    with save_path.open("w") as out_f:
        json.dump(save_data, out_f, indent=2)
    print(f"\nResults saved: {save_path}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Verify RAG system accuracy")
    parser.add_argument("--max-queries", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    verify_rag(max_queries=args.max_queries, top_k=args.top_k)


if __name__ == "__main__":
    main()
