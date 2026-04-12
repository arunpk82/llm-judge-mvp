"""
Science Gate Experiment: Grounding Ratio with Source Context
============================================================

WHAT THIS TESTS:
  Does providing source documentation to check_hallucination() improve
  its ability to separate fabricated responses from legitimate ones?

WHAT THIS IS NOT:
  This is NOT a RAG pipeline. There is no embedding retrieval, no vector
  search, no chunking, no similarity scoring. The "retrieval" is a
  controlled intent-to-document lookup — we hand the grounding function
  the exact documentation it needs, simulating a perfect retrieval hit.

  This is intentional. The Science Gate tests the MECHANISM (does source
  context help grounding?), not the retrieval quality. Retrieval quality
  is a separate concern for Step 5 (Functional Decomposition).

METHOD:
  For each of our 30 validation cases, run check_hallucination() twice:
    A) context = user query only              (current pipeline)
    B) context = user query + source docs     (simulated perfect retrieval)

  Compare grounding ratios. If separation improves, the mechanism works.

KNOWLEDGE BASE:
  Loaded from experiments/synthetic_knowledge_base.json — synthetic
  customer support policy docs, one per intent. Hand-written to match
  the vocabulary a real support knowledge base would contain.

Usage:
    python experiments/science_gate_experiment.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_judge.calibration.hallucination import check_hallucination


def load_knowledge_base(kb_path: Path) -> tuple[dict, dict]:
    """Load knowledge base and case-intent mapping from JSON."""
    with open(kb_path) as f:
        data = json.load(f)
    case_map = data["case_intent_map"]
    kb = {k: v["documentation"] for k, v in data["knowledge_base"].items()}
    return case_map, kb


def run_experiment():
    exp_dir = Path(__file__).parent
    repo_root = exp_dir.parent

    # Load knowledge base
    kb_path = exp_dir / "synthetic_knowledge_base.json"
    if not kb_path.exists():
        print(f"ERROR: {kb_path} not found")
        sys.exit(1)
    case_intent_map, kb = load_knowledge_base(kb_path)

    # Load validation cases
    data_path = repo_root / "cs_validation_scored.jsonl"
    if not data_path.exists():
        data_path = repo_root / "datasets" / "validation" / "cs_validation_scored.jsonl"
    with open(data_path) as f:
        cases = [json.loads(line) for line in f]

    print("=" * 76)
    print("SCIENCE GATE: Does Source Context Improve Grounding Separation?")
    print("=" * 76)
    print("Method: intent-exact-match lookup (NOT RAG retrieval)")
    print(
        f"Cases: {len(cases)}  |  Intents: {len(kb)}  |  KB: synthetic_knowledge_base.json"
    )
    print()

    results_a = []  # Baseline
    results_b = []  # With source context

    for case in cases:
        cid = case["case_id"]
        query = case["conversation"][0]["content"]
        answer = case["candidate_answer"]
        human = case["human_decision"]
        corr = case["human_scores"]["correctness"]
        intent = case_intent_map[cid]
        source_doc = kb[intent]

        # A) Baseline: context = user query only
        result_a = check_hallucination(
            response=answer,
            context=query,
            case_id=cid,
            grounding_threshold=0.3,
        )

        # B) With source docs: context = user query + documentation
        enriched_context = f"{query}\n\n--- Source Documentation ---\n{source_doc}"
        result_b = check_hallucination(
            response=answer,
            context=enriched_context,
            case_id=cid,
            grounding_threshold=0.3,
        )

        results_a.append((cid, human, corr, intent, result_a))
        results_b.append((cid, human, corr, intent, result_b))

    # =================================================================
    # Per-case results
    # =================================================================
    print(
        f"{'Case':<8} {'Human':<6} {'Corr':<5} {'Ground(A)':<11} {'Ground(B)':<11} "
        f"{'Delta':<9} {'Intent':<25}"
    )
    print("-" * 80)

    for (cid, human, corr, intent, ra), (_, _, _, _, rb) in zip(results_a, results_b):
        delta = rb.grounding_ratio - ra.grounding_ratio
        marker = " <<<" if cid == "cs_012" else ""
        print(
            f"{cid:<8} {human:<6} {corr:<5} {ra.grounding_ratio:<11.4f} "
            f"{rb.grounding_ratio:<11.4f} {delta:+.4f}   "
            f"{intent:<25}{marker}"
        )

    # =================================================================
    # Statistical summary
    # =================================================================
    def avg(items):
        return sum(items) / len(items) if items else 0.0

    pass_a = [r.grounding_ratio for _, h, _, _, r in results_a if h == "pass"]
    fail_a = [r.grounding_ratio for _, h, _, _, r in results_a if h == "fail"]
    pass_b = [r.grounding_ratio for _, h, _, _, r in results_b if h == "pass"]
    fail_b = [r.grounding_ratio for _, h, _, _, r in results_b if h == "fail"]

    cs012_a = [r.grounding_ratio for cid, _, _, _, r in results_a if cid == "cs_012"][0]
    cs012_b = [r.grounding_ratio for cid, _, _, _, r in results_b if cid == "cs_012"][0]

    print(f"\n{'='*76}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*76}")

    print("\nGrounding Ratio (higher = more grounded):")
    print(f"  {'Group':<25} {'Query Only (A)':<16} {'With Docs (B)':<16} {'Delta':<10}")
    print(f"  {'-'*25} {'-'*16} {'-'*16} {'-'*10}")
    print(
        f"  {'Pass cases':<25} {avg(pass_a):<16.4f} {avg(pass_b):<16.4f} {avg(pass_b)-avg(pass_a):+.4f}"
    )
    print(
        f"  {'Fail cases':<25} {avg(fail_a):<16.4f} {avg(fail_b):<16.4f} {avg(fail_b)-avg(fail_a):+.4f}"
    )
    print(
        f"  {'cs_012 (fabrication)':<25} {cs012_a:<16.4f} {cs012_b:<16.4f} {cs012_b-cs012_a:+.4f}"
    )

    gap_a = avg(pass_a) - cs012_a
    gap_b = avg(pass_b) - cs012_b

    print("\nSeparation (pass avg minus cs_012):")
    print(f"  Query only:  {avg(pass_a):.4f} - {cs012_a:.4f} = {gap_a:+.4f}")
    print(f"  With docs:   {avg(pass_b):.4f} - {cs012_b:.4f} = {gap_b:+.4f}")

    # =================================================================
    # Threshold analysis
    # =================================================================
    print("\nThreshold Analysis (With Source Docs):")
    print(
        f"  {'Thresh':<8} {'Pass<T':<10} {'Fail<T':<10} {'cs012<T':<10} {'FP rate':<10}"
    )
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pf = sum(1 for g in pass_b if g < t)
        ff = sum(1 for g in fail_b if g < t)
        cf = "YES" if cs012_b < t else "no"
        fp = pf / len(pass_b) if pass_b else 0
        print(
            f"  {t:<8.2f} {pf}/{len(pass_b):<7} {ff}/{len(fail_b):<7} {cf:<10} {fp:.1%}"
        )

    # =================================================================
    # Verdict
    # =================================================================
    print(f"\n{'='*76}")
    if gap_b > gap_a * 2 and gap_b > 0.05:
        print("VERDICT: PASS")
        print(f"  Separation improved from {gap_a:+.4f} to {gap_b:+.4f}")
        if gap_a > 0:
            print(f"  {gap_b/gap_a:.0f}x better discrimination")
        print("  Source context enables meaningful grounding measurement.")
        print("  → Proceed to Step 3 (Vision & Maturity)")
    elif gap_b > gap_a:
        print("VERDICT: MARGINAL PASS")
        print(f"  Separation improved from {gap_a:+.4f} to {gap_b:+.4f}")
        print("  Improvement exists but may not be sufficient for gating.")
        print("  Consider: embedding-based grounding or multi-property composite.")
    else:
        print("VERDICT: FAIL")
        print(f"  Separation: baseline={gap_a:+.4f}, with docs={gap_b:+.4f}")
        print("  Source context alone is insufficient. Investigate root cause.")
    print(f"{'='*76}")

    # Save structured results
    out = {
        "experiment": "science_gate_grounding_context",
        "retrieval_method": "intent_exact_match",
        "knowledge_base": "synthetic_knowledge_base.json",
        "cases": len(cases),
        "baseline": {
            "pass_avg": round(avg(pass_a), 4),
            "cs012": cs012_a,
            "gap": round(gap_a, 4),
        },
        "with_docs": {
            "pass_avg": round(avg(pass_b), 4),
            "cs012": cs012_b,
            "gap": round(gap_b, 4),
        },
        "verdict": (
            "PASS"
            if (gap_b > gap_a * 2 and gap_b > 0.05)
            else "MARGINAL" if gap_b > gap_a else "FAIL"
        ),
        "per_case": [
            {
                "case_id": cid,
                "human": h,
                "correctness": c,
                "intent": intent,
                "grounding_baseline": ra.grounding_ratio,
                "grounding_with_docs": rb.grounding_ratio,
                "delta": round(rb.grounding_ratio - ra.grounding_ratio, 4),
            }
            for (cid, h, c, intent, ra), (_, _, _, _, rb) in zip(results_a, results_b)
        ],
    }
    out_path = exp_dir / "science_gate_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults → {out_path}")


if __name__ == "__main__":
    run_experiment()
