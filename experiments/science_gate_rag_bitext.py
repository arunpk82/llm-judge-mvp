"""
Science Gate Experiment: RAG Context (Bitext Dataset)
=====================================================

Runs the grounding ratio A/B test using real Bitext support documentation
instead of synthetic docs.

Prerequisites:
    python experiments/prep_bitext_kb.py   (generates bitext_knowledge_base.json)

Usage:
    python experiments/science_gate_rag_bitext.py

If intent auto-matching fails, edit MANUAL_INTENT_MAP below.
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_judge.calibration.hallucination import check_hallucination

# =====================================================================
# Intent Mapping: our 30 cases → Bitext intents
# =====================================================================
# These map our case intents to Bitext intent names.
# After running prep_bitext_kb.py, check the intent list and adjust if needed.
#
# Our case intents (from the queries):
#   refund_double_charge  → Bitext: check_refund_policy / get_refund
#   password_reset        → Bitext: reset_password
#   missing_delivery      → Bitext: track_order / delivery_period
#   discount_request      → Bitext: newsletter_subscription / loyalty_program (?)
#   cancel_account        → Bitext: cancel_order / delete_account
#   product_compatibility → Bitext: check_compatibility (?)
#   change_shipping       → Bitext: change_shipping_address
#   app_crashing          → Bitext: complaint / problem_with_app (?)
#   cloud_storage_recovery→ Bitext: recover_password / data_recovery (?)
#   plan_comparison       → Bitext: compare_plans (?)
#   error_503             → Bitext: complaint / technical_issue (?)
#   account_balance       → Bitext: check_invoice / account_balance (?)

CASE_INTENT_KEYWORDS = {
    "cs_001": ["refund", "charge", "payment"],
    "cs_002": ["refund", "charge", "payment"],
    "cs_003": ["refund", "charge", "payment"],
    "cs_004": ["password", "reset"],
    "cs_005": ["password", "reset"],
    "cs_006": ["password", "reset"],
    "cs_007": ["delivery", "order", "track", "shipping"],
    "cs_008": ["delivery", "order", "track", "shipping"],
    "cs_009": ["delivery", "order", "track", "shipping"],
    "cs_010": ["discount", "loyalty", "offer", "promotion"],
    "cs_011": ["discount", "loyalty", "offer", "promotion"],
    "cs_012": ["discount", "loyalty", "offer", "promotion"],
    "cs_013": ["cancel", "account", "delete"],
    "cs_014": ["cancel", "account", "delete"],
    "cs_015": ["cancel", "account", "delete"],
    "cs_016": ["compatible", "compatibility", "mac", "windows", "platform"],
    "cs_017": ["compatible", "compatibility", "mac", "windows", "platform"],
    "cs_018": ["compatible", "compatibility", "mac", "windows", "platform"],
    "cs_019": ["shipping", "address", "change", "order"],
    "cs_020": ["shipping", "address", "change", "order"],
    "cs_021": ["crash", "app", "bug", "error", "phone"],
    "cs_022": ["crash", "app", "bug", "error", "phone"],
    "cs_023": ["crash", "app", "bug", "error", "phone"],
    "cs_024": ["delete", "recover", "file", "storage", "cloud"],
    "cs_025": ["delete", "recover", "file", "storage", "cloud"],
    "cs_026": ["plan", "pro", "enterprise", "compare", "difference", "upgrade"],
    "cs_027": ["plan", "pro", "enterprise", "compare", "difference", "upgrade"],
    "cs_028": ["error", "503", "service", "unavailable"],
    "cs_029": ["error", "503", "service", "unavailable"],
    "cs_030": ["account", "balance", "invoice", "billing"],
}

# Manual overrides — fill these in after seeing Bitext intent list
# Format: case_id → bitext_intent_name
MANUAL_INTENT_MAP: dict[str, str] = {
    # Example: "cs_001": "get_refund",
}


def auto_match_intent(case_id: str, keywords: list[str], bitext_intents: list[str]) -> str:
    """Best-effort match of case keywords to Bitext intent names."""
    if case_id in MANUAL_INTENT_MAP:
        return MANUAL_INTENT_MAP[case_id]

    best_score = 0.0
    best_intent = bitext_intents[0]

    for intent in bitext_intents:
        intent_words = set(intent.lower().replace("_", " ").split())
        kw_set = set(w.lower() for w in keywords)
        # Score: keyword overlap + fuzzy match on intent name
        overlap = len(intent_words & kw_set)
        fuzzy = max(
            SequenceMatcher(None, kw.lower(), intent.lower()).ratio()
            for kw in keywords
        )
        score = overlap * 2.0 + fuzzy
        if score > best_score:
            best_score = score
            best_intent = intent

    return best_intent


def build_context_from_kb(kb_entry: dict) -> str:
    """Build RAG context string from a knowledge base entry."""
    return kb_entry["documentation"]


def run_experiment():
    exp_dir = Path(__file__).parent
    kb_path = exp_dir / "bitext_knowledge_base.json"

    if not kb_path.exists():
        print("ERROR: bitext_knowledge_base.json not found.")
        print("Run first:  python experiments/prep_bitext_kb.py")
        sys.exit(1)

    with open(kb_path) as f:
        kb = json.load(f)

    bitext_intents = list(kb.keys())
    print(f"Loaded Bitext KB: {len(bitext_intents)} intents")

    # Load validation cases
    data_path = exp_dir.parent / "cs_validation_scored.jsonl"
    if not data_path.exists():
        data_path = exp_dir.parent / "datasets" / "validation" / "cs_validation_scored.jsonl"
    with open(data_path) as f:
        cases = [json.loads(line) for line in f]

    # Match cases to intents
    print("\nIntent Mapping:")
    case_intents = {}
    for case in cases:
        cid = case["case_id"]
        keywords = CASE_INTENT_KEYWORDS.get(cid, [])
        matched = auto_match_intent(cid, keywords, bitext_intents)
        case_intents[cid] = matched

    # Deduplicate for display
    unique_mappings = {}
    for cid, intent in case_intents.items():
        key = tuple(CASE_INTENT_KEYWORDS.get(cid, []))
        if key not in unique_mappings:
            unique_mappings[key] = (cid, intent)
            doc_len = len(kb.get(intent, {}).get("documentation", ""))
            examples = kb.get(intent, {}).get("total_examples", 0)
            print(f"  {cid} [{','.join(CASE_INTENT_KEYWORDS.get(cid, []))}] → {intent} ({examples} examples, {doc_len} chars)")

    print(f"\n{'='*72}")
    print("SCIENCE GATE: RAG Context — Bitext Dataset")
    print(f"{'='*72}\n")

    results_a = []
    results_b = []

    for case in cases:
        cid = case["case_id"]
        query = case["conversation"][0]["content"]
        answer = case["candidate_answer"]
        human = case["human_decision"]
        corr = case["human_scores"]["correctness"]
        intent = case_intents[cid]
        source_doc = build_context_from_kb(kb[intent])

        # A) Baseline: context = user query only
        result_a = check_hallucination(
            response=answer, context=query, case_id=cid,
            grounding_threshold=0.3,
        )

        # B) With RAG: context = user query + Bitext documentation
        enriched_context = f"{query}\n\n--- Support Documentation ---\n{source_doc}"
        result_b = check_hallucination(
            response=answer, context=enriched_context, case_id=cid,
            grounding_threshold=0.3,
        )

        results_a.append((cid, human, corr, result_a))
        results_b.append((cid, human, corr, result_b))

    # =====================================================================
    # Report
    # =====================================================================
    print(f"{'Case':<8} {'Human':<6} {'Corr':<5} {'Ground(A)':<11} {'Ground(B)':<11} {'Delta':<8} {'Intent':<30}")
    print("-" * 80)

    for (cid, human, corr, ra), (_, _, _, rb) in zip(results_a, results_b):
        delta = rb.grounding_ratio - ra.grounding_ratio
        marker = " <<<" if cid == "cs_012" else ""
        print(
            f"{cid:<8} {human:<6} {corr:<5} {ra.grounding_ratio:<11.4f} "
            f"{rb.grounding_ratio:<11.4f} {delta:+.4f}  "
            f"{case_intents[cid]:<30}{marker}"
        )

    # =====================================================================
    # Statistical summary
    # =====================================================================
    print(f"\n{'='*72}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*72}")

    pass_a = [r for _, h, _, r in results_a if h == "pass"]
    fail_a = [r for _, h, _, r in results_a if h == "fail"]
    pass_b = [r for _, h, _, r in results_b if h == "pass"]
    fail_b = [r for _, h, _, r in results_b if h == "fail"]

    cs012_a = [r for cid, _, _, r in results_a if cid == "cs_012"][0]
    cs012_b = [r for cid, _, _, r in results_b if cid == "cs_012"][0]

    low_corr_a = [r for _, _, c, r in results_a if c <= 2]
    low_corr_b = [r for _, _, c, r in results_b if c <= 2]
    high_corr_a = [r for _, _, c, r in results_a if c >= 4]
    high_corr_b = [r for _, _, c, r in results_b if c >= 4]

    def avg(results, attr="grounding_ratio"):
        vals = [getattr(r, attr) for r in results]
        return sum(vals) / len(vals) if vals else 0

    print("\nGrounding Ratio (higher = more grounded):")
    print(f"  {'Group':<25} {'Baseline (A)':<15} {'Bitext RAG (B)':<15} {'Improvement':<12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")
    for label, ga, gb in [
        ("All pass cases", pass_a, pass_b),
        ("All fail cases", fail_a, fail_b),
        ("High corr (4-5)", high_corr_a, high_corr_b),
        ("Low corr (1-2)", low_corr_a, low_corr_b),
    ]:
        print(f"  {label:<25} {avg(ga):<15.4f} {avg(gb):<15.4f} {avg(gb)-avg(ga):+.4f}")
    print(f"  {'cs_012 (fabrication)':<25} {cs012_a.grounding_ratio:<15.4f} {cs012_b.grounding_ratio:<15.4f} {cs012_b.grounding_ratio-cs012_a.grounding_ratio:+.4f}")

    gap_a = avg(pass_a) - cs012_a.grounding_ratio
    gap_b = avg(pass_b) - cs012_b.grounding_ratio

    print("\nSeparation Quality:")
    print(f"  Baseline:    pass avg={avg(pass_a):.4f}, cs_012={cs012_a.grounding_ratio:.4f}, gap={gap_a:+.4f}")
    print(f"  Bitext RAG:  pass avg={avg(pass_b):.4f}, cs_012={cs012_b.grounding_ratio:.4f}, gap={gap_b:+.4f}")
    if gap_a != 0:
        print(f"  Improvement: {gap_b/gap_a:.1f}x better separation")

    print("\nThreshold Analysis (With Bitext RAG):")
    thresholds = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    print(f"  {'Thresh':<8} {'Pass flag':<12} {'Fail flag':<12} {'cs012':<8} {'LowCorr flag':<14} {'FP rate':<10}")
    print(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*8} {'-'*14} {'-'*10}")
    for t in thresholds:
        pf = sum(1 for r in pass_b if r.grounding_ratio < t)
        ff = sum(1 for r in fail_b if r.grounding_ratio < t)
        cf = "YES" if cs012_b.grounding_ratio < t else "no"
        lcf = sum(1 for r in low_corr_b if r.grounding_ratio < t)
        fp_rate = pf / len(pass_b) if pass_b else 0
        print(f"  {t:<8.2f} {pf}/{len(pass_b):<9} {ff}/{len(fail_b):<9} {cf:<8} {lcf}/{len(low_corr_b):<11} {fp_rate:<10.1%}")

    # =====================================================================
    # Verdict
    # =====================================================================
    print(f"\n{'='*72}")
    improved = gap_b > gap_a * 2

    if improved:
        print("VERDICT: PASS")
        print(f"  Separation gap: baseline={gap_a:+.4f} → Bitext RAG={gap_b:+.4f}")
        ratio = gap_b / gap_a if gap_a != 0 else float('inf')
        print(f"  Improvement: {ratio:.0f}x better separation")
        print("  Proceed to Step 3 (Vision & Maturity).")
    else:
        print("VERDICT: FAIL or INCONCLUSIVE")
        print(f"  Separation gap: baseline={gap_a:+.4f} → Bitext RAG={gap_b:+.4f}")
        print("  RAG context insufficient — consider embedding-based approach.")
    print(f"{'='*72}")

    # Save raw results
    out_path = exp_dir / "science_gate_results_bitext.json"
    results_out = {
        "experiment": "science_gate_rag_bitext",
        "dataset": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "cases": len(cases),
        "baseline_pass_avg": round(avg(pass_a), 4),
        "baseline_cs012": cs012_a.grounding_ratio,
        "baseline_gap": round(gap_a, 4),
        "rag_pass_avg": round(avg(pass_b), 4),
        "rag_cs012": cs012_b.grounding_ratio,
        "rag_gap": round(gap_b, 4),
        "verdict": "PASS" if improved else "FAIL",
        "per_case": [
            {
                "case_id": cid, "human": h, "correctness": c,
                "intent": case_intents[cid],
                "grounding_baseline": ra.grounding_ratio,
                "grounding_rag": rb.grounding_ratio,
                "delta": round(rb.grounding_ratio - ra.grounding_ratio, 4),
            }
            for (cid, h, c, ra), (_, _, _, rb) in zip(results_a, results_b)
        ],
    }
    with open(out_path, "w") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run_experiment()
