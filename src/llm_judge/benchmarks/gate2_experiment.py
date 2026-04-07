"""
Experiment 3: Gate 2 LLM Routing for Property 1.1 Groundedness.

Purpose: Route ambiguous cases (grounding ratio 0.30-0.70) to Gemini for
LLM-based groundedness judgment. Test whether Gate 2 can close the gap
from F1=0.570 to the published F1=0.635.

Approach:
  - Gate 1: Embedding similarity (from Exp 1+2). Clear cases decided here.
  - Gate 2: Gemini judges ambiguous cases with a groundedness prompt.
  - Combined: Gate 1 handles clear cases, Gate 2 handles the rest.

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python -m llm_judge.benchmarks.gate2_experiment --max-cases 500
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
OUTPUT_DIR = Path("experiments")

# Thresholds from Experiment 2
SIM_THRESHOLD = 0.30
RATIO_FAIL = 0.30    # ratio < this = Gate 1 FAIL
RATIO_PASS = 0.70    # ratio >= this = Gate 1 PASS
# Between RATIO_FAIL and RATIO_PASS = route to Gate 2

GROUNDEDNESS_PROMPT = """You are evaluating whether a customer support response is grounded in the provided source context.

SOURCE CONTEXT:
{context}

RESPONSE TO EVALUATE:
{response}

Instructions:
- A response is GROUNDED if all factual claims in it are supported by or consistent with the source context.
- A response is HALLUCINATED if it contains specific facts, policies, statistics, names, or claims that are NOT in the source context and appear to be fabricated.
- General polite language, greetings, and offers to help are acceptable even if not explicitly in the source.
- Focus on factual claims, not style or tone.

Answer with exactly one word: GROUNDED or HALLUCINATED"""


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


@dataclass
class Gate2Case:
    """A case with pre-computed Gate 1 results, ready for Gate 2 if needed."""
    case_id: str
    expected: str
    injection_type: str
    response_text: str
    context_text: str
    grounding_ratio: float
    gate1_decision: str      # "fail", "pass", or "ambiguous"
    gate2_decision: str = "" # filled by Gemini
    gate2_raw: str = ""      # raw Gemini response
    final_decision: str = "" # combined


def compute_gate1(max_cases: int = 500) -> list[Gate2Case]:
    """Step 1: Run Gate 1 (embeddings) and classify cases into bands."""
    from llm_judge.benchmarks.master_gt import MasterGroundTruthAdapter
    from llm_judge.properties import get_embedding_provider

    adapter = MasterGroundTruthAdapter()
    provider = get_embedding_provider()
    cases: list[Gate2Case] = []

    print(f"Step 1: Computing Gate 1 (embedding similarity) for {max_cases} cases...")
    t0 = time.time()
    count = 0

    for bc in adapter.load_cases(max_cases=max_cases):
        expected = bc.ground_truth.property_labels.get("1.1")
        if expected is None:
            continue

        response_text = bc.request.candidate_answer
        context_parts = list(bc.request.source_context or [])
        conversation = " ".join(msg.content for msg in bc.request.conversation)
        context = conversation + ("\n\n" + "\n".join(context_parts) if context_parts else "")

        response_sents = _split_sentences(response_text)
        context_sents = _split_sentences(context)

        if not response_sents or not context_sents:
            continue

        resp_embs = provider.encode(response_sents)
        ctx_embs = provider.encode(context_sents)

        grounded = sum(
            1 for r_emb in resp_embs
            if provider.max_similarity(r_emb, ctx_embs) >= SIM_THRESHOLD
        )
        ratio = grounded / len(response_sents)

        if ratio < RATIO_FAIL:
            gate1 = "fail"
        elif ratio >= RATIO_PASS:
            gate1 = "pass"
        else:
            gate1 = "ambiguous"

        cases.append(Gate2Case(
            case_id=bc.case_id,
            expected=expected,
            injection_type=bc.metadata.get("injection_type", ""),
            response_text=response_text,
            context_text=context,
            grounding_ratio=round(ratio, 4),
            gate1_decision=gate1,
        ))

        count += 1
        if count % 100 == 0:
            print(f"  {count} cases processed ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Gate 1 complete: {len(cases)} cases in {elapsed:.1f}s")

    # Report band distribution
    fail_count = sum(1 for c in cases if c.gate1_decision == "fail")
    pass_count = sum(1 for c in cases if c.gate1_decision == "pass")
    ambig_count = sum(1 for c in cases if c.gate1_decision == "ambiguous")
    print(f"  Band distribution: FAIL={fail_count}, AMBIGUOUS={ambig_count}, PASS={pass_count}")

    return cases


def run_gate2(cases: list[Gate2Case]) -> None:
    """Step 2: Route ambiguous cases to Gemini via REST API."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set. Cannot run Gate 2.")
        return

    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    ambiguous = [c for c in cases if c.gate1_decision == "ambiguous"]
    print(f"\nStep 2: Routing {len(ambiguous)} ambiguous cases to Gemini...")
    print(f"  Using model: {model_name}")

    t0 = time.time()
    with httpx.Client(timeout=30.0) as client:
        for i, case in enumerate(ambiguous):
            prompt = GROUNDEDNESS_PROMPT.format(
                context=case.context_text[:2000],
                response=case.response_text[:1000],
            )

            url = f"{base_url}/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "topP": 1.0,
                },
            }

            try:
                resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                case.gate2_raw = raw

                if "HALLUCINATED" in raw:
                    case.gate2_decision = "fail"
                elif "GROUNDED" in raw:
                    case.gate2_decision = "pass"
                else:
                    case.gate2_decision = "pass"
                    print(f"  WARNING: Unclear response for {case.case_id}: {raw[:50]}")

            except Exception as e:
                case.gate2_decision = "pass"
                case.gate2_raw = f"ERROR: {str(e)[:80]}"
                print(f"  ERROR for {case.case_id}: {str(e)[:80]}")

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(ambiguous)} judged ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Gate 2 complete: {len(ambiguous)} cases in {elapsed:.1f}s")
    if ambiguous:
        print(f"  Average latency: {elapsed/len(ambiguous)*1000:.0f}ms per case")


def compute_final_decisions(cases: list[Gate2Case]) -> None:
    """Step 3: Combine Gate 1 and Gate 2 decisions."""
    for case in cases:
        if case.gate1_decision == "fail":
            case.final_decision = "fail"
        elif case.gate1_decision == "pass":
            case.final_decision = "pass"
        elif case.gate1_decision == "ambiguous":
            case.final_decision = case.gate2_decision if case.gate2_decision else "pass"


def print_report(cases: list[Gate2Case]) -> None:
    """Step 4: Full experiment report."""

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: GATE 2 LLM ROUTING — PROPERTY 1.1 GROUNDEDNESS")
    print("=" * 80)

    print("\nPURPOSE")
    print("  Route ambiguous cases (ratio 0.30-0.70) to Gemini.")
    print("  Close the gap from F1=0.570 (Exp 2) toward published F1=0.635.")

    # Band distribution
    fail_band = [c for c in cases if c.gate1_decision == "fail"]
    ambig_band = [c for c in cases if c.gate1_decision == "ambiguous"]
    pass_band = [c for c in cases if c.gate1_decision == "pass"]

    print("\nSTEP 1: GATE 1 BAND DISTRIBUTION")
    print(f"  Total cases with 1.1 labels: {len(cases)}")
    print(f"  FAIL band  (ratio < {RATIO_FAIL}):  {len(fail_band):>4} cases ({len(fail_band)/len(cases)*100:.1f}%)")
    print(f"  AMBIGUOUS  ({RATIO_FAIL} <= ratio < {RATIO_PASS}): {len(ambig_band):>4} cases ({len(ambig_band)/len(cases)*100:.1f}%)")
    print(f"  PASS band  (ratio >= {RATIO_PASS}):  {len(pass_band):>4} cases ({len(pass_band)/len(cases)*100:.1f}%)")

    # Gate 1 only accuracy (for comparison)
    g1_tp = g1_fp = g1_tn = g1_fn = 0
    for c in cases:
        g1_pred = "fail" if c.gate1_decision in ("fail", "ambiguous") else "pass"
        if g1_pred == "fail" and c.expected == "fail":
            g1_tp += 1
        elif g1_pred == "fail" and c.expected == "pass":
            g1_fp += 1
        elif g1_pred == "pass" and c.expected == "pass":
            g1_tn += 1
        elif g1_pred == "pass" and c.expected == "fail":
            g1_fn += 1

    # Gate 1 with Exp 2 thresholds (treating ambiguous as fail)
    g1e2_tp = g1e2_fp = g1e2_tn = g1e2_fn = 0
    for c in cases:
        g1e2_pred = "fail" if c.grounding_ratio < 0.60 else "pass"
        if g1e2_pred == "fail" and c.expected == "fail":
            g1e2_tp += 1
        elif g1e2_pred == "fail" and c.expected == "pass":
            g1e2_fp += 1
        elif g1e2_pred == "pass" and c.expected == "pass":
            g1e2_tn += 1
        elif g1e2_pred == "pass" and c.expected == "fail":
            g1e2_fn += 1

    def f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0, p, r

    g1e2_f1, g1e2_p, g1e2_r = f1(g1e2_tp, g1e2_fp, g1e2_fn)
    print(f"\n  Exp 2 baseline (ratio<0.60=fail): F1={g1e2_f1:.3f} P={g1e2_p:.3f} R={g1e2_r:.3f}")

    # Gate 2 accuracy on ambiguous band
    g2_f1 = g2_p = g2_r = 0.0
    if ambig_band:
        print("\nSTEP 2: GATE 2 (GEMINI) ON AMBIGUOUS BAND")
        g2_tp = g2_fp = g2_tn = g2_fn = 0
        for c in ambig_band:
            if c.gate2_decision == "fail" and c.expected == "fail":
                g2_tp += 1
            elif c.gate2_decision == "fail" and c.expected == "pass":
                g2_fp += 1
            elif c.gate2_decision == "pass" and c.expected == "pass":
                g2_tn += 1
            elif c.gate2_decision == "pass" and c.expected == "fail":
                g2_fn += 1

        g2_f1, g2_p, g2_r = f1(g2_tp, g2_fp, g2_fn)
        g2_fail_in_band = sum(1 for c in ambig_band if c.expected == "fail")
        g2_pass_in_band = sum(1 for c in ambig_band if c.expected == "pass")
        print(f"  Cases in band: {len(ambig_band)} (fail={g2_fail_in_band}, pass={g2_pass_in_band})")
        print(f"  Gate 2 F1: {g2_f1:.3f}  P={g2_p:.3f}  R={g2_r:.3f}")
        print(f"  Gate 2 TP={g2_tp} FP={g2_fp} TN={g2_tn} FN={g2_fn}")
        print(f"  Gemini judged {sum(1 for c in ambig_band if c.gate2_decision=='fail')} as HALLUCINATED, {sum(1 for c in ambig_band if c.gate2_decision=='pass')} as GROUNDED")

    # Combined result
    print("\nSTEP 3: COMBINED RESULT (GATE 1 + GATE 2)")
    tp = fp = tn = fn = 0
    for c in cases:
        if c.final_decision == "fail" and c.expected == "fail":
            tp += 1
        elif c.final_decision == "fail" and c.expected == "pass":
            fp += 1
        elif c.final_decision == "pass" and c.expected == "pass":
            tn += 1
        elif c.final_decision == "pass" and c.expected == "fail":
            fn += 1

    combined_f1, combined_p, combined_r = f1(tp, fp, fn)
    fire_rate = (tp + fp) / len(cases) if cases else 0

    print(f"  F1:        {combined_f1:.3f}")
    print(f"  Precision: {combined_p:.3f}")
    print(f"  Recall:    {combined_r:.3f}")
    print(f"  Fire rate: {fire_rate:.1%}")
    print(f"  TP={tp}  FP={fp}  TN={tn}  FN={fn}")

    # Progress table
    print("\nPROGRESS ACROSS ALL EXPERIMENTS")
    print(f"  {'Experiment':<30} {'F1':>8} {'P':>8} {'R':>8} {'Fire%':>8} {'Method'}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*30}")
    print(f"  {'Baseline':<30} {'0.461':>8} {'0.300':>8} {'1.000':>8} {'99.6%':>8} Token overlap")
    print(f"  {'Exp 1: Method upgrade':<30} {'0.456':>8} {'0.410':>8} {'0.512':>8} {'36.6%':>8} Embeddings, untuned")
    print(f"  {'Exp 2: Threshold sweep':<30} {'0.570':>8} {'0.447':>8} {'0.787':>8} {'54.2%':>8} Embeddings, tuned")
    print(f"  {'Exp 3: Gate 2 LLM':<30} {combined_f1:>8.3f} {combined_p:>8.3f} {combined_r:>8.3f} {fire_rate:>7.1%} Embeddings + Gemini")
    print(f"  {'Published (GPT-4)':<30} {'0.635':>8} {'—':>8} {'—':>8} {'—':>8} LLM-as-judge only")

    # Observations
    print("\nOBSERVATIONS")
    delta_f1 = combined_f1 - 0.570
    print(f"  1. F1 delta from Exp 2: {delta_f1:+.3f} ({delta_f1/0.570*100:+.1f}%)")
    print(f"  2. Cases routed to Gate 2: {len(ambig_band)} ({len(ambig_band)/len(cases)*100:.1f}% of total)")
    if ambig_band:
        print(f"  3. Gate 2 accuracy on routed cases: F1={g2_f1:.3f}")
    gap_to_published = 0.635 - combined_f1
    print(f"  4. Remaining gap to published: {gap_to_published:.3f}")

    print(f"\n{'='*80}")

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / "gate2_experiment_results.json"
    save_data = {
        "experiment": "Experiment 3: Gate 2 LLM Routing",
        "cases_total": len(cases),
        "band_distribution": {
            "fail": len(fail_band), "ambiguous": len(ambig_band), "pass": len(pass_band),
        },
        "exp2_baseline": {"f1": round(g1e2_f1, 4), "p": round(g1e2_p, 4), "r": round(g1e2_r, 4)},
        "gate2_on_ambiguous": {
            "f1": round(g2_f1, 4) if ambig_band else None,
            "cases": len(ambig_band),
        },
        "combined": {
            "f1": round(combined_f1, 4), "p": round(combined_p, 4), "r": round(combined_r, 4),
            "fire_rate": round(fire_rate, 4), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "per_case": [
            {
                "case_id": c.case_id, "expected": c.expected, "injection": c.injection_type,
                "ratio": c.grounding_ratio, "gate1": c.gate1_decision,
                "gate2": c.gate2_decision, "final": c.final_decision,
            }
            for c in cases
        ],
    }
    with save_path.open("w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nFull results saved: {save_path}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Experiment 3: Gate 2 LLM Routing")
    parser.add_argument("--max-cases", type=int, default=500)
    args = parser.parse_args()

    # Step 1: Gate 1
    cases = compute_gate1(max_cases=args.max_cases)

    # Step 2: Gate 2
    run_gate2(cases)

    # Step 3: Combine
    compute_final_decisions(cases)

    # Step 4: Report
    print_report(cases)


if __name__ == "__main__":
    main()
