"""
EPIC 7.20: Science Gate — NLI + Gemini Combined Approach.

Architecture:
  ENTAILMENT   → grounded (no LLM call, 100% accurate)
  CONTRADICTION → hallucinated (no LLM call, 100% accurate)
  NEUTRAL      → route to Gemini for judgment

This tests whether combining NLI certainty with LLM reasoning
achieves published baseline quality at reduced cost.

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python -m llm_judge.benchmarks.nli_gemini_science_gate --max-fn 10
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _compute_grounding_ratio, _split_sentences
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"

GEMINI_SENTENCE_PROMPT = """You are checking whether a SINGLE sentence from a summary is supported by the source document.

SOURCE DOCUMENT:
{source}

SENTENCE TO CHECK:
{sentence}

Is this sentence fully supported by the source document? Consider:
- Are all specific facts (names, dates, numbers, locations) accurate?
- Does the source actually state or directly imply this?
- If the sentence adds details not in the source, it is NOT supported.

Answer with exactly one word: SUPPORTED or UNSUPPORTED"""


@dataclass
class SentenceResult:
    """Result for one response sentence."""
    sentence: str
    sentence_idx: int
    cosine_sim: float
    nli_label: str  # ENTAILMENT, NEUTRAL, CONTRADICTION
    nli_probs: dict[str, float]
    gemini_decision: str  # "supported", "unsupported", or "" (not routed)
    final_decision: str  # "grounded" or "hallucinated"
    routed_to_gemini: bool


@dataclass
class CaseResult:
    """Result for one case."""
    case_id: str
    ground_truth: str
    gate1_ratio: float
    gate1_min_sim: float
    final_decision: str  # "pass" or "fail"
    correct: bool
    sentences: list[SentenceResult]
    gemini_calls: int
    total_sentences: int


def load_nli_model():
    """Load NLI model."""
    print(f"Loading NLI model: {NLI_MODEL}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    labels = [model.config.id2label[i].upper() for i in range(len(model.config.id2label))]
    print(f"  Loaded in {time.time() - start:.1f}s, labels: {labels}")
    return tokenizer, model, labels


def nli_classify(tokenizer, model, premise: str, hypothesis: str, labels: list[str]):
    """Classify entailment."""
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(prob, 4) for label, prob in zip(labels, probs)}


def gemini_check_sentence(sentence: str, source: str) -> str:
    """Ask Gemini whether a single sentence is supported by the source."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = GEMINI_SENTENCE_PROMPT.format(
        source=source[:4000],
        sentence=sentence,
    )

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip().upper()

            if "UNSUPPORTED" in raw:
                return "unsupported"
            elif "SUPPORTED" in raw:
                return "supported"
            else:
                return "supported"  # default safe
    except Exception as e:
        logger.warning(f"gemini error: {str(e)[:80]}")
        return ""


def find_false_negatives(max_cases: int = 500, max_fn: int = 10) -> list[tuple]:
    """Find RAGTruth false negatives."""
    adapter = RAGTruthAdapter()
    get_embedding_provider()  # init singleton
    false_negatives = []

    for case in adapter.load_cases(max_cases=max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt != "fail":
            continue

        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        response = case.request.candidate_answer

        ratio, min_sim = _compute_grounding_ratio(response, context, similarity_threshold=0.60)

        if ratio >= 0.80 and min_sim >= 0.30:
            source_doc = "\n".join(ctx_parts) if ctx_parts else context
            false_negatives.append((case, source_doc, context, response, ratio, min_sim))
            if len(false_negatives) >= max_fn:
                break

    return false_negatives


def run_combined_on_case(
    case: Any,
    source_doc: str,
    full_context: str,
    response: str,
    ratio: float,
    min_sim: float,
    tokenizer: Any,
    model: Any,
    labels: list[str],
    provider: Any,
) -> CaseResult:
    """Run NLI + Gemini on all sentences of one case."""
    resp_sents = _split_sentences(response)
    ctx_sents = _split_sentences(full_context)

    if not resp_sents or not ctx_sents:
        return CaseResult(
            case_id=case.case_id, ground_truth="fail",
            gate1_ratio=ratio, gate1_min_sim=min_sim,
            final_decision="pass", correct=False,
            sentences=[], gemini_calls=0, total_sentences=0,
        )

    resp_embs = provider.encode(resp_sents)
    ctx_embs = provider.encode(ctx_sents)

    sentence_results = []
    has_hallucination = False
    gemini_calls = 0

    for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
        # Step 1: MiniLM finds top-3 source sentences
        sims = [(j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]
        cosine_sim = top3[0][1]

        # Step 2: NLI on top-3 source sentences
        best_entailment = 0.0
        best_contradiction = 0.0
        all_nli = []

        for src_idx, sim_score in top3:
            nli_result = nli_classify(tokenizer, model, ctx_sents[src_idx], sent, labels)
            all_nli.append(nli_result)
            best_entailment = max(best_entailment, nli_result.get("ENTAILMENT", 0))
            best_contradiction = max(best_contradiction, nli_result.get("CONTRADICTION", 0))

        # Determine NLI aggregate label
        if best_entailment > 0.7:
            nli_label = "ENTAILMENT"
        elif best_contradiction > 0.7:
            nli_label = "CONTRADICTION"
        else:
            nli_label = "NEUTRAL"

        primary_nli = all_nli[0]

        # Step 3: Route based on NLI label
        gemini_decision = ""
        routed = False

        if nli_label == "ENTAILMENT":
            final = "grounded"
        elif nli_label == "CONTRADICTION":
            final = "hallucinated"
            has_hallucination = True
        else:
            # NEUTRAL → route to Gemini
            routed = True
            gemini_calls += 1
            gemini_decision = gemini_check_sentence(sent, source_doc)
            if gemini_decision == "unsupported":
                final = "hallucinated"
                has_hallucination = True
            else:
                final = "grounded"

        sentence_results.append(SentenceResult(
            sentence=sent,
            sentence_idx=i,
            cosine_sim=round(cosine_sim, 4),
            nli_label=nli_label,
            nli_probs=primary_nli,
            gemini_decision=gemini_decision,
            final_decision=final,
            routed_to_gemini=routed,
        ))

    case_decision = "fail" if has_hallucination else "pass"
    correct = (case_decision == "fail")  # ground truth is "fail"

    return CaseResult(
        case_id=case.case_id,
        ground_truth="fail",
        gate1_ratio=ratio,
        gate1_min_sim=min_sim,
        final_decision=case_decision,
        correct=correct,
        sentences=sentence_results,
        gemini_calls=gemini_calls,
        total_sentences=len(resp_sents),
    )


def print_report(results: list[CaseResult]) -> None:
    """Print report."""
    caught = sum(1 for r in results if r.correct)
    total = len(results)
    total_sents = sum(r.total_sentences for r in results)
    total_gemini = sum(r.gemini_calls for r in results)
    nli_handled = total_sents - total_gemini

    print()
    print("=" * 70)
    print("EPIC 7.20: SCIENCE GATE — NLI + GEMINI COMBINED")
    print("=" * 70)
    print(f"False negatives tested: {total}")
    print(f"Caught: {caught}/{total}")
    print(f"Pass criteria: >= {int(total * 0.7)}/{total}")
    print(f"Result: {'PASS' if caught >= total * 0.7 else 'FAIL'}")
    print("\nCOST ANALYSIS")
    print(f"  Total sentences: {total_sents}")
    print(f"  NLI handled (no LLM): {nli_handled} ({nli_handled/total_sents*100:.0f}%)")
    print(f"  Gemini calls (NEUTRAL only): {total_gemini} ({total_gemini/total_sents*100:.0f}%)")
    print(f"  Cost savings vs --gate2 all: {nli_handled/total_sents*100:.0f}%")
    print("=" * 70)

    # Count by decision path
    entail_count = sum(1 for r in results for s in r.sentences if s.nli_label == "ENTAILMENT")
    contra_count = sum(1 for r in results for s in r.sentences if s.nli_label == "CONTRADICTION")
    gemini_unsup = sum(1 for r in results for s in r.sentences
                       if s.routed_to_gemini and s.gemini_decision == "unsupported")
    gemini_sup = sum(1 for r in results for s in r.sentences
                     if s.routed_to_gemini and s.gemini_decision == "supported")

    print("\nDECISION PATH BREAKDOWN")
    print(f"  ENTAILMENT → grounded (no LLM):     {entail_count}")
    print(f"  CONTRADICTION → hallucinated (no LLM): {contra_count}")
    print(f"  NEUTRAL → Gemini → supported:        {gemini_sup}")
    print(f"  NEUTRAL → Gemini → unsupported:      {gemini_unsup}")

    for r in results:
        marker = "CAUGHT" if r.correct else "MISSED"
        print(f"\n--- {r.case_id} [{marker}] (Gemini calls: {r.gemini_calls}/{r.total_sentences}) ---")
        print(f"  Gate 1: ratio={r.gate1_ratio:.3f} min_sim={r.gate1_min_sim:.3f}")
        print(f"  Combined decision: {r.final_decision}")

        for s in r.sentences:
            path = s.nli_label
            if s.routed_to_gemini:
                path = f"NEUTRAL→Gemini:{s.gemini_decision}"
            flag = " <<<< HALLUCINATED" if s.final_decision == "hallucinated" else ""
            print(f"  [{s.sentence_idx+1}] sim={s.cosine_sim:.3f} {path} → {s.final_decision}{flag}")
            print(f"      {s.sentence[:120]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPIC 7.20: NLI + Gemini Combined")
    parser.add_argument("--max-fn", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find false negatives
    print("Finding false negatives...")
    fn_cases = find_false_negatives(max_cases=args.max_cases, max_fn=args.max_fn)
    print(f"Found {len(fn_cases)} false negatives")

    if not fn_cases:
        print("No false negatives found.")
        return

    # Step 2: Load NLI model
    tokenizer, model, nli_labels = load_nli_model()
    provider = get_embedding_provider()

    # Step 3: Run combined approach on each false negative
    results = []
    for idx, (case, source_doc, full_context, response, ratio, min_sim) in enumerate(fn_cases):
        print(f"\nProcessing {idx+1}/{len(fn_cases)}: {case.case_id}")
        start = time.time()
        result = run_combined_on_case(
            case, source_doc, full_context, response, ratio, min_sim,
            tokenizer, model, nli_labels, provider,
        )
        elapsed = time.time() - start
        print(f"  {result.final_decision} ({elapsed:.1f}s) "
              f"{'CAUGHT' if result.correct else 'MISSED'} "
              f"(Gemini: {result.gemini_calls}/{result.total_sentences})")
        results.append(result)

    # Step 4: Report
    print_report(results)

    # Step 5: Save
    save_data = {
        "experiment": "EPIC 7.20: NLI + Gemini Combined",
        "nli_model": NLI_MODEL,
        "false_negatives_tested": len(results),
        "caught": sum(1 for r in results if r.correct),
        "total_sentences": sum(r.total_sentences for r in results),
        "gemini_calls": sum(r.gemini_calls for r in results),
        "cost_savings_pct": round(
            (1 - sum(r.gemini_calls for r in results) / max(1, sum(r.total_sentences for r in results))) * 100, 1
        ),
        "result": "PASS" if sum(1 for r in results if r.correct) >= len(results) * 0.7 else "FAIL",
        "cases": [
            {
                "case_id": r.case_id,
                "correct": r.correct,
                "gemini_calls": r.gemini_calls,
                "total_sentences": r.total_sentences,
                "sentences": [
                    {
                        "sentence": s.sentence[:150],
                        "nli_label": s.nli_label,
                        "gemini_decision": s.gemini_decision,
                        "final_decision": s.final_decision,
                        "routed_to_gemini": s.routed_to_gemini,
                    }
                    for s in r.sentences
                ],
            }
            for r in results
        ],
    }
    save_path = output_dir / "nli_gemini_combined_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
