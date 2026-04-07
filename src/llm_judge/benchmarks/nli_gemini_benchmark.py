"""
NLI + Gemini Combined Benchmark on RAGTruth.

Runs the full NLI + Gemini architecture on RAGTruth to compute
F1/Precision/Recall against published baselines.

Architecture:
  Gate 1: MiniLM cosine similarity (dual threshold)
    FAIL → FAIL (no further checks)
    PASS → NLI checks each sentence:
      ENTAILMENT    → grounded (no LLM)
      CONTRADICTION → hallucinated (no LLM)
      NEUTRAL       → Gemini per-sentence check

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python -m llm_judge.benchmarks.nli_gemini_benchmark --max-cases 50
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
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
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
class CaseResult:
    case_id: str
    ground_truth: str  # "pass" or "fail"
    gate1_decision: str  # "pass" or "fail"
    nli_gemini_decision: str  # "pass" or "fail"
    final_decision: str  # "pass" or "fail"
    total_sentences: int
    gemini_calls: int
    nli_entailments: int
    nli_contradictions: int
    nli_neutrals: int
    elapsed: float


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
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(prob, 4) for label, prob in zip(labels, probs)}


def gemini_check_sentence(sentence: str, source: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    prompt = GEMINI_SENTENCE_PROMPT.format(source=source[:4000], sentence=sentence)
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
            return "supported"
    except Exception as e:
        logger.warning(f"gemini error: {str(e)[:80]}")
        return ""


def evaluate_case(
    case: Any,
    tokenizer: Any,
    model: Any,
    nli_labels: list[str],
    provider: Any,
) -> CaseResult:
    """Evaluate one case with Gate 1 + NLI + Gemini."""
    start = time.time()

    gt = case.ground_truth.property_labels.get("1.1", "pass")
    ctx_parts = list(case.request.source_context or [])
    conv = " ".join(msg.content for msg in case.request.conversation)
    context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
    source_doc = "\n".join(ctx_parts) if ctx_parts else context
    response = case.request.candidate_answer

    # Gate 1: MiniLM dual threshold
    ratio, min_sim = _compute_grounding_ratio(response, context, similarity_threshold=0.60)
    fail_ratio = ratio < 0.80
    fail_min = min_sim < 0.30

    if fail_ratio or fail_min:
        # Gate 1 FAIL — no further checks needed
        elapsed = time.time() - start
        return CaseResult(
            case_id=case.case_id, ground_truth=gt,
            gate1_decision="fail", nli_gemini_decision="",
            final_decision="fail",
            total_sentences=0, gemini_calls=0,
            nli_entailments=0, nli_contradictions=0, nli_neutrals=0,
            elapsed=elapsed,
        )

    # Gate 1 PASS — run NLI + Gemini on each sentence
    resp_sents = _split_sentences(response)
    ctx_sents = _split_sentences(context)

    if not resp_sents or not ctx_sents:
        elapsed = time.time() - start
        return CaseResult(
            case_id=case.case_id, ground_truth=gt,
            gate1_decision="pass", nli_gemini_decision="pass",
            final_decision="pass",
            total_sentences=0, gemini_calls=0,
            nli_entailments=0, nli_contradictions=0, nli_neutrals=0,
            elapsed=elapsed,
        )

    resp_embs = provider.encode(resp_sents)
    ctx_embs = provider.encode(ctx_sents)

    has_hallucination = False
    gemini_calls = 0
    entailments = 0
    contradictions = 0
    neutrals = 0

    for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
        # Find top-3 source sentences
        sims = [(j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)]
        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]

        # NLI on top-3
        best_entailment = 0.0
        best_contradiction = 0.0
        for src_idx, sim_score in top3:
            nli = nli_classify(tokenizer, model, ctx_sents[src_idx], sent, nli_labels)
            best_entailment = max(best_entailment, nli.get("ENTAILMENT", 0))
            best_contradiction = max(best_contradiction, nli.get("CONTRADICTION", 0))

        if best_entailment > 0.7:
            entailments += 1
            # Grounded — no LLM needed
        elif best_contradiction > 0.7:
            contradictions += 1
            has_hallucination = True
        else:
            neutrals += 1
            # Route to Gemini
            gemini_calls += 1
            gemini_result = gemini_check_sentence(sent, source_doc)
            if gemini_result == "unsupported":
                has_hallucination = True

    nli_gemini_decision = "fail" if has_hallucination else "pass"
    elapsed = time.time() - start

    return CaseResult(
        case_id=case.case_id, ground_truth=gt,
        gate1_decision="pass", nli_gemini_decision=nli_gemini_decision,
        final_decision=nli_gemini_decision,
        total_sentences=len(resp_sents), gemini_calls=gemini_calls,
        nli_entailments=entailments, nli_contradictions=contradictions,
        nli_neutrals=neutrals,
        elapsed=elapsed,
    )


def main():
    parser = argparse.ArgumentParser(description="NLI + Gemini RAGTruth Benchmark")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    tokenizer, model, nli_labels = load_nli_model()
    provider = get_embedding_provider()

    # Load and evaluate
    adapter = RAGTruthAdapter()
    results: list[CaseResult] = []
    total_start = time.time()

    for case in adapter.load_cases(max_cases=args.max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue

        idx = len(results)
        if idx % 10 == 0:
            print(f"  Processing case {idx+1}...")

        result = evaluate_case(case, tokenizer, model, nli_labels, provider)
        results.append(result)

    total_elapsed = time.time() - total_start

    # Compute metrics
    tp = fp = tn = fn = 0
    for r in results:
        if r.final_decision == "fail" and r.ground_truth == "fail":
            tp += 1
        elif r.final_decision == "fail" and r.ground_truth == "pass":
            fp += 1
        elif r.final_decision == "pass" and r.ground_truth == "pass":
            tn += 1
        elif r.final_decision == "pass" and r.ground_truth == "fail":
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    total_sents = sum(r.total_sentences for r in results)
    total_gemini = sum(r.gemini_calls for r in results)
    total_entail = sum(r.nli_entailments for r in results)
    total_contra = sum(r.nli_contradictions for r in results)
    total_neutral = sum(r.nli_neutrals for r in results)
    gate1_fails = sum(1 for r in results if r.gate1_decision == "fail")

    # Report
    print()
    print("=" * 70)
    print("NLI + GEMINI COMBINED BENCHMARK: RAGTruth")
    print("=" * 70)
    print(f"Cases evaluated: {len(results)}")
    print(f"Elapsed: {total_elapsed:.1f}s")
    print()
    print("RESPONSE-LEVEL RESULTS")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  (TP={tp} FP={fp} TN={tn} FN={fn})")
    print()
    print("PUBLISHED BASELINES (comparison)")
    print(f"  GPT-4-turbo: F1=0.635 [ours: {f1:.3f}, delta: {f1 - 0.635:+.3f}]")
    print(f"  GPT-3.5-turbo: F1=0.543 [ours: {f1:.3f}, delta: {f1 - 0.543:+.3f}]")
    print(f"  Llama-2-13B (fine-tuned): F1=0.622 [ours: {f1:.3f}, delta: {f1 - 0.622:+.3f}]")
    print()
    print("ARCHITECTURE BREAKDOWN")
    print(f"  Gate 1 FAIL (embeddings only): {gate1_fails} cases — no NLI or Gemini needed")
    print(f"  Gate 1 PASS → NLI + Gemini: {len(results) - gate1_fails} cases")
    print()
    print("COST ANALYSIS (sentences)")
    print(f"  Total sentences analysed by NLI: {total_sents}")
    print(f"  ENTAILMENT (no LLM):     {total_entail} ({total_entail/max(1,total_sents)*100:.0f}%)")
    print(f"  CONTRADICTION (no LLM):  {total_contra} ({total_contra/max(1,total_sents)*100:.0f}%)")
    print(f"  NEUTRAL → Gemini:        {total_neutral} ({total_neutral/max(1,total_sents)*100:.0f}%)")
    print(f"  Gemini calls:            {total_gemini}")
    nli_handled = total_entail + total_contra
    print(f"  NLI handled (no LLM):    {nli_handled} ({nli_handled/max(1,total_sents)*100:.0f}%)")
    print(f"  Cost savings vs all-LLM: {nli_handled/max(1,total_sents)*100:.0f}%")
    print()

    # Per-model breakdown
    for r in results:
        # Extract model from case_id pattern if available
        pass  # Would need adapter access for model info

    print(f"Avg latency per case: {total_elapsed/max(1,len(results))*1000:.0f}ms")
    print("=" * 70)

    # Save
    save_data = {
        "experiment": "NLI + Gemini Combined Benchmark on RAGTruth",
        "nli_model": NLI_MODEL,
        "cases": len(results),
        "elapsed_s": round(total_elapsed, 1),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "published_gpt4_f1": 0.635,
        "delta_vs_gpt4": round(f1 - 0.635, 4),
        "gate1_fails": gate1_fails,
        "total_sentences": total_sents,
        "nli_entailments": total_entail,
        "nli_contradictions": total_contra,
        "nli_neutrals": total_neutral,
        "gemini_calls": total_gemini,
        "cost_savings_pct": round(nli_handled / max(1, total_sents) * 100, 1),
    }
    save_path = output_dir / "nli_gemini_benchmark_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    main()
