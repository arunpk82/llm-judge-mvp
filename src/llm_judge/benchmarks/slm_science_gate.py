"""
Experiment 14: SLM Science Gate — Can Qwen2.5-3B match Gemini on L4 reasoning?

Tests whether a local 3B parameter model can match Gemini's F1=0.900
on the L4 sentences that need reasoning. Uses raw source text context
(validated as the winning strategy in Experiment 13).

Same 20 cases, same L4 sentences, same prompt — only the model changes.

Usage:
    # Install model first (downloads ~6GB):
    poetry run python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

    # Run experiment:
    poetry run python -m llm_judge.benchmarks.slm_science_gate --max-fn 10 --max-tn 10
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
DEFAULT_SLM = "Qwen/Qwen2.5-3B-Instruct"

PROMPT_TEMPLATE = """You are checking whether a SINGLE sentence from a summary is supported by the source document.

SOURCE DOCUMENT:
{source}

SENTENCE TO CHECK:
{sentence}

Is this sentence fully supported by the source document? Consider:
- Are all specific facts (names, dates, numbers, locations) accurate?
- Does the source actually state or directly imply this?
- If the sentence adds details not in the source, it is NOT supported.

Answer with exactly one word: SUPPORTED or UNSUPPORTED"""


# --- SLM inference ---

_slm_model: Any = None
_slm_tokenizer: Any = None


def load_slm(model_name: str):
    global _slm_model, _slm_tokenizer
    print(f"Loading SLM: {model_name}")
    start = time.time()
    _slm_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _slm_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    _slm_model.eval()
    elapsed = time.time() - start
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in _slm_model.parameters()):,}")


def slm_check_sentence(sentence: str, source: str) -> tuple[str, float]:
    """Check one sentence with SLM. Returns (decision, elapsed)."""
    prompt = PROMPT_TEMPLATE.format(source=source[:4000], sentence=sentence)
    messages = [{"role": "user", "content": prompt}]
    text = _slm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _slm_tokenizer([text], return_tensors="pt")

    start = time.time()
    with torch.no_grad():
        outputs = _slm_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=1.0,  # greedy with do_sample=False
            do_sample=False,
        )
    elapsed = time.time() - start

    response = (
        _slm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        .strip()
        .upper()
    )

    decision = "unsupported" if "UNSUPPORTED" in response else "supported"
    return decision, elapsed


# --- Layer filters (same as funnel_analysis) ---


def nli_classify(tokenizer, model, premise, hypothesis, labels):
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(p, 4) for label, p in zip(labels, probs)}


def deterministic_match(sentence, source_sentences, source_full):
    norm_sent = re.sub(r"\s+", " ", sentence.lower().strip())
    norm_source = re.sub(r"\s+", " ", source_full.lower().strip())
    if norm_sent in norm_source:
        return True
    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return False
    for src_sent in source_sentences:
        norm_src = re.sub(r"\s+", " ", src_sent.lower().strip())
        if SequenceMatcher(None, norm_sent, norm_src).ratio() > 0.85:
            return True
        src_tokens = set(re.findall(r"\w+", src_sent.lower()))
        if src_tokens:
            jaccard = len(sent_tokens & src_tokens) / len(sent_tokens | src_tokens)
            if jaccard > 0.80:
                return True
    return False


def find_test_cases(max_cases=500, max_fn=10, max_tn=10):
    adapter = RAGTruthAdapter()
    get_embedding_provider()  # init singleton
    fns: list[tuple] = []
    tns: list[tuple] = []
    for case in adapter.load_cases(max_cases=max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue
        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        ctx = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        resp = case.request.candidate_answer
        ratio, min_sim = _compute_grounding_ratio(resp, ctx, similarity_threshold=0.60)
        if ratio >= 0.80 and min_sim >= 0.30:
            src = "\n".join(ctx_parts) if ctx_parts else ctx
            entry = (case, src, ctx, resp, ratio, min_sim)
            if gt == "fail" and len(fns) < max_fn:
                fns.append(entry)
            elif gt == "pass" and len(tns) < max_tn:
                tns.append(entry)
        if len(fns) >= max_fn and len(tns) >= max_tn:
            break
    return fns, tns


def main():
    parser = argparse.ArgumentParser(description="Experiment 14: SLM Science Gate")
    parser.add_argument("--max-fn", type=int, default=10)
    parser.add_argument("--max-tn", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument("--slm-model", type=str, default=DEFAULT_SLM)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"SLM model: {args.slm_model}")
    print("Finding test cases...")
    fn_cases, tn_cases = find_test_cases(args.max_cases, args.max_fn, args.max_tn)
    print(f"Found {len(fn_cases)} FN + {len(tn_cases)} TN")

    # Load models
    print("Loading models...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    nli_labels = [
        nli_model.config.id2label[i].upper()
        for i in range(len(nli_model.config.id2label))
    ]
    provider = get_embedding_provider()

    # Load SLM
    load_slm(args.slm_model)

    # Process cases
    all_results = []
    total_l4 = 0
    total_slm_time = 0.0

    for label, cases, gt_val in [("FN", fn_cases, "fail"), ("TN", tn_cases, "pass")]:
        for idx, (case, src, ctx, resp, ratio, min_sim) in enumerate(cases):
            print(f"\n{label} {idx+1}/{len(cases)}: {case.case_id}")
            case_start = time.time()

            resp_sents = _split_sentences(resp)
            ctx_sents = _split_sentences(ctx)
            if not resp_sents or not ctx_sents:
                continue

            resp_embs = provider.encode(resp_sents)
            ctx_embs = provider.encode(ctx_sents)

            has_hallucination = False
            l4_count = 0
            l4_details = []

            for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
                # L0: Deterministic
                if deterministic_match(sent, ctx_sents, src):
                    continue

                # L2: NLI
                sims = [
                    (j, provider.max_similarity(emb, [ce]))
                    for j, ce in enumerate(ctx_embs)
                ]
                sims.sort(key=lambda x: x[1], reverse=True)
                best_e = 0.0
                for src_idx, _ in sims[:3]:
                    nli = nli_classify(
                        nli_tokenizer, nli_model, ctx_sents[src_idx], sent, nli_labels
                    )
                    best_e = max(best_e, nli.get("ENTAILMENT", 0))

                if best_e > 0.7:
                    continue

                # L4: SLM reasoning (skip L3 GraphRAG for this experiment)
                l4_count += 1
                total_l4 += 1
                decision, slm_elapsed = slm_check_sentence(sent, src)
                total_slm_time += slm_elapsed

                if decision == "unsupported":
                    has_hallucination = True

                l4_details.append(
                    {
                        "sentence_idx": i,
                        "sentence": sent[:120],
                        "decision": decision,
                        "elapsed": round(slm_elapsed, 1),
                    }
                )

            case_decision = "fail" if has_hallucination else "pass"
            correct = case_decision == gt_val
            case_elapsed = time.time() - case_start

            print(
                f"  decision={case_decision} gt={gt_val} L4={l4_count} "
                f"{'CORRECT' if correct else 'WRONG'} ({case_elapsed:.1f}s)"
            )

            all_results.append(
                {
                    "case_id": case.case_id,
                    "gt": gt_val,
                    "decision": case_decision,
                    "correct": correct,
                    "l4_count": l4_count,
                    "l4_details": l4_details,
                }
            )

    # Compute metrics
    fn_results = [r for r in all_results if r["gt"] == "fail"]
    tn_results = [r for r in all_results if r["gt"] == "pass"]

    tp = sum(1 for r in fn_results if r["decision"] == "fail")
    fn = sum(1 for r in fn_results if r["decision"] == "pass")
    fp = sum(1 for r in tn_results if r["decision"] == "fail")
    tn = sum(1 for r in tn_results if r["decision"] == "pass")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"\n{'='*70}")
    print("EXPERIMENT 14: SLM SCIENCE GATE")
    print(f"{'='*70}")
    print(f"Model: {args.slm_model}")
    print(
        f"Cases: {len(fn_results)} FN + {len(tn_results)} TN = {len(all_results)} total"
    )
    print(f"L4 sentences: {total_l4}")
    print(f"Avg SLM latency: {total_slm_time/max(1,total_l4):.1f}s per sentence")
    print()
    print("RESULTS")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
    print(f"  FN caught: {tp}/{len(fn_results)}  TN FP: {fp}/{len(tn_results)}")
    print()
    print("COMPARISON vs GEMINI (Experiment 13)")
    print("  Gemini F1:       0.900")
    print(f"  SLM F1:          {f1:.3f}")
    print(f"  Delta:           {f1 - 0.900:+.3f}")
    print("  Pass criteria:   F1 >= 0.800")
    print(f"  Result:          {'PASS' if f1 >= 0.800 else 'FAIL'}")
    print(f"{'='*70}")

    # Per-case details
    print("\n--- L4 SENTENCE DETAILS ---")
    for r in all_results:
        if not r["l4_details"]:
            continue
        marker = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n  {r['case_id']} (gt={r['gt']}) [{marker}]")
        for d in r["l4_details"]:
            print(
                f"    [{d['sentence_idx']+1}] {d['decision']:>11} ({d['elapsed']:.1f}s) {d['sentence']}"
            )

    # Save
    save_data = {
        "experiment": "Experiment 14: SLM Science Gate",
        "slm_model": args.slm_model,
        "fn_tested": len(fn_results),
        "tn_tested": len(tn_results),
        "l4_sentences": total_l4,
        "avg_slm_latency": round(total_slm_time / max(1, total_l4), 1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gemini_f1": 0.900,
        "delta": round(f1 - 0.900, 4),
        "pass_criteria": "F1 >= 0.800",
        "result": "PASS" if f1 >= 0.800 else "FAIL",
        "cases": all_results,
    }
    save_path = output_dir / "slm_science_gate_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
