"""
Experiment 13: SLM + GraphRAG Science Gate.

Architecture: For sentences that reach L4 (39% that no deterministic method handles):
  1. spaCy extracts SVO triplets from source document (GraphRAG)
  2. For each L4 sentence, extract its SVO triplets
  3. Query source graph for relevant triplets (by entity overlap)
  4. Pass structured facts + sentence to reasoning model
  5. Reasoning model judges SUPPORTED/UNSUPPORTED

Tests TWO context strategies:
  A. Raw source text (baseline — same as Experiment 9)
  B. GraphRAG structured triplets (hypothesis: better precision)

Supports --backend gemini (API, fast) or --backend slm (local, free)

Usage:
    # With Gemini (fast validation):
    export GEMINI_API_KEY=your_key
    poetry run python -m llm_judge.benchmarks.slm_graphrag_science_gate --max-fn 10 --max-tn 10 --backend gemini

    # With local SLM (production target):
    poetry run python -m llm_judge.benchmarks.slm_graphrag_science_gate --max-fn 10 --max-tn 10 --backend slm
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge.benchmarks.graphrag_science_gate import (
    Triplet,
    _entity_overlap,
    extract_svo_triplets,
)
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
SPACY_MODEL = "en_core_web_sm"

# --- Prompts ---

RAW_TEXT_PROMPT = """You are checking whether a SINGLE sentence from a summary is supported by the source document.

SOURCE DOCUMENT:
{source}

SENTENCE TO CHECK:
{sentence}

Is this sentence fully supported by the source document? Consider:
- Are all specific facts (names, dates, numbers, locations) accurate?
- Does the source actually state or directly imply this?
- If the sentence adds details not in the source, it is NOT supported.

Answer with exactly one word: SUPPORTED or UNSUPPORTED"""

GRAPHRAG_PROMPT = """You are checking whether a SINGLE sentence is supported by structured facts extracted from a source document.

SOURCE FACTS (Subject, Verb, Object triplets from the source document):
{triplets}

SENTENCE TO CHECK:
{sentence}

Based on the source facts above, is this sentence supported? Consider:
- Does any source fact confirm the key claim in this sentence?
- If the sentence states a relationship between entities, does a source fact confirm that same relationship?
- If the sentence mentions specific details (dates, numbers, names) not present in ANY source fact, it is NOT supported.
- A sentence that reasonably summarises or paraphrases source facts IS supported.

Answer with exactly one word: SUPPORTED or UNSUPPORTED"""


@dataclass
class SentenceResult:
    sentence: str
    sentence_idx: int
    layer: str  # L0/L2_entailment/L2_contradiction/L3_exact/L4_reasoning
    raw_decision: str  # for L4: supported/unsupported
    graphrag_decision: str  # for L4: supported/unsupported
    relevant_triplets: list[str]


@dataclass
class CaseResult:
    case_id: str
    ground_truth: str
    gate1_decision: str
    raw_final: str  # pass/fail using raw text for L4
    graphrag_final: str  # pass/fail using GraphRAG for L4
    correct_raw: bool
    correct_graphrag: bool
    sentences: list[SentenceResult]
    l4_count: int
    gemini_calls: int


def nli_classify(tokenizer, model, premise, hypothesis, labels):
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(p, 4) for label, p in zip(labels, probs)}


def _normalize_text(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_set(text):
    return set(re.findall(r"\w+", text.lower()))


def deterministic_match(sentence, source_sentences, source_full):
    from difflib import SequenceMatcher

    norm_sent = _normalize_text(sentence)
    norm_source = _normalize_text(source_full)
    if norm_sent in norm_source:
        return True
    sent_tokens = _token_set(sentence)
    if not sent_tokens:
        return False
    for src_sent in source_sentences:
        norm_src = _normalize_text(src_sent)
        if SequenceMatcher(None, norm_sent, norm_src).ratio() > 0.85:
            return True
        src_tokens = _token_set(src_sent)
        if src_tokens:
            jaccard = len(sent_tokens & src_tokens) / len(sent_tokens | src_tokens)
            if jaccard > 0.80:
                return True
    return False


def find_relevant_triplets(
    sentence_triplets: list[Triplet],
    source_triplets: list[Triplet],
    sentence_text: str,
) -> list[Triplet]:
    """Find source triplets relevant to a given sentence by entity overlap."""
    relevant = []
    seen = set()
    sent_tokens = _token_set(sentence_text)

    for st in source_triplets:
        st_key = str(st)
        if st_key in seen:
            continue

        # Check entity overlap with sentence text
        st_tokens = _token_set(f"{st.subject} {st.predicate} {st.obj}")
        overlap = sent_tokens & st_tokens
        if len(overlap) >= 2:
            relevant.append(st)
            seen.add(st_key)
            continue

        # Check entity overlap with sentence triplets
        for rt in sentence_triplets:
            subj_sim = _entity_overlap(rt.subject, st.subject)
            obj_sim = _entity_overlap(rt.obj, st.obj)
            if max(subj_sim, obj_sim) >= 0.5:
                relevant.append(st)
                seen.add(st_key)
                break

    return relevant


def call_reasoning_model(prompt: str, backend: str) -> str:
    """Call reasoning model (Gemini API or local SLM)."""
    if backend == "gemini":
        return _call_gemini(prompt)
    elif backend == "slm":
        return _call_slm(prompt)
    return ""


def _call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return ""
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip().upper()
            return "unsupported" if "UNSUPPORTED" in raw else "supported"
    except Exception as e:
        logger.warning(f"gemini error: {str(e)[:80]}")
        return ""


# SLM placeholder — will be implemented when model is chosen
_slm_model: Any = None
_slm_tokenizer: Any = None


def _load_slm():
    global _slm_model, _slm_tokenizer
    if _slm_model is not None:
        return
    from transformers import AutoModelForCausalLM

    slm_name = os.environ.get("SLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"  Loading SLM: {slm_name}")
    start = time.time()
    _slm_tokenizer = AutoTokenizer.from_pretrained(slm_name)
    _slm_model = AutoModelForCausalLM.from_pretrained(
        slm_name,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    _slm_model.eval()
    print(f"  SLM loaded in {time.time() - start:.1f}s")


def _call_slm(prompt: str) -> str:
    _load_slm()
    messages = [{"role": "user", "content": prompt}]
    text = _slm_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = _slm_tokenizer([text], return_tensors="pt")
    with torch.no_grad():
        outputs = _slm_model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
        )
    response = (
        _slm_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )
        .strip()
        .upper()
    )
    return "unsupported" if "UNSUPPORTED" in response else "supported"


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


def evaluate_case(
    case,
    source_doc,
    full_context,
    response,
    ratio,
    min_sim,
    gt,
    nli_tokenizer,
    nli_model,
    nli_labels,
    provider,
    nlp,
    backend,
):
    resp_sents = _split_sentences(response)
    ctx_sents = _split_sentences(full_context)
    if not resp_sents or not ctx_sents:
        return None

    resp_embs = provider.encode(resp_sents)
    ctx_embs = provider.encode(ctx_sents)
    source_triplets = extract_svo_triplets(source_doc, nlp)

    sentences = []
    raw_has_hallucination = False
    graphrag_has_hallucination = False
    l4_count = 0
    gemini_calls = 0

    for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
        # L0: Deterministic
        if deterministic_match(sent, ctx_sents, source_doc):
            sentences.append(
                SentenceResult(
                    sentence=sent,
                    sentence_idx=i,
                    layer="L0_deterministic",
                    raw_decision="",
                    graphrag_decision="",
                    relevant_triplets=[],
                )
            )
            continue

        # L2: NLI
        sims = [
            (j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        best_e, best_c = 0.0, 0.0
        for src_idx, _ in sims[:3]:
            nli = nli_classify(
                nli_tokenizer, nli_model, ctx_sents[src_idx], sent, nli_labels
            )
            best_e = max(best_e, nli.get("ENTAILMENT", 0))
            best_c = max(best_c, nli.get("CONTRADICTION", 0))

        if best_e > 0.7:
            sentences.append(
                SentenceResult(
                    sentence=sent,
                    sentence_idx=i,
                    layer="L2_entailment",
                    raw_decision="",
                    graphrag_decision="",
                    relevant_triplets=[],
                )
            )
            continue

        # L2 CONTRADICTION — still route to L4 (unreliable)
        # L3: GraphRAG exact match
        resp_triplets = extract_svo_triplets(sent, nlp)
        has_exact = False
        all_matched = True
        for rt in resp_triplets:
            if rt.obj == "(intransitive)":
                continue
            found = False
            for st in source_triplets:
                if st.obj == "(intransitive)":
                    continue
                subj_sim = _entity_overlap(rt.subject, st.subject)
                obj_sim = _entity_overlap(rt.obj, st.obj)
                pred_sim = _entity_overlap(rt.predicate, st.predicate)
                if (subj_sim + obj_sim) / 2 >= 0.5 and pred_sim >= 0.5:
                    found = True
                    break
            if found:
                has_exact = True
            else:
                all_matched = False

        if not best_c > 0.7 and resp_triplets and all_matched and has_exact:
            sentences.append(
                SentenceResult(
                    sentence=sent,
                    sentence_idx=i,
                    layer="L3_graphrag_exact",
                    raw_decision="",
                    graphrag_decision="",
                    relevant_triplets=[],
                )
            )
            continue

        # L4: Needs reasoning — test BOTH strategies
        l4_count += 1
        gemini_calls += 2  # one for raw, one for graphrag

        # Strategy A: Raw source text
        raw_prompt = RAW_TEXT_PROMPT.format(source=source_doc[:4000], sentence=sent)
        raw_decision = call_reasoning_model(raw_prompt, backend)
        if raw_decision == "unsupported":
            raw_has_hallucination = True

        # Strategy B: GraphRAG structured context
        relevant = find_relevant_triplets(resp_triplets, source_triplets, sent)
        triplet_text = (
            "\n".join(f"  - {t}" for t in relevant)
            if relevant
            else "  (No relevant source facts found for this sentence)"
        )
        graphrag_prompt = GRAPHRAG_PROMPT.format(triplets=triplet_text, sentence=sent)
        graphrag_decision = call_reasoning_model(graphrag_prompt, backend)
        if graphrag_decision == "unsupported":
            graphrag_has_hallucination = True

        sentences.append(
            SentenceResult(
                sentence=sent,
                sentence_idx=i,
                layer="L4_reasoning",
                raw_decision=raw_decision,
                graphrag_decision=graphrag_decision,
                relevant_triplets=[str(t) for t in relevant],
            )
        )

    raw_final = "fail" if raw_has_hallucination else "pass"
    graphrag_final = "fail" if graphrag_has_hallucination else "pass"

    return CaseResult(
        case_id=case.case_id,
        ground_truth=gt,
        gate1_decision="pass",
        raw_final=raw_final,
        graphrag_final=graphrag_final,
        correct_raw=(raw_final == gt),
        correct_graphrag=(graphrag_final == gt),
        sentences=sentences,
        l4_count=l4_count,
        gemini_calls=gemini_calls,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 13: SLM + GraphRAG Science Gate"
    )
    parser.add_argument("--max-fn", type=int, default=10)
    parser.add_argument("--max-tn", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=500)
    parser.add_argument(
        "--backend", type=str, default="gemini", choices=["gemini", "slm"]
    )
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Backend: {args.backend}")
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
    nlp = spacy.load(SPACY_MODEL)

    if args.backend == "slm":
        _load_slm()

    results = []
    for label, cases, gt_val in [("FN", fn_cases, "fail"), ("TN", tn_cases, "pass")]:
        for idx, (case, src, ctx, resp, ratio, min_sim) in enumerate(cases):
            print(f"\n{label} {idx+1}/{len(cases)}: {case.case_id}")
            start = time.time()
            r = evaluate_case(
                case,
                src,
                ctx,
                resp,
                ratio,
                min_sim,
                gt_val,
                nli_tokenizer,
                nli_model,
                nli_labels,
                provider,
                nlp,
                args.backend,
            )
            if r is None:
                continue
            elapsed = time.time() - start
            print(
                f"  raw={r.raw_final} graphrag={r.graphrag_final} gt={gt_val} "
                f"L4={r.l4_count} ({elapsed:.1f}s)"
            )
            results.append(r)

    # Compute metrics
    fn_results = [r for r in results if r.ground_truth == "fail"]
    tn_results = [r for r in results if r.ground_truth == "pass"]

    # Raw text strategy
    raw_tp = sum(1 for r in fn_results if r.raw_final == "fail")
    raw_fp = sum(1 for r in tn_results if r.raw_final == "fail")
    raw_tn = sum(1 for r in tn_results if r.raw_final == "pass")
    raw_fn = sum(1 for r in fn_results if r.raw_final == "pass")

    # GraphRAG strategy
    gr_tp = sum(1 for r in fn_results if r.graphrag_final == "fail")
    gr_fp = sum(1 for r in tn_results if r.graphrag_final == "fail")
    gr_tn = sum(1 for r in tn_results if r.graphrag_final == "pass")
    gr_fn = sum(1 for r in fn_results if r.graphrag_final == "pass")

    def calc_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        return p, r, f

    raw_p, raw_r, raw_f1 = calc_f1(raw_tp, raw_fp, raw_fn)
    gr_p, gr_r, gr_f1 = calc_f1(gr_tp, gr_fp, gr_fn)

    total_l4 = sum(r.l4_count for r in results)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT 13: SLM + GRAPHRAG SCIENCE GATE (backend={args.backend})")
    print(f"{'='*70}")
    print(f"Cases: {len(fn_results)} FN + {len(tn_results)} TN = {len(results)} total")
    print(f"L4 sentences (reasoning needed): {total_l4}")

    print("\n--- STRATEGY A: Raw source text ---")
    print(f"  TP={raw_tp} FP={raw_fp} TN={raw_tn} FN={raw_fn}")
    print(f"  Precision: {raw_p:.3f}  Recall: {raw_r:.3f}  F1: {raw_f1:.3f}")
    print(f"  FN caught: {raw_tp}/{len(fn_results)}  TN FP: {raw_fp}/{len(tn_results)}")

    print("\n--- STRATEGY B: GraphRAG structured triplets ---")
    print(f"  TP={gr_tp} FP={gr_fp} TN={gr_tn} FN={gr_fn}")
    print(f"  Precision: {gr_p:.3f}  Recall: {gr_r:.3f}  F1: {gr_f1:.3f}")
    print(f"  FN caught: {gr_tp}/{len(fn_results)}  TN FP: {gr_fp}/{len(tn_results)}")

    print("\n--- COMPARISON ---")
    print(f"  Raw text F1:   {raw_f1:.3f}")
    print(f"  GraphRAG F1:   {gr_f1:.3f}")
    print(f"  Delta:         {gr_f1 - raw_f1:+.3f}")
    winner = "GraphRAG" if gr_f1 > raw_f1 else ("Raw text" if raw_f1 > gr_f1 else "Tie")
    print(f"  Winner:        {winner}")
    print(f"{'='*70}")

    # Per-case details for L4 sentences
    print("\n--- L4 SENTENCE DETAILS ---")
    for r in results:
        l4_sents = [s for s in r.sentences if s.layer == "L4_reasoning"]
        if not l4_sents:
            continue
        print(f"\n  {r.case_id} (gt={r.ground_truth})")
        for s in l4_sents:
            _ = (
                "OK"
                if (s.raw_decision == "unsupported") == (r.ground_truth == "fail")
                else "WRONG"
            )
            _ = (
                "OK"
                if (s.graphrag_decision == "unsupported") == (r.ground_truth == "fail")
                else "WRONG"
            )
            # For gt=pass, supported=correct. For gt=fail, unsupported on at least one = correct.
            print(f"    [{s.sentence_idx+1}] {s.sentence[:100]}")
            print(
                f"      Raw: {s.raw_decision:>11}  GraphRAG: {s.graphrag_decision:>11}"
            )
            if s.relevant_triplets:
                for t in s.relevant_triplets[:3]:
                    print(f"      Source fact: {t}")
                if len(s.relevant_triplets) > 3:
                    print(f"      ... +{len(s.relevant_triplets)-3} more")

    # Save
    save_data = {
        "experiment": "Experiment 13: SLM + GraphRAG Science Gate",
        "backend": args.backend,
        "fn_tested": len(fn_results),
        "tn_tested": len(tn_results),
        "l4_sentences": total_l4,
        "raw_text": {
            "tp": raw_tp,
            "fp": raw_fp,
            "tn": raw_tn,
            "fn": raw_fn,
            "precision": round(raw_p, 4),
            "recall": round(raw_r, 4),
            "f1": round(raw_f1, 4),
        },
        "graphrag": {
            "tp": gr_tp,
            "fp": gr_fp,
            "tn": gr_tn,
            "fn": gr_fn,
            "precision": round(gr_p, 4),
            "recall": round(gr_r, 4),
            "f1": round(gr_f1, 4),
        },
        "winner": winner,
    }
    save_path = output_dir / "slm_graphrag_science_gate_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
