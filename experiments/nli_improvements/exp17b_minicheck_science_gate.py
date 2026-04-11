"""
Experiment 17b: MiniCheck Science Gate — L2 NLI replacement.

Compares MiniCheck (Flan-T5-Large fine-tuned for factual consistency)
vs DeBERTa NLI on the same sentences from funnel analysis.

MiniCheck takes (document, claim) → Yes/No. No retrieval step needed —
it handles the full document internally.

Usage:
    poetry run python experiments/nli_improvements/exp17b_minicheck_science_gate.py
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
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
MINICHECK_MODEL = "lytang/MiniCheck-Flan-T5-Large"

MINICHECK_PROMPT = (
    "Determine whether the following claim is consistent with "
    "the corresponding document.\nDocument: {document}\nClaim: {claim}"
)


# --- Shared helpers ---


def deterministic_match(
    sentence: str,
    source_sentences: list[str],
    source_full: str,
) -> bool:
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


def nli_classify(
    tokenizer: Any,
    model: Any,
    premise: str,
    hypothesis: str,
    labels: list[str],
) -> dict[str, float]:
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(p, 4) for label, p in zip(labels, probs)}


# --- MiniCheck wrapper ---


def minicheck_score(
    sentence: str,
    source_doc: str,
    mc_tokenizer: Any,
    mc_model: Any,
) -> tuple[bool, float]:
    """
    Score a sentence against source using MiniCheck.
    Returns (is_supported, confidence).
    """
    prompt = MINICHECK_PROMPT.format(
        document=source_doc[:3500],
        claim=sentence,
    )
    inputs = mc_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    )

    with torch.no_grad():
        outputs = mc_model.generate(
            **inputs,
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode the generated token (T5 decoder output is just generated tokens, no input)
    generated = mc_tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True,
    ).strip()

    # Get confidence from logits of first generated token
    if outputs.scores:
        first_token_logits = outputs.scores[0][0]
        probs = torch.softmax(first_token_logits, dim=-1)

        # MiniCheck outputs "1" (supported) / "0" (unsupported)
        sup_ids = mc_tokenizer.encode("1", add_special_tokens=False)
        unsup_ids = mc_tokenizer.encode("0", add_special_tokens=False)

        sup_prob = probs[sup_ids[0]].item() if sup_ids else 0.0
        unsup_prob = probs[unsup_ids[0]].item() if unsup_ids else 0.0

        confidence = (
            sup_prob / (sup_prob + unsup_prob) if (sup_prob + unsup_prob) > 0 else 0.5
        )
    else:
        confidence = 1.0 if generated == "1" else 0.0

    is_supported = generated == "1"
    return is_supported, round(confidence, 4)


# --- Main experiment ---


@dataclass
class SentenceResult:
    sentence_idx: int
    sentence: str
    layer: str
    nli_entailment: float = 0.0
    minicheck_score: float = 0.0
    minicheck_supported: bool = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 17b: MiniCheck Science Gate")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--nli-threshold", type=float, default=0.7)
    parser.add_argument("--minicheck-threshold", type=float, default=0.5)
    parser.add_argument(
        "--output-dir", type=str, default="experiments/nli_improvements"
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    provider = get_embedding_provider()

    # Load DeBERTa NLI (current L2)
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    nli_labels = [
        nli_model.config.id2label[i].upper()
        for i in range(len(nli_model.config.id2label))
    ]

    # Load MiniCheck (proposed L2)
    print(f"Loading MiniCheck: {MINICHECK_MODEL}")
    mc_start = time.time()
    mc_tokenizer = AutoTokenizer.from_pretrained(MINICHECK_MODEL, use_fast=False)
    mc_model = AutoModelForSeq2SeqLM.from_pretrained(MINICHECK_MODEL)
    mc_model.eval()
    print(f"  MiniCheck loaded in {time.time() - mc_start:.1f}s")

    # Track funnel for both
    adapter = RAGTruthAdapter()
    total_cases = 0
    gate1_fail = 0
    total_sentences = 0
    l0_deterministic = 0

    l2_nli_only = 0
    l2_mc_only = 0
    l2_both = 0
    l2_neither = 0

    all_results: list[SentenceResult] = []
    start_time = time.time()

    for case in adapter.load_cases(max_cases=args.max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue

        total_cases += 1
        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        source_doc = "\n".join(ctx_parts) if ctx_parts else context
        response = case.request.candidate_answer

        ratio, min_sim = _compute_grounding_ratio(
            response,
            context,
            similarity_threshold=0.60,
        )
        if ratio < 0.80 or min_sim < 0.30:
            gate1_fail += 1
            continue

        resp_sents = _split_sentences(response)
        ctx_sents = _split_sentences(context)
        if not resp_sents or not ctx_sents:
            continue

        resp_embs = provider.encode(resp_sents)
        ctx_embs = provider.encode(ctx_sents)

        for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
            total_sentences += 1

            if deterministic_match(sent, ctx_sents, source_doc):
                l0_deterministic += 1
                all_results.append(SentenceResult(i, sent[:100], "L0"))
                continue

            # L2a: DeBERTa NLI (current)
            sims = [
                (j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            best_entailment = 0.0
            for src_idx, _ in sims[:3]:
                nli = nli_classify(
                    nli_tokenizer,
                    nli_model,
                    ctx_sents[src_idx],
                    sent,
                    nli_labels,
                )
                best_entailment = max(best_entailment, nli.get("ENTAILMENT", 0))
            nli_grounded = best_entailment > args.nli_threshold

            # L2b: MiniCheck (proposed)
            mc_supported, mc_conf = minicheck_score(
                sent,
                source_doc,
                mc_tokenizer,
                mc_model,
            )

            if nli_grounded and mc_supported:
                l2_both += 1
                all_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_both",
                        nli_entailment=best_entailment,
                        minicheck_score=mc_conf,
                        minicheck_supported=True,
                    )
                )
            elif nli_grounded and not mc_supported:
                l2_nli_only += 1
                all_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_nli_only",
                        nli_entailment=best_entailment,
                        minicheck_score=mc_conf,
                    )
                )
            elif mc_supported and not nli_grounded:
                l2_mc_only += 1
                all_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_mc_only",
                        nli_entailment=best_entailment,
                        minicheck_score=mc_conf,
                        minicheck_supported=True,
                    )
                )
            else:
                l2_neither += 1
                all_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L4_needed",
                        nli_entailment=best_entailment,
                        minicheck_score=mc_conf,
                    )
                )

        if total_cases % 10 == 0:
            elapsed_so_far = time.time() - start_time
            print(f"  Processed {total_cases} cases... ({elapsed_so_far:.0f}s)")

    elapsed = time.time() - start_time
    remaining = total_sentences - l0_deterministic
    nli_total = l2_both + l2_nli_only
    mc_total = l2_both + l2_mc_only

    print(f"\n{'='*70}")
    print("EXPERIMENT 17b: MINICHECK vs DeBERTa NLI")
    print(f"{'='*70}")
    print(f"Cases: {total_cases}, Gate 1 FAIL: {gate1_fail}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("\nFUNNEL:")
    print(f"  Total sentences:     {total_sentences}")
    print(
        f"  L0 Deterministic:    {l0_deterministic:>4} "
        f"({l0_deterministic/max(1,total_sentences)*100:.0f}%)"
    )
    print(f"  Remaining for L2:    {remaining:>4}")

    print(f"\nL2 COMPARISON (of {remaining} remaining sentences):")
    print(
        f"  DeBERTa NLI catches: {nli_total:>4} "
        f"({nli_total/max(1,remaining)*100:.0f}%)"
    )
    print(
        f"  MiniCheck catches:   {mc_total:>4} "
        f"({mc_total/max(1,remaining)*100:.0f}%)"
    )
    print(f"  Improvement:         {mc_total - nli_total:>+4} sentences")

    print("\n  Breakdown:")
    print(f"    Both catch:        {l2_both:>4}")
    print(f"    DeBERTa only:      {l2_nli_only:>4}")
    print(f"    MiniCheck only:    {l2_mc_only:>4}")
    print(f"    Neither (to L4):   {l2_neither:>4}")

    nli_free = l0_deterministic + nli_total
    mc_free = l0_deterministic + mc_total
    print("\nTOTAL FREE (L0 + L2):")
    print(
        f"  With DeBERTa:        {nli_free:>4}/{total_sentences} "
        f"({nli_free/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  With MiniCheck:      {mc_free:>4}/{total_sentences} "
        f"({mc_free/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L4 with DeBERTa:     {total_sentences - nli_free:>4} "
        f"({(total_sentences-nli_free)/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L4 with MiniCheck:   {total_sentences - mc_free:>4} "
        f"({(total_sentences-mc_free)/max(1,total_sentences)*100:.0f}%)"
    )

    mc_only_sents = [r for r in all_results if r.layer == "L2_mc_only"]
    if mc_only_sents:
        print(f"\n--- MiniCheck catches that DeBERTa misses ({len(mc_only_sents)}) ---")
        for r in mc_only_sents[:10]:
            print(
                f"  NLI={r.nli_entailment:.3f} MC={r.minicheck_score:.3f} "
                f"{r.sentence}"
            )

    nli_only_sents = [r for r in all_results if r.layer == "L2_nli_only"]
    if nli_only_sents:
        print(
            f"\n--- DeBERTa catches that MiniCheck misses ({len(nli_only_sents)}) ---"
        )
        for r in nli_only_sents[:10]:
            print(
                f"  NLI={r.nli_entailment:.3f} MC={r.minicheck_score:.3f} "
                f"{r.sentence}"
            )
    print(f"{'='*70}")

    save_data = {
        "experiment": "Experiment 17b: MiniCheck vs DeBERTa NLI",
        "total_cases": total_cases,
        "total_sentences": total_sentences,
        "l0_deterministic": l0_deterministic,
        "nli_threshold": args.nli_threshold,
        "minicheck_threshold": args.minicheck_threshold,
        "l2_nli_total": nli_total,
        "l2_mc_total": mc_total,
        "l2_both": l2_both,
        "l2_nli_only": l2_nli_only,
        "l2_mc_only": l2_mc_only,
        "l2_neither": l2_neither,
        "free_pct_nli": round(nli_free / max(1, total_sentences) * 100, 1),
        "free_pct_mc": round(mc_free / max(1, total_sentences) * 100, 1),
        "sentences": [
            {
                "idx": r.sentence_idx,
                "sentence": r.sentence,
                "layer": r.layer,
                "nli": r.nli_entailment,
                "minicheck": r.minicheck_score,
            }
            for r in all_results
        ],
    }
    save_path = output_dir / "exp17b_minicheck_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
