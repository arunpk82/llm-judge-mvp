"""
Experiment 17: AlignScore Science Gate — L2 NLI replacement.

Compares AlignScore vs DeBERTa NLI on the same 175 sentences from funnel analysis.
AlignScore is trained on 4.7M examples across 7 tasks (NLI, QA, paraphrasing,
fact verification, IR, semantic similarity, summarization).

Architecture:
  Same funnel as Experiment 12, but L2 uses AlignScore instead of DeBERTa NLI.
  Measures: how many more sentences get resolved at L2.

Usage:
    pip install alignscore
    python experiments/nli_improvements/exp17_alignscore_science_gate.py
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
ALIGNSCORE_MODEL = "AlignScore/AlignScore-large"


# --- Shared helpers ---


def deterministic_match(
    sentence: str, source_sentences: list[str], source_full: str
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
    tokenizer: Any, model: Any, premise: str, hypothesis: str, labels: list[str]
) -> dict[str, float]:
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(p, 4) for label, p in zip(labels, probs)}


# --- AlignScore wrapper ---

_alignscore_scorer = None


def load_alignscore() -> Any:
    global _alignscore_scorer
    if _alignscore_scorer is not None:
        return _alignscore_scorer

    try:
        from alignscore import AlignScore

        print("Loading AlignScore model...")
        start = time.time()
        _alignscore_scorer = AlignScore(
            model="roberta-large",
            batch_size=8,
            device="cpu",
            ckpt_path=ALIGNSCORE_MODEL,
            evaluation_mode="nli_sp",
        )
        print(f"  AlignScore loaded in {time.time() - start:.1f}s")
        return _alignscore_scorer
    except ImportError:
        print("ERROR: alignscore not installed. Run: pip install alignscore")
        raise
    except Exception as e:
        print(f"ERROR loading AlignScore: {e}")
        # Fallback: try loading via HuggingFace directly
        print("Trying HuggingFace fallback...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _alignscore_scorer = {
            "tokenizer": AutoTokenizer.from_pretrained("roberta-large"),
            "model": AutoModelForSequenceClassification.from_pretrained(
                ALIGNSCORE_MODEL
            ),
        }
        return _alignscore_scorer


def alignscore_check(
    sentence: str, source_sentences: list[str], top_k: int = 3, threshold: float = 0.7
) -> tuple[bool, float]:
    """
    Check if sentence is grounded using AlignScore.
    Returns (is_grounded, best_score).
    """
    scorer = load_alignscore()

    best_score = 0.0
    for src_sent in source_sentences[:top_k]:
        try:
            score = scorer.score(contexts=[src_sent], claims=[sentence])[0]
            best_score = max(best_score, score)
        except Exception as e:
            logger.debug(f"alignscore error: {e}")
            continue

    return best_score >= threshold, best_score


# --- Main experiment ---


@dataclass
class SentenceResult:
    sentence_idx: int
    sentence: str
    layer: str  # "L0", "L2_nli", "L2_alignscore", "L4_needed"
    nli_entailment: float = 0.0
    alignscore: float = 0.0


def main():
    parser = argparse.ArgumentParser(description="Exp 17: AlignScore Science Gate")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--alignscore-threshold", type=float, default=0.7)
    parser.add_argument("--nli-threshold", type=float, default=0.7)
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

    # Load AlignScore (proposed L2)
    load_alignscore()

    # Track funnel for both
    adapter = RAGTruthAdapter()
    total_cases = 0
    gate1_fail = 0
    total_sentences = 0
    l0_deterministic = 0

    # L2 comparison
    l2_nli_only = 0  # caught by DeBERTa only
    l2_align_only = 0  # caught by AlignScore only
    l2_both = 0  # caught by both
    l2_neither = 0  # caught by neither → L4

    all_sentence_results: list[SentenceResult] = []
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

        # L1: Gate 1
        ratio, min_sim = _compute_grounding_ratio(
            response, context, similarity_threshold=0.60
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

            # L0: Deterministic
            if deterministic_match(sent, ctx_sents, source_doc):
                l0_deterministic += 1
                all_sentence_results.append(SentenceResult(i, sent[:100], "L0"))
                continue

            # L2a: DeBERTa NLI (current)
            sims = [
                (j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            best_entailment = 0.0
            for src_idx, _ in sims[:3]:
                nli = nli_classify(
                    nli_tokenizer, nli_model, ctx_sents[src_idx], sent, nli_labels
                )
                best_entailment = max(best_entailment, nli.get("ENTAILMENT", 0))
            nli_grounded = best_entailment > args.nli_threshold

            # L2b: AlignScore (proposed)
            # Use same top-3 source sentences for fair comparison
            top3_sources = [ctx_sents[idx] for idx, _ in sims[:3]]
            align_grounded, align_score = alignscore_check(
                sent,
                top3_sources,
                threshold=args.alignscore_threshold,
            )

            # Compare
            if nli_grounded and align_grounded:
                l2_both += 1
                all_sentence_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_both",
                        nli_entailment=best_entailment,
                        alignscore=align_score,
                    )
                )
            elif nli_grounded and not align_grounded:
                l2_nli_only += 1
                all_sentence_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_nli_only",
                        nli_entailment=best_entailment,
                        alignscore=align_score,
                    )
                )
            elif align_grounded and not nli_grounded:
                l2_align_only += 1
                all_sentence_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L2_align_only",
                        nli_entailment=best_entailment,
                        alignscore=align_score,
                    )
                )
            else:
                l2_neither += 1
                all_sentence_results.append(
                    SentenceResult(
                        i,
                        sent[:100],
                        "L4_needed",
                        nli_entailment=best_entailment,
                        alignscore=align_score,
                    )
                )

        if total_cases % 10 == 0:
            print(f"  Processed {total_cases} cases...")

    elapsed = time.time() - start_time
    remaining = total_sentences - l0_deterministic
    nli_total = l2_both + l2_nli_only
    align_total = l2_both + l2_align_only

    print(f"\n{'='*70}")
    print("EXPERIMENT 17: ALIGNSCORE vs DeBERTa NLI")
    print(f"{'='*70}")
    print(f"Cases: {total_cases}, Gate 1 FAIL: {gate1_fail}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("\nFUNNEL:")
    print(f"  Total sentences:     {total_sentences}")
    print(
        f"  L0 Deterministic:    {l0_deterministic:>4} ({l0_deterministic/max(1,total_sentences)*100:.0f}%)"
    )
    print(f"  Remaining for L2:    {remaining:>4}")

    print(f"\nL2 COMPARISON (of {remaining} remaining sentences):")
    print(
        f"  DeBERTa NLI catches: {nli_total:>4} ({nli_total/max(1,remaining)*100:.0f}%)"
    )
    print(
        f"  AlignScore catches:  {align_total:>4} ({align_total/max(1,remaining)*100:.0f}%)"
    )
    print(f"  Improvement:         {align_total - nli_total:>+4} sentences")

    print("\n  Breakdown:")
    print(f"    Both catch:        {l2_both:>4}")
    print(f"    DeBERTa only:      {l2_nli_only:>4}")
    print(f"    AlignScore only:   {l2_align_only:>4}")
    print(f"    Neither (→ L4):    {l2_neither:>4}")

    nli_free = l0_deterministic + nli_total
    align_free = l0_deterministic + align_total
    print("\nTOTAL FREE (L0 + L2):")
    print(
        f"  With DeBERTa:        {nli_free:>4}/{total_sentences} ({nli_free/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  With AlignScore:     {align_free:>4}/{total_sentences} ({align_free/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L4 with DeBERTa:     {total_sentences - nli_free:>4} ({(total_sentences-nli_free)/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L4 with AlignScore:  {total_sentences - align_free:>4} ({(total_sentences-align_free)/max(1,total_sentences)*100:.0f}%)"
    )

    # Show AlignScore-only catches (what DeBERTa misses)
    align_only_sents = [r for r in all_sentence_results if r.layer == "L2_align_only"]
    if align_only_sents:
        print(
            f"\n--- AlignScore catches that DeBERTa misses ({len(align_only_sents)}) ---"
        )
        for r in align_only_sents[:10]:
            print(
                f"  [{r.sentence_idx}] NLI={r.nli_entailment:.3f} AlignScore={r.alignscore:.3f}"
            )
            print(f"       {r.sentence}")

    # Show DeBERTa-only catches (what AlignScore misses)
    nli_only_sents = [r for r in all_sentence_results if r.layer == "L2_nli_only"]
    if nli_only_sents:
        print(
            f"\n--- DeBERTa catches that AlignScore misses ({len(nli_only_sents)}) ---"
        )
        for r in nli_only_sents[:10]:
            print(
                f"  [{r.sentence_idx}] NLI={r.nli_entailment:.3f} AlignScore={r.alignscore:.3f}"
            )
            print(f"       {r.sentence}")
    print(f"{'='*70}")

    # Save
    save_data = {
        "experiment": "Experiment 17: AlignScore vs DeBERTa NLI",
        "total_cases": total_cases,
        "total_sentences": total_sentences,
        "l0_deterministic": l0_deterministic,
        "nli_threshold": args.nli_threshold,
        "alignscore_threshold": args.alignscore_threshold,
        "l2_nli_total": nli_total,
        "l2_align_total": align_total,
        "l2_both": l2_both,
        "l2_nli_only": l2_nli_only,
        "l2_align_only": l2_align_only,
        "l2_neither": l2_neither,
        "free_pct_nli": round(nli_free / max(1, total_sentences) * 100, 1),
        "free_pct_align": round(align_free / max(1, total_sentences) * 100, 1),
        "sentences": [
            {
                "idx": r.sentence_idx,
                "sentence": r.sentence,
                "layer": r.layer,
                "nli": r.nli_entailment,
                "alignscore": r.alignscore,
            }
            for r in all_sentence_results
        ],
    }
    save_path = output_dir / "exp17_alignscore_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
