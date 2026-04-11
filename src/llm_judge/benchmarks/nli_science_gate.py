"""
EPIC 7.20: Science Gate — NLI Validation on RAGTruth False Negatives.

Purpose: Verify that DeBERTa-v3-large-mnli can detect fact reassignment
hallucinations that embedding cosine similarity misses.

Pass criteria: >= 7/10 false negatives correctly identified by NLI contradiction.

Usage:
    poetry run python -m llm_judge.benchmarks.nli_science_gate --max-fn 10
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
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
NLI_LABELS = ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]


@dataclass
class NLISentenceResult:
    """NLI result for one response sentence."""

    sentence: str
    sentence_idx: int
    cosine_sim: float
    best_source_sentence: str
    nli_label: str  # ENTAILMENT, NEUTRAL, CONTRADICTION
    nli_probs: dict[str, float]  # probabilities for each label
    top3_nli: list[dict[str, Any]]  # NLI results against top-3 source sentences


@dataclass
class NLICaseResult:
    """NLI result for one RAGTruth case."""

    case_id: str
    ground_truth: str  # "pass" or "fail"
    gate1_ratio: float
    gate1_min_sim: float
    gate1_decision: str  # "pass" (false negative)
    nli_has_contradiction: bool  # any sentence flagged as contradiction?
    nli_decision: str  # "fail" if any contradiction, else "pass"
    correct: bool  # did NLI catch the hallucination?
    sentences: list[NLISentenceResult]


def load_nli_model():
    """Load DeBERTa NLI model."""
    print(f"Loading NLI model: {NLI_MODEL}")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    model.eval()
    elapsed = time.time() - start
    labels = [
        model.config.id2label[i].upper() for i in range(len(model.config.id2label))
    ]
    print(f"  Model loaded in {elapsed:.1f}s, labels: {labels}")
    return tokenizer, model, labels


def nli_classify(
    tokenizer: Any,
    model: Any,
    premise: str,
    hypothesis: str,
    labels: list[str] | None = None,
) -> dict[str, float]:
    """
    Classify entailment between premise and hypothesis.

    Returns: dict with probabilities for CONTRADICTION, NEUTRAL, ENTAILMENT
    """
    if labels is None:
        labels = NLI_LABELS
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0].tolist()

    return {label: round(prob, 4) for label, prob in zip(labels, probs)}


def find_false_negatives(max_cases: int = 500, max_fn: int = 10) -> list[tuple]:
    """Find RAGTruth cases where Gate 1 says PASS but ground truth is FAIL."""
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

        ratio, min_sim = _compute_grounding_ratio(
            response, context, similarity_threshold=0.60
        )

        # Gate 1 says PASS (both above threshold) — false negative
        if ratio >= 0.80 and min_sim >= 0.30:
            false_negatives.append((case, context, response, ratio, min_sim))
            if len(false_negatives) >= max_fn:
                break

    return false_negatives


def run_nli_on_case(
    case: Any,
    context: str,
    response: str,
    ratio: float,
    min_sim: float,
    tokenizer: Any,
    model: Any,
    provider: Any,
    labels: list[str] | None = None,
) -> NLICaseResult:
    """Run NLI on all sentences of one false negative case."""
    resp_sents = _split_sentences(response)
    ctx_sents = _split_sentences(context)

    if not resp_sents or not ctx_sents:
        return NLICaseResult(
            case_id=case.case_id,
            ground_truth="fail",
            gate1_ratio=ratio,
            gate1_min_sim=min_sim,
            gate1_decision="pass",
            nli_has_contradiction=False,
            nli_decision="pass",
            correct=False,
            sentences=[],
        )

    # Encode all sentences
    resp_embs = provider.encode(resp_sents)
    ctx_embs = provider.encode(ctx_sents)

    sentence_results = []
    has_contradiction = False

    for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
        # Find top-3 source sentences by cosine similarity
        sims = [
            (j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        top3 = sims[:3]

        # Run NLI against top-3
        top3_nli = []
        best_entailment = 0.0
        worst_contradiction = 0.0

        for src_idx, sim_score in top3:
            nli_result = nli_classify(
                tokenizer, model, ctx_sents[src_idx], sent, labels=labels
            )
            top3_nli.append(
                {
                    "source_idx": src_idx,
                    "source_sentence": ctx_sents[src_idx][:100],
                    "cosine_sim": round(sim_score, 4),
                    "nli": nli_result,
                    "label": max(nli_result, key=lambda k: nli_result.get(k, 0.0)),
                }
            )
            best_entailment = max(best_entailment, nli_result.get("ENTAILMENT", 0))
            worst_contradiction = max(
                worst_contradiction, nli_result.get("CONTRADICTION", 0)
            )

        # Aggregate: best-case — if ANY source entails, it's grounded
        # Use the NLI result from the most similar source sentence for reporting
        primary_nli = top3_nli[0]["nli"]
        primary_label = top3_nli[0]["label"]

        # Check if ALL top-3 show contradiction (no entailment support)
        any_entail = any(t["label"] == "ENTAILMENT" for t in top3_nli)

        # Conservative: flag if best source shows contradiction and no source entails
        if worst_contradiction > 0.5 and not any_entail:
            has_contradiction = True

        sentence_results.append(
            NLISentenceResult(
                sentence=sent,
                sentence_idx=i,
                cosine_sim=round(top3[0][1], 4),
                best_source_sentence=ctx_sents[top3[0][0]][:120],
                nli_label=primary_label,
                nli_probs=primary_nli,
                top3_nli=top3_nli,
            )
        )

    nli_decision = "fail" if has_contradiction else "pass"
    correct = (
        nli_decision == "fail"
    )  # ground truth is fail, so correct if NLI says fail

    return NLICaseResult(
        case_id=case.case_id,
        ground_truth="fail",
        gate1_ratio=ratio,
        gate1_min_sim=min_sim,
        gate1_decision="pass",
        nli_has_contradiction=has_contradiction,
        nli_decision=nli_decision,
        correct=correct,
        sentences=sentence_results,
    )


def print_report(results: list[NLICaseResult]) -> None:
    """Print detailed Science Gate report."""
    caught = sum(1 for r in results if r.correct)
    total = len(results)

    print()
    print("=" * 70)
    print("EPIC 7.20: SCIENCE GATE — NLI VALIDATION")
    print("=" * 70)
    print(f"False negatives tested: {total}")
    print(f"Caught by NLI: {caught}/{total}")
    print(f"Pass criteria: >= {int(total * 0.7)}/{total}")
    print(f"Result: {'PASS' if caught >= total * 0.7 else 'FAIL'}")
    print("=" * 70)

    for r in results:
        marker = "CAUGHT" if r.correct else "MISSED"
        print(f"\n--- {r.case_id} [{marker}] ---")
        print(
            f"  Gate 1: ratio={r.gate1_ratio:.3f} min_sim={r.gate1_min_sim:.3f} → PASS (wrong)"
        )
        print(f"  NLI decision: {r.nli_decision}")

        for s in r.sentences:
            flag = ""
            if s.nli_label == "CONTRADICTION":
                flag = " <<<< CONTRADICTION"
            elif s.nli_label == "ENTAILMENT":
                flag = " (entailed)"
            print(
                f"  [{s.sentence_idx+1}] sim={s.cosine_sim:.3f} NLI={s.nli_label} "
                f"(C={s.nli_probs.get('CONTRADICTION', 0):.3f} "
                f"N={s.nli_probs.get('NEUTRAL', 0):.3f} "
                f"E={s.nli_probs.get('ENTAILMENT', 0):.3f}){flag}"
            )
            print(f"      Response: {s.sentence[:100]}")
            print(f"      Source:   {s.best_source_sentence[:100]}")

    # Distribution check (silent failure mode detection)
    all_labels = [s.nli_label for r in results for s in r.sentences]
    label_dist: dict[str, int] = {}
    for lbl in all_labels:
        label_dist[lbl] = label_dist.get(lbl, 0) + 1
    total_sents = len(all_labels)
    print("\nNLI LABEL DISTRIBUTION (silent failure check)")
    for label, count in label_dist.items():
        pct = count / total_sents * 100 if total_sents > 0 else 0
        flag = " ⚠ >90% — possible silent failure!" if pct > 90 else ""
        print(f"  {label}: {count}/{total_sents} ({pct:.1f}%){flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EPIC 7.20: NLI Science Gate")
    parser.add_argument(
        "--max-fn",
        type=int,
        default=10,
        help="Number of false negatives to test (default: 10)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=500,
        help="Max RAGTruth cases to scan for false negatives",
    )
    parser.add_argument(
        "--output-dir", type=str, default="experiments", help="Directory for output"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Find false negatives
    print("Finding false negatives (Gate 1 PASS, ground truth FAIL)...")
    fn_cases = find_false_negatives(max_cases=args.max_cases, max_fn=args.max_fn)
    print(f"Found {len(fn_cases)} false negatives")

    if not fn_cases:
        print("No false negatives found. Cannot run Science Gate.")
        return

    # Step 2: Load NLI model
    tokenizer, model, nli_labels = load_nli_model()

    # Step 3: Run NLI on each false negative
    provider = get_embedding_provider()
    results = []
    for idx, (case, context, response, ratio, min_sim) in enumerate(fn_cases):
        print(f"\nProcessing {idx+1}/{len(fn_cases)}: {case.case_id}")
        start = time.time()
        result = run_nli_on_case(
            case,
            context,
            response,
            ratio,
            min_sim,
            tokenizer,
            model,
            provider,
            labels=nli_labels,
        )
        elapsed = time.time() - start
        print(
            f"  {result.nli_decision} ({elapsed:.1f}s) "
            f"{'CAUGHT' if result.correct else 'MISSED'}"
        )
        results.append(result)

    # Step 4: Report
    print_report(results)

    # Step 5: Save results
    save_data = {
        "experiment": "EPIC 7.20: Science Gate — NLI Validation",
        "model": NLI_MODEL,
        "false_negatives_tested": len(results),
        "caught": sum(1 for r in results if r.correct),
        "pass_criteria": f">= {int(len(results) * 0.7)}/{len(results)}",
        "result": (
            "PASS"
            if sum(1 for r in results if r.correct) >= len(results) * 0.7
            else "FAIL"
        ),
        "cases": [
            {
                "case_id": r.case_id,
                "gate1_ratio": r.gate1_ratio,
                "gate1_min_sim": r.gate1_min_sim,
                "nli_decision": r.nli_decision,
                "correct": r.correct,
                "sentences": [
                    {
                        "sentence": s.sentence[:150],
                        "cosine_sim": s.cosine_sim,
                        "nli_label": s.nli_label,
                        "nli_probs": s.nli_probs,
                    }
                    for s in r.sentences
                ],
            }
            for r in results
        ],
    }
    save_path = output_dir / "nli_science_gate_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
