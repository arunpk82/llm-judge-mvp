"""
Experiment 20: DeBERTa NLI Threshold Sweep.

Sweeps the entailment threshold from 0.3 to 0.9 on RAGTruth benchmark
to find the optimal value for CS domain. Currently using 0.7 — may not
be optimal.

Zero cost: same model, same sentences, just different threshold.

Usage:
    python experiments/nli_improvements/exp20_threshold_sweep.py
"""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _compute_grounding_ratio, _split_sentences
from llm_judge.properties import get_embedding_provider

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"


def deterministic_match(sentence: str, source_sentences: list[str], source_full: str) -> bool:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp 20: NLI Threshold Sweep")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments/nli_improvements")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    provider = get_embedding_provider()
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    nli_labels = [nli_model.config.id2label[i].upper() for i in range(len(nli_model.config.id2label))]

    # Collect all entailment scores for sentences that pass L0
    print("Collecting entailment scores...")
    adapter = RAGTruthAdapter()
    all_scores: list[dict[str, Any]] = []
    total_cases = 0
    gate1_fail = 0
    l0_count = 0

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

        ratio, min_sim = _compute_grounding_ratio(response, context, similarity_threshold=0.60)
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
            if deterministic_match(sent, ctx_sents, source_doc):
                l0_count += 1
                continue

            # Get best entailment score
            sims = [(j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)]
            sims.sort(key=lambda x: x[1], reverse=True)
            best_entailment = 0.0
            for src_idx, _ in sims[:3]:
                inputs = nli_tokenizer(
                    ctx_sents[src_idx], sent,
                    return_tensors="pt", truncation=True, max_length=512,
                )
                with torch.no_grad():
                    probs = torch.softmax(nli_model(**inputs).logits, dim=-1)[0].tolist()
                nli_scores = {label: round(p, 4) for label, p in zip(nli_labels, probs)}
                best_entailment = max(best_entailment, nli_scores.get("ENTAILMENT", 0))

            all_scores.append({
                "case_id": case.case_id,
                "gt": gt,
                "sentence_idx": i,
                "sentence": sent[:100],
                "best_entailment": round(best_entailment, 4),
            })

        if total_cases % 10 == 0:
            print(f"  Processed {total_cases} cases...")

    total_sentences = l0_count + len(all_scores)
    print(f"\nCollected {len(all_scores)} entailment scores from {total_sentences} sentences")
    print(f"  L0 deterministic: {l0_count}")
    print(f"  Gate 1 fail: {gate1_fail}")

    # Sweep thresholds
    thresholds = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]

    # Separate grounded (gt=pass) and hallucinated (gt=fail) sentences
    grounded_scores = [s for s in all_scores if s["gt"] == "pass"]
    hallucinated_scores = [s for s in all_scores if s["gt"] == "fail"]

    print(f"\n{'='*80}")
    print("EXPERIMENT 20: NLI ENTAILMENT THRESHOLD SWEEP")
    print(f"{'='*80}")
    print(f"Sentences: {len(all_scores)} (grounded={len(grounded_scores)}, hallucinated={len(hallucinated_scores)})")
    print(f"L0 deterministic: {l0_count}")
    print()
    print(f"{'Threshold':>10} {'L2 catches':>12} {'L2 %':>8} {'Grounded→L2':>14} {'Halluc→L2':>12} {'FP risk':>10}")
    print("-" * 80)

    sweep_results = []
    for t in thresholds:
        l2_grounded = sum(1 for s in grounded_scores if s["best_entailment"] >= t)
        l2_halluc = sum(1 for s in hallucinated_scores if s["best_entailment"] >= t)
        l2_total = l2_grounded + l2_halluc
        l2_pct = l2_total / max(1, len(all_scores)) * 100
        # FP risk: hallucinated sentences wrongly confirmed as grounded
        fp_risk = l2_halluc / max(1, len(hallucinated_scores)) * 100

        marker = " <<<" if t == 0.7 else ""
        print(f"  {t:>8.2f} {l2_total:>12} {l2_pct:>7.1f}% {l2_grounded:>14} {l2_halluc:>12} {fp_risk:>9.1f}%{marker}")

        sweep_results.append({
            "threshold": t,
            "l2_total": l2_total,
            "l2_pct": round(l2_pct, 1),
            "l2_grounded": l2_grounded,
            "l2_hallucinated": l2_halluc,
            "fp_risk_pct": round(fp_risk, 1),
        })

    print()
    print("  <<< = current threshold (0.7)")
    print("  FP risk = % of hallucinated sentences wrongly confirmed as grounded at L2")
    print()

    # Find optimal threshold (max L2 catches with FP risk ≤ 5%)
    optimal = max(
        (r for r in sweep_results if r["fp_risk_pct"] <= 5.0),
        key=lambda r: r["l2_total"],
        default=None,
    )
    if optimal:
        print(f"OPTIMAL THRESHOLD: {optimal['threshold']} (L2={optimal['l2_total']}, FP risk={optimal['fp_risk_pct']}%)")
        current = next(r for r in sweep_results if r["threshold"] == 0.7)
        print(f"  vs current 0.7: L2={current['l2_total']}, improvement={optimal['l2_total'] - current['l2_total']:+d}")
    print(f"{'='*80}")

    # Save
    save_data = {
        "experiment": "Experiment 20: NLI Threshold Sweep",
        "total_sentences": total_sentences,
        "l0_deterministic": l0_count,
        "remaining_for_l2": len(all_scores),
        "grounded_sentences": len(grounded_scores),
        "hallucinated_sentences": len(hallucinated_scores),
        "sweep": sweep_results,
        "optimal": optimal,
        "scores": all_scores,
    }
    save_path = output_dir / "exp20_threshold_sweep_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
