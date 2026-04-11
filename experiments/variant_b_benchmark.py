"""
Variant B: Retrieval-Augmented MiniCheck

Instead of raw source[:3500], use MiniLM to retrieve top-5 most relevant
source sentences per claim, concatenate, feed to MiniCheck.

Same 50 cases, 283 sentences as Exp 26 baseline.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _split_sentences


@dataclass
class SentenceResult:
    case_id: str
    sentence_idx: int
    sentence: str
    gt_label: str
    gt_type: str
    response_level: str
    source_len: int
    # Baseline (from Exp 26 — raw source[:3500])
    mc_baseline: float
    # Variant B — retrieved context
    mc_retrieved: float
    retrieved_context_len: int
    top5_sims: str  # cosine sims of top-5 retrieved sentences


def label_sentences(response, span_annotations, response_level):
    sentences = _split_sentences(response)
    labels = []
    for i, sent in enumerate(sentences):
        sent_start = response.find(sent[:50])
        sent_end = sent_start + len(sent) if sent_start >= 0 else -1
        is_hall = False
        hall_type = ""
        if response_level == "fail" and span_annotations:
            for span in span_annotations:
                if sent_start >= 0 and span.start < sent_end and span.end > sent_start:
                    is_hall = True
                    hall_type = span.label_type
                    break
        labels.append(
            {
                "idx": i,
                "sentence": sent,
                "label": "hallucinated" if is_hall else "clean",
                "type": hall_type,
            }
        )
    return labels


def run_minicheck(sentence: str, context: str) -> float:
    """Run MiniCheck with given context. Returns 1.0 or 0.0."""
    try:
        import torch

        from llm_judge.calibration.hallucination import (
            _load_minicheck,
        )

        _load_minicheck()

        import llm_judge.calibration.hallucination as hal

        prompt = hal._MINICHECK_PROMPT.format(
            document=context[:3500],
            claim=sentence,
        )
        inputs = hal._mc_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        with torch.no_grad():
            outputs = hal._mc_model.generate(**inputs, max_new_tokens=5)
        generated = hal._mc_tokenizer.decode(
            outputs[0], skip_special_tokens=True
        ).strip()
        return 1.0 if generated == "1" else 0.0
    except Exception as e:
        print(f"    MC error: {str(e)[:80]}")
        return -1.0


def retrieve_top_k(
    claim: str, source_sentences: list[str], k: int = 5
) -> tuple[list[str], list[float]]:
    """Retrieve top-k source sentences by cosine similarity using MiniLM."""
    from llm_judge.properties import get_embedding_provider

    provider = get_embedding_provider()

    if not source_sentences:
        return [], []

    claim_emb = provider.encode([claim])[0]
    src_embs = provider.encode(source_sentences)

    sims = []
    for j, se in enumerate(src_embs):
        sim = provider.max_similarity(claim_emb, [se])
        sims.append((j, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    top_k = sims[:k]

    retrieved = [source_sentences[j] for j, _ in top_k]
    scores = [round(s, 4) for _, s in top_k]
    return retrieved, scores


MC_THRESHOLD = 0.5


class LiveStats:
    def __init__(self):
        self.cases = 0
        self.sents = 0
        # Baseline
        self.b_tp = 0
        self.b_fp = 0
        self.b_fn = 0
        self.b_tn = 0
        # Variant B
        self.v_tp = 0
        self.v_fp = 0
        self.v_fn = 0
        self.v_tn = 0

    def add(self, gt, baseline_score, variant_score):
        # Baseline
        if baseline_score >= 0:
            bg = baseline_score >= MC_THRESHOLD
            if gt == "hallucinated" and not bg:
                self.b_tp += 1
            elif gt == "clean" and not bg:
                self.b_fp += 1
            elif gt == "hallucinated" and bg:
                self.b_fn += 1
            else:
                self.b_tn += 1
        # Variant B
        if variant_score >= 0:
            vg = variant_score >= MC_THRESHOLD
            if gt == "hallucinated" and not vg:
                self.v_tp += 1
            elif gt == "clean" and not vg:
                self.v_fp += 1
            elif gt == "hallucinated" and vg:
                self.v_fn += 1
            else:
                self.v_tn += 1

    def _metrics(self, tp, fp, fn, tn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(0.001, p + r)
        return p, r, f1

    def show(self, elapsed, total):
        bp, br, bf = self._metrics(self.b_tp, self.b_fp, self.b_fn, self.b_tn)
        vp, vr, vf = self._metrics(self.v_tp, self.v_fp, self.v_fn, self.v_tn)
        fp_delta = self.v_fp - self.b_fp
        fn_delta = self.v_fn - self.b_fn

        print(
            f"\n--- {self.cases}/{total} cases | {self.sents} sents | {elapsed:.0f}s ---"
        )
        print(
            f"  A. Baseline     : TP={self.b_tp:3d} FP={self.b_fp:3d} FN={self.b_fn:3d} TN={self.b_tn:3d} | P={bp:.3f} R={br:.3f} F1={bf:.3f}"
        )
        print(
            f"  B. Retrieved ctx: TP={self.v_tp:3d} FP={self.v_fp:3d} FN={self.v_fn:3d} TN={self.v_tn:3d} | P={vp:.3f} R={vr:.3f} F1={vf:.3f}"
        )
        print(f"     Delta:         FP {fp_delta:+d}   FN {fn_delta:+d}")


def resp_metrics(results, field):
    cids = sorted(set(r.case_id for r in results))
    tp = fp = fn = tn = 0
    for cid in cids:
        cr = [r for r in results if r.case_id == cid]
        gt_fail = cr[0].response_level == "fail"
        flagged = any(
            getattr(r, field) >= 0 and getattr(r, field) < MC_THRESHOLD for r in cr
        )
        if gt_fail and flagged:
            tp += 1
        elif not gt_fail and flagged:
            fp += 1
        elif gt_fail and not flagged:
            fn += 1
        else:
            tn += 1
    p = tp / max(1, tp + fp)
    r_ = tp / max(1, tp + fn)
    f1 = 2 * p * r_ / max(0.001, p + r_)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "p": round(p, 3),
        "r": round(r_, 3),
        "f1": round(f1, 3),
    }


def main():
    print("=" * 70)
    print("VARIANT B: Retrieval-Augmented MiniCheck")
    print("=" * 70)
    print("  Baseline A: raw claim + source[:3500]")
    print("  Variant  B: raw claim + top-5 MiniLM-retrieved source sentences")
    print("  Same 50 cases, 283 sentences. MiniCheck threshold=0.5")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    print(f"\nCases: {len(cases)} ({n_hall} hallucinated, {len(cases)-n_hall} clean)")

    all_results: list[SentenceResult] = []
    stats = LiveStats()
    t0 = time.time()

    for ci, case in enumerate(cases):
        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )
        response = case.request.candidate_answer or ""
        gt = case.ground_truth
        labeled = label_sentences(response, gt.span_annotations, gt.response_level)
        source_sentences = _split_sentences(source)

        for sl in labeled:
            sent = sl["sentence"]

            # A. Baseline — raw source[:3500]
            mc_baseline = run_minicheck(sent, source)

            # B. Retrieved context — top-5 MiniLM sentences
            retrieved, sims = retrieve_top_k(sent, source_sentences, k=5)
            retrieved_context = " ".join(retrieved)
            mc_retrieved = run_minicheck(sent, retrieved_context)

            stats.add(sl["label"], mc_baseline, mc_retrieved)

            all_results.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    response_level=gt.response_level,
                    source_len=len(source),
                    mc_baseline=mc_baseline,
                    mc_retrieved=mc_retrieved,
                    retrieved_context_len=len(retrieved_context),
                    top5_sims=",".join(str(s) for s in sims),
                )
            )

        stats.cases = ci + 1
        stats.sents = len(all_results)
        if (ci + 1) % 5 == 0:
            stats.show(time.time() - t0, len(cases))

    # === FINAL ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"FINAL — {len(all_results)} sentences, {len(cases)} cases, {elapsed:.0f}s")
    print(f"{'='*70}")
    stats.show(elapsed, len(cases))

    # Response-level
    mr_base = resp_metrics(all_results, "mc_baseline")
    mr_retr = resp_metrics(all_results, "mc_retrieved")

    print(f"\n{'='*70}")
    print("RESPONSE-LEVEL COMPARISON")
    print(f"{'='*70}")
    print(
        f"  A. Baseline:  TP={mr_base['tp']} FP={mr_base['fp']} FN={mr_base['fn']} TN={mr_base['tn']} | P={mr_base['p']:.3f} R={mr_base['r']:.3f} F1={mr_base['f1']:.3f}"
    )
    print(
        f"  B. Retrieved: TP={mr_retr['tp']} FP={mr_retr['fp']} FN={mr_retr['fn']} TN={mr_retr['tn']} | P={mr_retr['p']:.3f} R={mr_retr['r']:.3f} F1={mr_retr['f1']:.3f}"
    )

    print("\n  Published baselines (response-level F1):")
    print("    GPT-4-turbo:    0.634")
    print("    FT Llama-2-13B: 0.787")
    print(f"    *** Baseline A: {mr_base['f1']:.3f} ***")
    print(f"    *** Variant B:  {mr_retr['f1']:.3f} ***")

    # Per-sentence changes
    print(f"\n{'='*70}")
    print("WHAT CHANGED — sentence-by-sentence")
    print(f"{'='*70}")

    fixed = [
        r
        for r in all_results
        if r.mc_baseline < MC_THRESHOLD and r.mc_retrieved >= MC_THRESHOLD
    ]
    broken = [
        r
        for r in all_results
        if r.mc_baseline >= MC_THRESHOLD and r.mc_retrieved < MC_THRESHOLD
    ]

    print(f"\n  FIXED by retrieval ({len(fixed)} sentences now correctly grounded):")
    for r in fixed:
        t = " [TRUNC]" if r.source_len > 3500 else ""
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} base={r.mc_baseline:.0f}->retr={r.mc_retrieved:.0f}{t} ctx={r.retrieved_context_len}ch | {r.sentence[:90]}"
        )

    print(f"\n  BROKEN by retrieval ({len(broken)} sentences now wrongly ungrounded):")
    for r in broken:
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} base={r.mc_baseline:.0f}->retr={r.mc_retrieved:.0f} sims={r.top5_sims} | {r.sentence[:90]}"
        )

    # Truncation analysis
    trunc_fixed = [r for r in fixed if r.source_len > 3500]
    non_trunc_fixed = [r for r in fixed if r.source_len <= 3500]
    print(f"\n  Fixed from truncated sources: {len(trunc_fixed)}")
    print(f"  Fixed from non-truncated sources: {len(non_trunc_fixed)}")

    # Save
    output = {
        "config": {"mc_threshold": MC_THRESHOLD, "retrieval_k": 5},
        "elapsed_s": round(elapsed, 1),
        "baseline_response": mr_base,
        "variant_b_response": mr_retr,
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/variant_b_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/variant_b_benchmark_results.json")


if __name__ == "__main__":
    main()
