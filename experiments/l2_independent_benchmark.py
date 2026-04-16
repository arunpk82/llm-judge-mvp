"""
L2 Independent Benchmark — Per-layer analysis with live progress.

Runs L0, L1, L2a (MiniCheck), L2b (DeBERTa) independently on every sentence.
Reports per-layer stats as each case completes.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _l1_substring_match,
    _l3_minilm_gate_check,
    _split_sentences,
)


@dataclass
class SentenceResult:
    case_id: str
    sentence_idx: int
    sentence: str
    gt_label: str
    gt_type: str
    response_level: str
    source_len: int
    l0_match: bool
    mc_score: float
    nli_score: float


@dataclass
class CaseResult:
    case_id: str
    response_level: str
    source_len: int
    total_sents: int
    l0_matches: int
    gate1_decision: str
    grounding_ratio: float
    min_sim: float
    mc_grounded: int
    mc_ungrounded: int
    nli_grounded: int
    nli_ungrounded: int


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


def run_minicheck(sentence, source_doc):
    try:
        from llm_judge.calibration.hallucination import _l3_minicheck_score

        return _l3_minicheck_score(sentence, source_doc)
    except Exception:
        return -1.0


def run_deberta(sentence, source_doc):
    try:
        import torch

        import llm_judge.calibration.hallucination as hal

        hal._load_nli()
        from llm_judge.properties import get_embedding_provider

        provider = get_embedding_provider()
        ctx_sents = _split_sentences(source_doc)
        if not ctx_sents:
            return 0.0
        resp_emb = provider.encode([sentence])[0]
        ctx_embs = provider.encode(ctx_sents)
        sims = [
            (j, provider.max_similarity(resp_emb, [ce]))
            for j, ce in enumerate(ctx_embs)
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        best = 0.0
        for src_idx, _ in sims[:3]:
            inputs = hal._nli_tokenizer(
                ctx_sents[src_idx],
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            with torch.no_grad():
                probs = torch.softmax(hal._nli_model(**inputs).logits, dim=-1)[
                    0
                ].tolist()
            scores = {label: p for label, p in zip(hal._nli_labels or [], probs)}
            best = max(best, scores.get("ENTAILMENT", 0))
        return best
    except Exception:
        return -1.0


MC_THRESHOLD = 0.5
NLI_THRESHOLD = 0.7


class LiveStats:
    def __init__(self):
        self.cases = 0
        self.sents = 0
        self.l0_total = 0
        self.l0_on_clean = 0
        self.l0_on_hall = 0
        self.l1_pass = 0
        self.l1_fail = 0
        self.l1_ambiguous = 0
        self.mc_tp = 0
        self.mc_fp = 0
        self.mc_fn = 0
        self.mc_tn = 0
        self.nli_tp = 0
        self.nli_fp = 0
        self.nli_fn = 0
        self.nli_tn = 0

    def add_mc(self, gt, score):
        if score < 0:
            return
        grounded = score >= MC_THRESHOLD
        if gt == "hallucinated" and not grounded:
            self.mc_tp += 1
        elif gt == "clean" and not grounded:
            self.mc_fp += 1
        elif gt == "hallucinated" and grounded:
            self.mc_fn += 1
        else:
            self.mc_tn += 1

    def add_nli(self, gt, score):
        if score < 0:
            return
        grounded = score >= NLI_THRESHOLD
        if gt == "hallucinated" and not grounded:
            self.nli_tp += 1
        elif gt == "clean" and not grounded:
            self.nli_fp += 1
        elif gt == "hallucinated" and grounded:
            self.nli_fn += 1
        else:
            self.nli_tn += 1

    def _f1(self, tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return p, r, 2 * p * r / max(0.001, p + r)

    def show(self, elapsed, total_cases):
        mc_p, mc_r, mc_f1 = self._f1(self.mc_tp, self.mc_fp, self.mc_fn)
        nli_p, nli_r, nli_f1 = self._f1(self.nli_tp, self.nli_fp, self.nli_fn)
        print(
            f"\n--- {self.cases}/{total_cases} cases | {self.sents} sents | {elapsed:.0f}s ---"
        )
        print(
            f"  L0 Deterministic : {self.l0_total} matches (on clean: {self.l0_on_clean}, on hallucinated: {self.l0_on_hall})"
        )
        print(
            f"  L1 Gate 1 MiniLM : pass={self.l1_pass}  fail={self.l1_fail}  ambiguous={self.l1_ambiguous}"
        )
        print(
            f"  L2a MiniCheck    : TP={self.mc_tp:3d} FP={self.mc_fp:3d} FN={self.mc_fn:3d} TN={self.mc_tn:3d} | P={mc_p:.3f} R={mc_r:.3f} F1={mc_f1:.3f}"
        )
        print(
            f"  L2b DeBERTa      : TP={self.nli_tp:3d} FP={self.nli_fp:3d} FN={self.nli_fn:3d} TN={self.nli_tn:3d} | P={nli_p:.3f} R={nli_r:.3f} F1={nli_f1:.3f}"
        )


def main():
    print("=" * 70)
    print("L2 INDEPENDENT BENCHMARK — Per-Layer Analysis")
    print("=" * 70)
    print("  L0: Deterministic text match (per sentence)")
    print("  L1: Gate 1 MiniLM embeddings (per case)")
    print("  L2a: MiniCheck Flan-T5-Large (per sentence, independent)")
    print("  L2b: DeBERTa NLI (per sentence, independent)")
    print("  Both L2a and L2b run on ALL sentences — no chaining")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    n_hall = sum(1 for c in cases if c.ground_truth.response_level == "fail")
    n_clean = sum(1 for c in cases if c.ground_truth.response_level == "pass")
    print(f"\nCases: {len(cases)} ({n_hall} hallucinated, {n_clean} clean)")

    all_sents: list[SentenceResult] = []
    all_cases: list[CaseResult] = []
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
        ctx_sents = _split_sentences(source)

        # L1: case-level
        g1_dec, g1_ratio, g1_min = _l3_minilm_gate_check(response, source)
        if g1_dec == "pass":
            stats.l1_pass += 1
        elif g1_dec == "fail":
            stats.l1_fail += 1
        else:
            stats.l1_ambiguous += 1

        mc_g = mc_ug = nli_g = nli_ug = l0_m = 0

        for sl in labeled:
            sent = sl["sentence"]

            # L0
            l0 = _l1_substring_match(sent, ctx_sents, source)
            if l0:
                l0_m += 1
                stats.l0_total += 1
                if sl["label"] == "clean":
                    stats.l0_on_clean += 1
                else:
                    stats.l0_on_hall += 1

            # L2a MiniCheck (independent)
            mc = run_minicheck(sent, source)
            stats.add_mc(sl["label"], mc)
            if mc >= MC_THRESHOLD:
                mc_g += 1
            else:
                mc_ug += 1

            # L2b DeBERTa (independent)
            nli = run_deberta(sent, source)
            stats.add_nli(sl["label"], nli)
            if nli >= NLI_THRESHOLD:
                nli_g += 1
            else:
                nli_ug += 1

            all_sents.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    response_level=gt.response_level,
                    source_len=len(source),
                    l0_match=l0,
                    mc_score=round(mc, 4),
                    nli_score=round(nli, 4),
                )
            )

        all_cases.append(
            CaseResult(
                case_id=case.case_id,
                response_level=gt.response_level,
                source_len=len(source),
                total_sents=len(labeled),
                l0_matches=l0_m,
                gate1_decision=g1_dec,
                grounding_ratio=round(g1_ratio, 4),
                min_sim=round(g1_min, 4),
                mc_grounded=mc_g,
                mc_ungrounded=mc_ug,
                nli_grounded=nli_g,
                nli_ungrounded=nli_ug,
            )
        )

        stats.cases = ci + 1
        stats.sents = len(all_sents)
        if (ci + 1) % 5 == 0:
            stats.show(time.time() - t0, len(cases))

    # === FINAL ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(
        f"FINAL — {stats.sents} sentences across {stats.cases} cases in {elapsed:.0f}s"
    )
    print(f"{'='*70}")
    stats.show(elapsed, len(cases))

    # Response-level
    def resp_metrics(field, thr):
        cids = sorted(set(r.case_id for r in all_sents))
        tp = fp = fn = tn = 0
        for cid in cids:
            cr = [r for r in all_sents if r.case_id == cid]
            gt_fail = cr[0].response_level == "fail"
            flagged = any(
                getattr(r, field) >= 0 and getattr(r, field) < thr for r in cr
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
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(0.001, p + r)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "p": round(p, 3),
            "r": round(r, 3),
            "f1": round(f1, 3),
        }

    mc_r = resp_metrics("mc_score", MC_THRESHOLD)
    nli_r = resp_metrics("nli_score", NLI_THRESHOLD)

    print(f"\n{'='*70}")
    print("RESPONSE-LEVEL (any ungrounded sentence = fail)")
    print(f"{'='*70}")
    print(
        f"  MiniCheck: TP={mc_r['tp']} FP={mc_r['fp']} FN={mc_r['fn']} TN={mc_r['tn']} | P={mc_r['p']:.3f} R={mc_r['r']:.3f} F1={mc_r['f1']:.3f}"
    )
    print(
        f"  DeBERTa:   TP={nli_r['tp']} FP={nli_r['fp']} FN={nli_r['fn']} TN={nli_r['tn']} | P={nli_r['p']:.3f} R={nli_r['r']:.3f} F1={nli_r['f1']:.3f}"
    )
    print("\n  Published baselines (response-level F1):")
    print("    GPT-4-turbo:    0.634")
    print("    FT Llama-2-13B: 0.787")
    print(f"    *** MiniCheck:  {mc_r['f1']:.3f} ***")
    print(f"    *** DeBERTa:    {nli_r['f1']:.3f} ***")

    # Failures
    print(f"\n{'='*70}")
    print("MINICHECK FAILURES")
    print(f"{'='*70}")
    fps = [
        r for r in all_sents if r.gt_label == "clean" and 0 <= r.mc_score < MC_THRESHOLD
    ]
    fns = [
        r
        for r in all_sents
        if r.gt_label == "hallucinated" and r.mc_score >= MC_THRESHOLD
    ]
    trunc = sum(1 for r in fps if r.source_len > 3500)
    print(f"  FP: {len(fps)} (truncated sources: {trunc})")
    for r in fps[:10]:
        t = " [TRUNC]" if r.source_len > 3500 else ""
        print(
            f"    {r.case_id} S{r.sentence_idx} MC={r.mc_score:.3f}{t} | {r.sentence[:100]}"
        )
    print(f"  FN: {len(fns)}")
    for r in fns[:10]:
        print(
            f"    {r.case_id} S{r.sentence_idx} MC={r.mc_score:.3f} type={r.gt_type} | {r.sentence[:100]}"
        )

    print(f"\n{'='*70}")
    print("DEBERTA FAILURES")
    print(f"{'='*70}")
    fps2 = [
        r
        for r in all_sents
        if r.gt_label == "clean" and 0 <= r.nli_score < NLI_THRESHOLD
    ]
    fns2 = [
        r
        for r in all_sents
        if r.gt_label == "hallucinated" and r.nli_score >= NLI_THRESHOLD
    ]
    print(f"  FP: {len(fps2)}")
    for r in fps2[:10]:
        print(
            f"    {r.case_id} S{r.sentence_idx} NLI={r.nli_score:.3f} | {r.sentence[:100]}"
        )
    print(f"  FN: {len(fns2)}")
    for r in fns2[:10]:
        print(
            f"    {r.case_id} S{r.sentence_idx} NLI={r.nli_score:.3f} type={r.gt_type} | {r.sentence[:100]}"
        )

    # Disagreements
    print(f"\n{'='*70}")
    print("DISAGREEMENTS")
    print(f"{'='*70}")
    mc_only = [
        r
        for r in all_sents
        if r.mc_score >= MC_THRESHOLD and 0 <= r.nli_score < NLI_THRESHOLD
    ]
    nli_only = [
        r
        for r in all_sents
        if r.nli_score >= NLI_THRESHOLD and 0 <= r.mc_score < MC_THRESHOLD
    ]
    print(
        f"  MC=grounded, DeBERTa=not: {len(mc_only)} (hall: {sum(1 for r in mc_only if r.gt_label=='hallucinated')})"
    )
    print(
        f"  DeBERTa=grounded, MC=not: {len(nli_only)} (hall: {sum(1 for r in nli_only if r.gt_label=='hallucinated')})"
    )

    # Save
    output = {
        "config": {"mc_threshold": MC_THRESHOLD, "nli_threshold": NLI_THRESHOLD},
        "elapsed_s": round(elapsed, 1),
        "minicheck_response": mc_r,
        "deberta_response": nli_r,
        "cases": [asdict(c) for c in all_cases],
        "sentences": [asdict(r) for r in all_sents],
    }
    with open("experiments/l2_independent_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/l2_independent_benchmark_results.json")


if __name__ == "__main__":
    main()
