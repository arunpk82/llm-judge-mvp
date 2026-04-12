"""
Experiment 28b: LLM-based Claim Decomposition (Gemini)

Tests whether claim decomposition ITSELF is the right idea,
using Gemini (best-in-class) to eliminate decomposition quality as a variable.

If this doesn't improve over Exp 26b baseline, decomposition is the wrong approach.
If it does improve, we find a local SLM that matches Gemini's decomposition quality.

Baseline: Exp 26b (chunking, F1=0.636)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import _chunk_source, _split_sentences

# --- Gemini decomposition ---

_DECOMPOSE_PROMPT = """Break the following sentence into independent, self-contained factual claims.

Rules:
- Each claim must be a complete sentence with its own subject and predicate.
- Resolve pronouns: replace "He", "She", "They" with the actual entity name from context.
- Each claim must be independently verifiable against a source document.
- If the sentence contains only one factual claim, return it unchanged.
- Do NOT add information. Only restructure what is already in the sentence.
- Return ONLY a JSON array of strings. No explanation, no markdown.

Sentence: "{sentence}"

Context (for pronoun resolution): "{context}"

JSON array:"""


def gemini_decompose(sentence: str, response_context: str = "") -> list[str]:
    """Use Gemini to decompose a sentence into atomic claims."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return [sentence]

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = _DECOMPOSE_PROMPT.format(
        sentence=sentence,
        context=response_context[:500],
    )

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
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()

            # Clean markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            claims = json.loads(raw)
            if isinstance(claims, list) and len(claims) > 0:
                # Filter too-short fragments
                claims = [
                    c for c in claims if isinstance(c, str) and len(c.strip()) > 15
                ]
                if claims:
                    return claims

            return [sentence]

    except Exception as e:
        print(f"    Gemini decompose error: {str(e)[:80]}")
        return [sentence]


# --- MiniCheck with chunking (same as Exp 26b) ---


def run_minicheck_chunked(claim: str, source_doc: str) -> float:
    """MiniCheck with chunking. Returns 1.0 or 0.0."""
    try:
        import torch

        import llm_judge.calibration.hallucination as hal

        hal._load_minicheck()

        chunks = _chunk_source(source_doc)
        for chunk in chunks:
            prompt = hal._MINICHECK_PROMPT.format(document=chunk, claim=claim)
            inputs = hal._mc_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            with torch.no_grad():
                outputs = hal._mc_model.generate(**inputs, max_new_tokens=5)
            generated = hal._mc_tokenizer.decode(
                outputs[0], skip_special_tokens=True
            ).strip()
            if generated == "1":
                return 1.0
        return 0.0
    except Exception as e:
        print(f"    MC error: {str(e)[:80]}")
        return -1.0


# --- Labelling ---


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


@dataclass
class SentenceResult:
    case_id: str
    sentence_idx: int
    sentence: str
    gt_label: str
    gt_type: str
    response_level: str
    source_len: int
    baseline_score: float
    decomposed: bool
    num_claims: int
    claims: str
    claim_scores: str
    exp28b_score: float


MC_THRESHOLD = 0.5


class LiveStats:
    def __init__(self):
        self.cases = 0
        self.sents = 0
        self.decomposed_count = 0
        self.total_claims = 0
        self.split_count = 0
        self.b_tp = 0
        self.b_fp = 0
        self.b_fn = 0
        self.b_tn = 0
        self.e_tp = 0
        self.e_fp = 0
        self.e_fn = 0
        self.e_tn = 0

    def add(self, gt, baseline, exp28b):
        for score, c in [(baseline, "b"), (exp28b, "e")]:
            if score < 0:
                continue
            g = score >= MC_THRESHOLD
            if gt == "hallucinated" and not g:
                setattr(self, f"{c}_tp", getattr(self, f"{c}_tp") + 1)
            elif gt == "clean" and not g:
                setattr(self, f"{c}_fp", getattr(self, f"{c}_fp") + 1)
            elif gt == "hallucinated" and g:
                setattr(self, f"{c}_fn", getattr(self, f"{c}_fn") + 1)
            else:
                setattr(self, f"{c}_tn", getattr(self, f"{c}_tn") + 1)

    def _m(self, tp, fp, fn):
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        return p, r, 2 * p * r / max(0.001, p + r)

    def show(self, elapsed, total):
        bp, br, bf = self._m(self.b_tp, self.b_fp, self.b_fn)
        ep, er, ef = self._m(self.e_tp, self.e_fp, self.e_fn)
        avg_c = self.total_claims / max(1, self.split_count)
        print(
            f"\n--- {self.cases}/{total} cases | {self.sents} sents | {elapsed:.0f}s ---"
        )
        print(
            f"  Decomposed: {self.decomposed_count}/{self.sents} ({avg_c:.1f} avg claims when split)"
        )
        print(
            f"  26b Baseline  : TP={self.b_tp:3d} FP={self.b_fp:3d} FN={self.b_fn:3d} TN={self.b_tn:3d} | P={bp:.3f} R={br:.3f} F1={bf:.3f}"
        )
        print(
            f"  28b Gemini+MC : TP={self.e_tp:3d} FP={self.e_fp:3d} FN={self.e_fn:3d} TN={self.e_tn:3d} | P={ep:.3f} R={er:.3f} F1={ef:.3f}"
        )
        print(
            f"     Delta:       FP {self.e_fp - self.b_fp:+d}   FN {self.e_fn - self.b_fn:+d}"
        )


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


def main():
    # Check Gemini API key
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set. Export it before running.")
        print("  export GEMINI_API_KEY=your_key_here")
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 28b: Gemini Claim Decomposition + MiniCheck")
    print("=" * 70)
    print("  Baseline: Exp 26b (chunking, full sentence)")
    print("  Exp 28b:  Gemini decomposes into atomic claims, MiniCheck verifies each")
    print("  Question: Is decomposition the right idea with the wrong tool (regex)?")
    print("            Or the wrong idea entirely?")
    print("  Strict aggregation: ALL claims must pass")
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

        for sl in labeled:
            sent = sl["sentence"]

            # Baseline: Exp 26b (chunking, full sentence)
            baseline_score = run_minicheck_chunked(sent, source)

            # Exp 28b: Gemini decomposition + MiniCheck per claim
            claims = gemini_decompose(sent, response)
            was_decomposed = len(claims) > 1

            if was_decomposed:
                stats.split_count += 1
                stats.total_claims += len(claims)
                claim_scores = []
                for claim in claims:
                    cs = run_minicheck_chunked(claim, source)
                    claim_scores.append(cs)

                if any(s < 0 for s in claim_scores):
                    exp28b_score = -1.0
                elif all(s >= MC_THRESHOLD for s in claim_scores):
                    exp28b_score = 1.0
                else:
                    exp28b_score = 0.0
            else:
                claim_scores = [baseline_score]
                exp28b_score = baseline_score

            stats.add(sl["label"], baseline_score, exp28b_score)
            stats.decomposed_count += 1 if was_decomposed else 0

            all_results.append(
                SentenceResult(
                    case_id=case.case_id,
                    sentence_idx=sl["idx"],
                    sentence=sent[:150],
                    gt_label=sl["label"],
                    gt_type=sl["type"],
                    response_level=gt.response_level,
                    source_len=len(source),
                    baseline_score=baseline_score,
                    decomposed=was_decomposed,
                    num_claims=len(claims),
                    claims=" | ".join(c[:80] for c in claims),
                    claim_scores=",".join(f"{s:.0f}" for s in claim_scores),
                    exp28b_score=exp28b_score,
                )
            )

        stats.cases = ci + 1
        stats.sents = len(all_results)
        if (ci + 1) % 5 == 0:
            stats.show(time.time() - t0, len(cases))

    # === FINAL ===
    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(
        f"FINAL \u2014 {len(all_results)} sentences, {len(cases)} cases, {elapsed:.0f}s"
    )
    print(f"{'='*70}")
    stats.show(elapsed, len(cases))

    mr_base = resp_metrics(all_results, "baseline_score")
    mr_exp = resp_metrics(all_results, "exp28b_score")

    print(f"\n{'='*70}")
    print("RESPONSE-LEVEL COMPARISON")
    print(f"{'='*70}")
    print(
        f"  26b Baseline:  TP={mr_base['tp']} FP={mr_base['fp']} FN={mr_base['fn']} TN={mr_base['tn']} | P={mr_base['p']:.3f} R={mr_base['r']:.3f} F1={mr_base['f1']:.3f}"
    )
    print(
        f"  28b Gemini+MC: TP={mr_exp['tp']} FP={mr_exp['fp']} FN={mr_exp['fn']} TN={mr_exp['tn']} | P={mr_exp['p']:.3f} R={mr_exp['r']:.3f} F1={mr_exp['f1']:.3f}"
    )
    print("\n  Published baselines:")
    print("    GPT-4-turbo:     0.634")
    print("    FT Llama-2-13B:  0.787")
    print(f"    *** 26b Baseline:  {mr_base['f1']:.3f} ***")
    print(f"    *** 28b Gemini+MC: {mr_exp['f1']:.3f} ***")

    # What changed
    print(f"\n{'='*70}")
    print("WHAT CHANGED \u2014 sentence-by-sentence")
    print(f"{'='*70}")

    fixed = [
        r
        for r in all_results
        if r.baseline_score < MC_THRESHOLD and r.exp28b_score >= MC_THRESHOLD
    ]
    broken = [
        r
        for r in all_results
        if r.baseline_score >= MC_THRESHOLD and r.exp28b_score < MC_THRESHOLD
    ]

    print(f"\n  FIXED by Gemini decomposition ({len(fixed)} sentences now grounded):")
    for r in fixed:
        t = " [TRUNC]" if r.source_len > 3500 else ""
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} claims={r.num_claims}{t}"
        )
        print(f"      Sentence: {r.sentence[:100]}")
        print(f"      Claims: {r.claims[:150]}")
        print(f"      Scores: {r.claim_scores}")

    print(
        f"\n  BROKEN by Gemini decomposition ({len(broken)} sentences now ungrounded):"
    )
    for r in broken:
        print(
            f"    {r.case_id} S{r.sentence_idx} GT={r.gt_label} claims={r.num_claims}"
        )
        print(f"      Sentence: {r.sentence[:100]}")
        print(f"      Claims: {r.claims[:150]}")
        print(f"      Scores: {r.claim_scores}")

    # Decomposition quality
    decomposed = [r for r in all_results if r.decomposed]
    print(f"\n{'='*70}")
    print("DECOMPOSITION STATISTICS")
    print(f"{'='*70}")
    print(f"  Total sentences: {len(all_results)}")
    print(
        f"  Decomposed by Gemini: {len(decomposed)} ({len(decomposed)/len(all_results)*100:.1f}%)"
    )
    print(
        f"  Avg claims per decomposed: {stats.total_claims / max(1, stats.split_count):.1f}"
    )
    from collections import Counter

    dist = Counter(r.num_claims for r in decomposed)
    for n, cnt in sorted(dist.items()):
        print(f"    {n} claims: {cnt} sentences")

    # Save
    output = {
        "config": {"mc_threshold": MC_THRESHOLD, "decomposer": "gemini-2.5-flash"},
        "elapsed_s": round(elapsed, 1),
        "baseline_response": mr_base,
        "exp28b_response": mr_exp,
        "decomposed_count": len(decomposed),
        "fixed_count": len(fixed),
        "broken_count": len(broken),
        "sentences": [asdict(r) for r in all_results],
    }
    with open("experiments/exp28b_gemini_decomposition_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved: experiments/exp28b_gemini_decomposition_results.json")

    # === VERDICT ===
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    if mr_exp["f1"] > mr_base["f1"]:
        print(
            f"  Gemini decomposition IMPROVED F1: {mr_base['f1']:.3f} \u2192 {mr_exp['f1']:.3f}"
        )
        print("  Decomposition is the RIGHT IDEA. Find a local SLM to replace Gemini.")
    elif mr_exp["f1"] == mr_base["f1"]:
        print(f"  Gemini decomposition had NO IMPACT: F1 = {mr_base['f1']:.3f}")
        print("  Decomposition does not help even with best-in-class decomposer.")
    else:
        print(
            f"  Gemini decomposition REGRESSED F1: {mr_base['f1']:.3f} \u2192 {mr_exp['f1']:.3f}"
        )
        print(
            "  Decomposition is the WRONG APPROACH. MiniCheck needs full sentences, not fragments."
        )
        print("  Move to model-level experiments (Exp 32/33/34).")


if __name__ == "__main__":
    main()
