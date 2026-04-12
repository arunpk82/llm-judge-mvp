"""
Layer funnel analysis: How many sentences does each detection layer
handle confidently, and how many must be escalated to LLM reasoning?

Layers (in order):
  L0: Deterministic text matching (exact/near-exact string match, n-gram overlap)
  L1: Gate 1 embeddings (MiniLM dual threshold) — case-level FAIL
  L2: NLI ENTAILMENT — sentence-level confident PASS (grounded)
  L3: GraphRAG spaCy EXACT match — sentence-level confident PASS (grounded)
  L4: Remaining — needs LLM reasoning (SLM or Gemini)

Usage:
    poetry run python -m llm_judge.benchmarks.funnel_analysis --max-cases 50
"""

from __future__ import annotations

import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from llm_judge.benchmarks.graphrag_science_gate import (
    _entity_overlap,
    extract_svo_triplets,
)
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.calibration.hallucination import (
    _compute_grounding_ratio,
    _split_sentences,
)
from llm_judge.properties import get_embedding_provider

NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
SPACY_MODEL = "en_core_web_sm"


def _normalize_text(text: str) -> str:
    """Normalize text for deterministic comparison."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _token_set(text: str) -> set[str]:
    """Get word tokens for overlap calculation."""
    return set(re.findall(r"\w+", text.lower()))


def deterministic_match(
    sentence: str, source_sentences: list[str], source_full: str
) -> tuple[bool, str]:
    """
    L0: Pure deterministic text matching. No model needed.

    Checks (in order):
      1. Exact substring: response sentence found verbatim in source
      2. Near-exact match: SequenceMatcher ratio > 0.85 against any source sentence
      3. High token overlap: Jaccard similarity > 0.80 against any source sentence

    Returns: (is_match, match_type)
    """
    norm_sent = _normalize_text(sentence)
    norm_source = _normalize_text(source_full)

    # Check 1: Exact substring match
    if norm_sent in norm_source:
        return True, "exact_substring"

    # Check 2 & 3: Compare against individual source sentences
    sent_tokens = _token_set(sentence)
    if not sent_tokens:
        return False, "no_tokens"

    for src_sent in source_sentences:
        norm_src = _normalize_text(src_sent)

        # Check 2: Near-exact (SequenceMatcher)
        ratio = SequenceMatcher(None, norm_sent, norm_src).ratio()
        if ratio > 0.85:
            return True, f"near_exact (ratio={ratio:.2f})"

        # Check 3: Token overlap (Jaccard)
        src_tokens = _token_set(src_sent)
        if src_tokens:
            intersection = sent_tokens & src_tokens
            union = sent_tokens | src_tokens
            jaccard = len(intersection) / len(union)
            if jaccard > 0.80:
                return True, f"token_overlap (jaccard={jaccard:.2f})"

    return False, "no_match"


def nli_classify(tokenizer, model, premise, hypothesis, labels):
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=512
    )
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)[0].tolist()
    return {label: round(prob, 4) for label, prob in zip(labels, probs)}


def main():
    parser = argparse.ArgumentParser(description="Layer funnel analysis")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all models
    print("Loading models...")
    provider = get_embedding_provider()

    print(f"  NLI: {NLI_MODEL}")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    nli_labels = [
        nli_model.config.id2label[i].upper()
        for i in range(len(nli_model.config.id2label))
    ]

    print(f"  spaCy: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)

    # Track funnel
    total_cases = 0
    gate1_fail_cases = 0
    gate1_pass_cases = 0
    gate1_fail_correct = 0
    gate1_fail_wrong = 0

    # Sentence level counters
    total_sentences = 0
    l0_deterministic = 0  # exact/near-exact string match
    l2_entailment = 0  # NLI confident PASS
    l2_contradiction = 0  # NLI CONTRADICTION (unreliable)
    l3_exact_match = 0  # GraphRAG confirms grounded
    l4_needs_reasoning = 0  # must go to LLM

    # L0 breakdown
    l0_exact_substring = 0
    l0_near_exact = 0
    l0_token_overlap = 0

    gt_counts = {"pass": 0, "fail": 0}

    adapter = RAGTruthAdapter()
    start_time = time.time()

    for case in adapter.load_cases(max_cases=args.max_cases):
        gt = case.ground_truth.property_labels.get("1.1")
        if gt not in ("pass", "fail"):
            continue

        total_cases += 1
        gt_counts[gt] += 1

        ctx_parts = list(case.request.source_context or [])
        conv = " ".join(msg.content for msg in case.request.conversation)
        context = conv + ("\n\n" + "\n".join(ctx_parts) if ctx_parts else "")
        source_doc = "\n".join(ctx_parts) if ctx_parts else context
        response = case.request.candidate_answer

        # L1: Gate 1 embeddings
        ratio, min_sim = _compute_grounding_ratio(
            response, context, similarity_threshold=0.60
        )
        if ratio < 0.80 or min_sim < 0.30:
            gate1_fail_cases += 1
            if gt == "fail":
                gate1_fail_correct += 1
            else:
                gate1_fail_wrong += 1
            continue

        gate1_pass_cases += 1

        resp_sents = _split_sentences(response)
        ctx_sents = _split_sentences(context)
        if not resp_sents or not ctx_sents:
            continue

        resp_embs = provider.encode(resp_sents)
        ctx_embs = provider.encode(ctx_sents)
        source_triplets = extract_svo_triplets(source_doc, nlp)

        for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
            total_sentences += 1

            # L0: Deterministic text matching
            is_match, match_type = deterministic_match(sent, ctx_sents, source_doc)
            if is_match:
                l0_deterministic += 1
                if "exact_substring" in match_type:
                    l0_exact_substring += 1
                elif "near_exact" in match_type:
                    l0_near_exact += 1
                elif "token_overlap" in match_type:
                    l0_token_overlap += 1
                continue

            # L2: NLI on top-3 source sentences
            sims = [
                (j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)
            ]
            sims.sort(key=lambda x: x[1], reverse=True)
            top3 = sims[:3]

            best_e, best_c = 0.0, 0.0
            for src_idx, _ in top3:
                nli = nli_classify(
                    nli_tokenizer, nli_model, ctx_sents[src_idx], sent, nli_labels
                )
                best_e = max(best_e, nli.get("ENTAILMENT", 0))
                best_c = max(best_c, nli.get("CONTRADICTION", 0))

            if best_e > 0.7:
                l2_entailment += 1
                continue

            if best_c > 0.7:
                l2_contradiction += 1
                continue

            # L3: GraphRAG exact match check
            resp_triplets = extract_svo_triplets(sent, nlp)
            has_exact = False
            all_matched = True

            for rt in resp_triplets:
                if rt.obj == "(intransitive)":
                    continue
                found_exact = False
                for st in source_triplets:
                    if st.obj == "(intransitive)":
                        continue
                    subj_sim = _entity_overlap(rt.subject, st.subject)
                    obj_sim = _entity_overlap(rt.obj, st.obj)
                    pred_sim = _entity_overlap(rt.predicate, st.predicate)
                    score = (subj_sim + obj_sim) / 2
                    if score >= 0.5 and pred_sim >= 0.5:
                        found_exact = True
                        break
                if found_exact:
                    has_exact = True
                else:
                    all_matched = False

            if resp_triplets and all_matched and has_exact:
                l3_exact_match += 1
                continue

            # L4: needs reasoning
            l4_needs_reasoning += 1

        if total_cases % 10 == 0:
            print(f"  Processed {total_cases} cases...")

    elapsed = time.time() - start_time

    # Derived
    confident_handled = l0_deterministic + l2_entailment + l3_exact_match
    needs_llm = l4_needs_reasoning + l2_contradiction

    # Report
    print(f"\n{'='*70}")
    print("LAYER FUNNEL ANALYSIS")
    print(f"{'='*70}")
    print(
        f"Total cases: {total_cases} (pass={gt_counts['pass']}, fail={gt_counts['fail']})"
    )
    print(f"Elapsed: {elapsed:.1f}s")

    print(f"\n{'='*70}")
    print("CASE-LEVEL FUNNEL")
    print(f"{'='*70}")
    print(
        f"  L1: Gate 1 embeddings FAIL:     {gate1_fail_cases:>4}/{total_cases} cases ({gate1_fail_cases/max(1,total_cases)*100:.0f}%)"
    )
    print(f"    Correct (gt=fail):            {gate1_fail_correct:>4}")
    print(f"    Wrong (gt=pass):              {gate1_fail_wrong:>4}")
    print(
        f"  Gate 1 PASS → sentence analysis: {gate1_pass_cases:>4}/{total_cases} cases ({gate1_pass_cases/max(1,total_cases)*100:.0f}%)"
    )

    print(f"\n{'='*70}")
    print("SENTENCE-LEVEL FUNNEL (Gate 1 PASS cases only)")
    print(f"{'='*70}")
    print(f"  Total sentences analysed:         {total_sentences:>4}")
    print()
    print(
        f"  L0: Deterministic match:          {l0_deterministic:>4} ({l0_deterministic/max(1,total_sentences)*100:.0f}%)  ← NO model, FREE"
    )
    print(f"       Exact substring:             {l0_exact_substring:>4}")
    print(f"       Near-exact (ratio>0.85):     {l0_near_exact:>4}")
    print(f"       Token overlap (jaccard>0.80): {l0_token_overlap:>4}")
    print()
    print(
        f"  L2: NLI ENTAILMENT:               {l2_entailment:>4} ({l2_entailment/max(1,total_sentences)*100:.0f}%)  ← DeBERTa, trusted"
    )
    print(
        f"  L2: NLI CONTRADICTION:             {l2_contradiction:>4} ({l2_contradiction/max(1,total_sentences)*100:.0f}%)  ← unreliable (68% FP)"
    )
    print()
    print(
        f"  L3: GraphRAG exact match:          {l3_exact_match:>4} ({l3_exact_match/max(1,total_sentences)*100:.0f}%)  ← spaCy SVO, trusted"
    )
    print()
    print(
        f"  L4: Needs LLM reasoning:           {l4_needs_reasoning:>4} ({l4_needs_reasoning/max(1,total_sentences)*100:.0f}%)  ← SLM/Gemini"
    )

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(
        f"  Confidently handled (no LLM):      {confident_handled:>4}/{total_sentences} ({confident_handled/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  Needs LLM (NEUTRAL + CONTRADICTION): {needs_llm:>4}/{total_sentences} ({needs_llm/max(1,total_sentences)*100:.0f}%)"
    )

    print(f"\n{'='*70}")
    print("COST COMPARISON")
    print(f"{'='*70}")
    print(f"  No filtering:            {total_sentences:>4} LLM calls (100%)")
    print(
        f"  L0 only:                 {total_sentences - l0_deterministic:>4} LLM calls ({(total_sentences - l0_deterministic)/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L0 + NLI:                {total_sentences - l0_deterministic - l2_entailment:>4} LLM calls ({(total_sentences - l0_deterministic - l2_entailment)/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  L0 + NLI + GraphRAG:     {needs_llm:>4} LLM calls ({needs_llm/max(1,total_sentences)*100:.0f}%)"
    )
    print(
        f"  Savings:                 {confident_handled:>4} fewer calls ({confident_handled/max(1,total_sentences)*100:.0f}% saved)"
    )

    print(f"\n{'='*70}")
    print("PRODUCTION ARCHITECTURE")
    print(f"{'='*70}")
    print(
        f"  L1 Gate 1 (MiniLM):         {gate1_fail_cases:>4} cases stopped         | Free, ~0.1s/case"
    )
    print(
        f"  L0 Deterministic:           {l0_deterministic:>4} sentences confirmed   | Free, ~0.001s/sent"
    )
    print(
        f"  L2 NLI ENTAILMENT:          {l2_entailment:>4} sentences confirmed   | Free, ~0.05s/sent"
    )
    print(
        f"  L3 GraphRAG exact:          {l3_exact_match:>4} sentences confirmed   | Free, ~0.005s/sent"
    )
    print(
        f"  L4 LLM/SLM reasoning:      {needs_llm:>4} sentences need LLM    | Per-call cost"
    )
    print(f"{'='*70}")

    save_data = {
        "total_cases": total_cases,
        "gate1_fail": gate1_fail_cases,
        "gate1_fail_correct": gate1_fail_correct,
        "gate1_fail_wrong": gate1_fail_wrong,
        "gate1_pass": gate1_pass_cases,
        "total_sentences": total_sentences,
        "l0_deterministic": l0_deterministic,
        "l0_exact_substring": l0_exact_substring,
        "l0_near_exact": l0_near_exact,
        "l0_token_overlap": l0_token_overlap,
        "l2_entailment": l2_entailment,
        "l2_contradiction": l2_contradiction,
        "l3_exact_match": l3_exact_match,
        "l4_needs_reasoning": l4_needs_reasoning,
        "confident_handled_pct": round(
            confident_handled / max(1, total_sentences) * 100, 1
        ),
        "llm_calls_needed_pct": round(needs_llm / max(1, total_sentences) * 100, 1),
    }
    save_path = output_dir / "funnel_analysis_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
