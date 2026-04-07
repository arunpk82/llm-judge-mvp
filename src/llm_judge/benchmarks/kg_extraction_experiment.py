"""
Experiment 15: SLM-extracted Knowledge Graph vs spaCy for L3 filtering.

Hypothesis: Using Gemini to extract triplets at index time produces cleaner
triplets that improve L3 exact match rate, reducing L4 Gemini calls.

Architecture:
  Index time: Gemini extracts (Subject, Verb, Object) from source docs
  Query time: L3 compares response triplets against clean source graph
  
Comparison:
  A) spaCy extraction (current L3) — 17% of sentences handled
  B) Gemini extraction (proposed)  — ?% of sentences handled

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python -m llm_judge.benchmarks.kg_extraction_experiment --max-cases 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from difflib import SequenceMatcher
from pathlib import Path

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
from llm_judge.calibration.hallucination import _compute_grounding_ratio, _split_sentences
from llm_judge.properties import get_embedding_provider

logger = logging.getLogger(__name__)
NLI_MODEL = "cross-encoder/nli-deberta-v3-large"
SPACY_MODEL = "en_core_web_sm"

EXTRACTION_PROMPT = """Extract all factual claims from this text as (Subject, Verb, Object) triplets.

Rules:
- Resolve all pronouns to actual names (e.g. "She" → "Thomas", "The company" → "Blue Bell")
- Keep compound verbs together (e.g. "shut down", "charged with", "taken into custody")
- Include dates, numbers, and locations as separate triplets when they modify a claim
- Each triplet should be a standalone fact that can be verified independently
- Output ONLY triplets, one per line, in this exact format: (Subject, Verb, Object)

TEXT:
{text}

TRIPLETS:"""


def gemini_extract_triplets(text: str) -> list[Triplet]:
    """Extract triplets using Gemini."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return []
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = EXTRACTION_PROMPT.format(text=text[:4000])
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()
            return _parse_triplet_text(raw)
    except Exception as e:
        logger.warning(f"gemini extraction error: {str(e)[:80]}")
        return []


def _parse_triplet_text(text: str) -> list[Triplet]:
    """Parse triplets from Gemini output like (Subject, Verb, Object)."""
    triplets = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading bullet/number
        line = re.sub(r"^[-\d.*]+\s*", "", line).strip()
        # Match (Subject, Verb, Object) pattern
        match = re.match(r"\((.+?),\s*(.+?),\s*(.+?)\)$", line)
        if match:
            s, v, o = match.group(1).strip(), match.group(2).strip(), match.group(3).strip()
            if s and v and o:
                triplets.append(Triplet(subject=s, predicate=v, obj=o))
    return triplets


def nli_classify(tokenizer, model, premise, hypothesis, labels):
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
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


def check_exact_match(resp_triplets: list[Triplet], src_triplets: list[Triplet]) -> bool:
    """Check if all response triplets have exact matches in source."""
    if not resp_triplets:
        return False

    has_exact = False
    all_matched = True

    for rt in resp_triplets:
        if rt.obj == "(intransitive)":
            continue
        found = False
        for st in src_triplets:
            if st.obj == "(intransitive)":
                continue
            subj_sim = _entity_overlap(rt.subject, st.subject)
            obj_sim = _entity_overlap(rt.obj, st.obj)
            pred_sim = _entity_overlap(rt.predicate, st.predicate)
            score = (subj_sim + obj_sim) / 2
            if score >= 0.5 and pred_sim >= 0.5:
                found = True
                break
        if found:
            has_exact = True
        else:
            all_matched = False

    return has_exact and all_matched


def main():
    parser = argparse.ArgumentParser(description="Experiment 15: KG Extraction Comparison")
    parser.add_argument("--max-cases", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="experiments")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load models
    print("Loading models...")
    provider = get_embedding_provider()

    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    nli_model.eval()
    nli_labels = [nli_model.config.id2label[i].upper() for i in range(len(nli_model.config.id2label))]

    nlp = spacy.load(SPACY_MODEL)

    # Track funnel for both extraction methods
    total_cases = 0
    gate1_fail = 0
    total_sentences = 0
    l0_deterministic = 0
    l2_entailment = 0
    l2_contradiction = 0

    # L3 comparison
    l3_spacy_exact = 0
    l3_gemini_exact = 0
    l3_both = 0       # caught by both
    l3_spacy_only = 0  # caught by spaCy but not Gemini
    l3_gemini_only = 0 # caught by Gemini but not spaCy
    l3_neither = 0     # caught by neither → goes to L4

    # Cache Gemini extractions per source doc
    gemini_cache: dict[str, list[Triplet]] = {}
    gemini_calls = 0

    adapter = RAGTruthAdapter()
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

        # Extract source triplets with both methods
        spacy_src_triplets = extract_svo_triplets(source_doc, nlp)

        # Gemini extraction (cached per source doc)
        src_key = source_doc[:200]
        if src_key not in gemini_cache:
            gemini_cache[src_key] = gemini_extract_triplets(source_doc)
            gemini_calls += 1
        gemini_src_triplets = gemini_cache[src_key]

        for i, (sent, emb) in enumerate(zip(resp_sents, resp_embs)):
            total_sentences += 1

            # L0: Deterministic
            if deterministic_match(sent, ctx_sents, source_doc):
                l0_deterministic += 1
                continue

            # L2: NLI
            sims = [(j, provider.max_similarity(emb, [ce])) for j, ce in enumerate(ctx_embs)]
            sims.sort(key=lambda x: x[1], reverse=True)
            best_e, best_c = 0.0, 0.0
            for src_idx, _ in sims[:3]:
                nli = nli_classify(nli_tokenizer, nli_model, ctx_sents[src_idx], sent, nli_labels)
                best_e = max(best_e, nli.get("ENTAILMENT", 0))
                best_c = max(best_c, nli.get("CONTRADICTION", 0))

            if best_e > 0.7:
                l2_entailment += 1
                continue

            if best_c > 0.7:
                l2_contradiction += 1
                continue

            # L3: Compare BOTH extraction methods
            # spaCy triplets for response sentence
            spacy_resp_triplets = extract_svo_triplets(sent, nlp)
            spacy_match = check_exact_match(spacy_resp_triplets, spacy_src_triplets)

            # For Gemini comparison, also extract response triplets with spaCy
            # (we use spaCy for response at query time — only source extraction changes)
            gemini_match = check_exact_match(spacy_resp_triplets, gemini_src_triplets)

            if spacy_match and gemini_match:
                l3_both += 1
            elif spacy_match and not gemini_match:
                l3_spacy_only += 1
            elif gemini_match and not spacy_match:
                l3_gemini_only += 1
            else:
                l3_neither += 1

            if spacy_match:
                l3_spacy_exact += 1
            if gemini_match:
                l3_gemini_exact += 1

        if total_cases % 10 == 0:
            print(f"  Processed {total_cases} cases... (Gemini extractions: {gemini_calls})")

    elapsed = time.time() - start_time

    # Report
    remaining_after_l2 = total_sentences - l0_deterministic - l2_entailment
    l4_spacy = remaining_after_l2 - l3_spacy_exact
    l4_gemini = remaining_after_l2 - l3_gemini_exact

    print(f"\n{'='*70}")
    print("EXPERIMENT 15: KNOWLEDGE GRAPH EXTRACTION COMPARISON")
    print(f"{'='*70}")
    print(f"Cases: {total_cases}, Elapsed: {elapsed:.1f}s")
    print(f"Gemini extraction calls: {gemini_calls} (one per unique source doc)")

    print("\n--- SHARED LAYERS (same for both) ---")
    print(f"  L1 Gate 1 FAIL:         {gate1_fail:>4} cases")
    print(f"  Total sentences:        {total_sentences:>4}")
    print(f"  L0 Deterministic:       {l0_deterministic:>4} ({l0_deterministic/max(1,total_sentences)*100:.0f}%)")
    print(f"  L2 NLI ENTAILMENT:      {l2_entailment:>4} ({l2_entailment/max(1,total_sentences)*100:.0f}%)")
    print(f"  L2 NLI CONTRADICTION:   {l2_contradiction:>4} ({l2_contradiction/max(1,total_sentences)*100:.0f}%)")
    print(f"  Remaining for L3:       {remaining_after_l2:>4}")

    print("\n--- L3 COMPARISON ---")
    print(f"  spaCy source triplets (avg):   {sum(len(extract_svo_triplets('', nlp)) for _ in [0]) if False else 'N/A'}")
    print(f"  spaCy L3 exact match:     {l3_spacy_exact:>4} ({l3_spacy_exact/max(1,remaining_after_l2)*100:.0f}% of remaining)")
    print(f"  Gemini L3 exact match:    {l3_gemini_exact:>4} ({l3_gemini_exact/max(1,remaining_after_l2)*100:.0f}% of remaining)")
    print(f"  Improvement:              {l3_gemini_exact - l3_spacy_exact:>+4} sentences")
    print()
    print("  Breakdown:")
    print(f"    Both matched:           {l3_both:>4}")
    print(f"    spaCy only:             {l3_spacy_only:>4}")
    print(f"    Gemini only:            {l3_gemini_only:>4}")
    print(f"    Neither (→ L4):         {l3_neither:>4}")

    print("\n--- COST IMPACT ---")
    print(f"  With spaCy L3:    {l4_spacy + l2_contradiction:>4} sentences need L4 ({(l4_spacy + l2_contradiction)/max(1,total_sentences)*100:.0f}%)")
    print(f"  With Gemini L3:   {l4_gemini + l2_contradiction:>4} sentences need L4 ({(l4_gemini + l2_contradiction)/max(1,total_sentences)*100:.0f}%)")
    print(f"  Savings:          {l4_spacy - l4_gemini:>4} fewer L4 calls")

    spacy_free = l0_deterministic + l2_entailment + l3_spacy_exact
    gemini_free = l0_deterministic + l2_entailment + l3_gemini_exact
    print("\n--- TOTAL FREE HANDLING ---")
    print(f"  With spaCy:       {spacy_free:>4}/{total_sentences} ({spacy_free/max(1,total_sentences)*100:.0f}%)")
    print(f"  With Gemini KG:   {gemini_free:>4}/{total_sentences} ({gemini_free/max(1,total_sentences)*100:.0f}%)")
    print(f"{'='*70}")

    # Show sample Gemini vs spaCy triplets
    if gemini_cache:
        first_key = list(gemini_cache.keys())[0]
        print("\n--- SAMPLE: Source triplet quality ---")
        print("  Gemini triplets (first source):")
        for t in gemini_cache[first_key][:8]:
            print(f"    {t}")
        print("  spaCy triplets (same source):")
        # Re-extract for display
        for key, case_data in [(k, v) for k, v in zip(list(gemini_cache.keys())[:1], [None])]:
            break

    # Save
    save_data = {
        "experiment": "Experiment 15: KG Extraction Comparison",
        "total_cases": total_cases,
        "total_sentences": total_sentences,
        "gemini_extraction_calls": gemini_calls,
        "l0_deterministic": l0_deterministic,
        "l2_entailment": l2_entailment,
        "l2_contradiction": l2_contradiction,
        "l3_spacy_exact": l3_spacy_exact,
        "l3_gemini_exact": l3_gemini_exact,
        "l3_both": l3_both,
        "l3_spacy_only": l3_spacy_only,
        "l3_gemini_only": l3_gemini_only,
        "l3_neither": l3_neither,
        "l4_with_spacy": l4_spacy + l2_contradiction,
        "l4_with_gemini": l4_gemini + l2_contradiction,
        "free_pct_spacy": round(spacy_free / max(1, total_sentences) * 100, 1),
        "free_pct_gemini": round(gemini_free / max(1, total_sentences) * 100, 1),
    }
    save_path = output_dir / "kg_extraction_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
