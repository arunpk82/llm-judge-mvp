"""
Benchmark Loader — single source of truth for test data.

All experiments that compute F1 scores MUST use this loader to ensure
results are comparable across experiments.

Usage:
    from experiments.benchmark_loader import load_ragtruth_50

    responses, sentences = load_ragtruth_50()
    # responses: list of 50 raw response dicts
    # sentences: list of ~306 dicts with sentence text, label, source
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_FILE = REPO_ROOT / "datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json"
RESPONSE_FILE = REPO_ROOT / "datasets/benchmarks/ragtruth/response.jsonl"
SOURCE_FILE = REPO_ROOT / "datasets/benchmarks/ragtruth/source_info.jsonl"

_spacy_nlp: Any = None


def _load_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    import spacy
    _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def load_ragtruth_50() -> tuple[list[dict], list[dict]]:
    """
    Load the fixed 50-response benchmark with spaCy sentence splitting.

    Returns:
        responses: list of 50 response dicts (raw from RAGTruth)
        sentences: list of sentence dicts, each containing:
            - sentence: str (the sentence text)
            - response_id: str
            - source_id: str
            - source_text: str
            - sentence_idx: int (position in response)
            - ground_truth: "hallucinated" or "grounded"
            - hall_type: str (label type if hallucinated, "clean" otherwise)
            - hall_text: str (the labeled span if hallucinated)
    """
    # Load benchmark definition
    with open(BENCHMARK_FILE) as f:
        benchmark = json.load(f)
    valid_ids = set(str(rid) for rid in benchmark["response_ids"])

    # Load sources
    sources = {}
    with open(SOURCE_FILE) as f:
        for line in f:
            item = json.loads(line)
            sources[item["source_id"]] = item.get("source_info", "")

    # Load responses
    responses = []
    with open(RESPONSE_FILE) as f:
        for line in f:
            item = json.loads(line)
            if str(item["id"]) in valid_ids:
                item["_source_text"] = sources.get(item["source_id"], "")
                responses.append(item)

    # Split into sentences using spaCy (L56 fix)
    nlp = _load_spacy()
    sentences = []

    for item in responses:
        response_text = item.get("response", "")
        src_id = item["source_id"]
        src_text = item["_source_text"]
        labels = item.get("labels", [])
        hall_texts = [lab.get("text", "") for lab in labels]
        hall_types = {lab.get("text", ""): lab.get("label_type", "") for lab in labels}

        # spaCy sentence splitting (NOT regex)
        doc = nlp(response_text)
        for sent_idx, sent in enumerate(doc.sents):
            sent_text = sent.text.strip()
            if not sent_text:
                continue

            # Label: hallucinated if ANY labeled span is contained in this sentence
            is_hall = False
            matched_hall = ""
            matched_type = "clean"
            for ht in hall_texts:
                if ht and ht in sent_text:
                    is_hall = True
                    matched_hall = ht
                    matched_type = hall_types.get(ht, "")
                    break

            sentences.append({
                "sentence": sent_text,
                "response_id": str(item["id"]),
                "source_id": src_id,
                "source_text": src_text,
                "sentence_idx": sent_idx,
                "ground_truth": "hallucinated" if is_hall else "grounded",
                "hall_type": matched_type,
                "hall_text": matched_hall,
            })

    return responses, sentences


if __name__ == "__main__":
    responses, sentences = load_ragtruth_50()
    n_hall = sum(1 for s in sentences if s["ground_truth"] == "hallucinated")
    n_clean = len(sentences) - n_hall

    print(f"RAGTruth-50 Benchmark")
    print(f"  Responses: {len(responses)}")
    print(f"  Sentences: {len(sentences)} (spaCy split)")
    print(f"  Hallucinated: {n_hall} ({n_hall*100/len(sentences):.1f}%)")
    print(f"  Clean: {n_clean} ({n_clean*100/len(sentences):.1f}%)")
    print(f"  Sources: {len(set(s['source_id'] for s in sentences))}")

    # Show per-response breakdown
    from collections import Counter
    resp_counts = Counter(s["response_id"] for s in sentences)
    print(f"\n  Sentences per response: min={min(resp_counts.values())}, "
          f"max={max(resp_counts.values())}, avg={sum(resp_counts.values())/len(resp_counts):.1f}")
