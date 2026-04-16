"""
Experiment 39: MiniCheck Ceiling — Same as 38d, Gemini → MiniCheck

Identical to Exp 38d in every way except the verifier:
  - Same 11 test cases
  - Same 4 source formats (raw, hybrid, isolated, overlapping)
  - Same cached extractions from Exp 38d
  - Only change: MiniCheck Flan-T5-Large replaces Gemini 2.5 Flash

This isolates the single variable: does the verifier model change the
outcome? Gemini scored 11/11 on all 4 formats. Does MiniCheck?

Prerequisites:
  - Exp 38d extraction files in experiments/exp38d_results/
  - MiniCheck downloads on first run (~3.1GB)

Usage:
    poetry run python experiments/exp39_minicheck_ceiling.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path("experiments/exp39_results")
EXTRACTION_DIR = REPO_ROOT / "experiments/exp38d_results"

_mc_model: Any = None
_mc_tokenizer: Any = None

_MINICHECK_PROMPT = (
    "Determine whether the following claim is consistent with "
    "the corresponding document.\nDocument: {document}\nClaim: {claim}"
)


def _load_minicheck():
    global _mc_model, _mc_tokenizer
    if _mc_model is not None:
        return
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "lytang/MiniCheck-Flan-T5-Large"
    print(f"  Loading MiniCheck: {model_name} (~3.1GB)...")
    _mc_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _mc_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _mc_model.eval()
    print(f"  MiniCheck loaded.")


def minicheck_verify(sentence: str, source_doc: str) -> dict:
    import torch
    _load_minicheck()

    prompt = _MINICHECK_PROMPT.format(document=source_doc, claim=sentence)
    inputs = _mc_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = _mc_model.generate(
            **inputs, max_new_tokens=5,
            return_dict_in_generate=True, output_scores=True,
        )

    generated = _mc_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
    verdict_supported = generated == "1"

    confidence = 0.5
    prob_supported = 0.5
    if outputs.scores:
        first_token_logits = outputs.scores[0][0]
        tok_1 = _mc_tokenizer.encode("1", add_special_tokens=False)
        tok_0 = _mc_tokenizer.encode("0", add_special_tokens=False)
        if tok_1 and tok_0:
            logit_1 = first_token_logits[tok_1[0]].item()
            logit_0 = first_token_logits[tok_0[0]].item()
            max_logit = max(logit_1, logit_0)
            exp_1 = torch.exp(torch.tensor(logit_1 - max_logit)).item()
            exp_0 = torch.exp(torch.tensor(logit_0 - max_logit)).item()
            prob_supported = exp_1 / (exp_1 + exp_0)
            confidence = prob_supported if verdict_supported else (1 - prob_supported)

    verdict = "GROUNDED" if verdict_supported else "HALLUCINATED"
    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "prob_supported": round(prob_supported, 4),
        "generated_token": generated,
        "input_tokens": input_len,
        "truncated": input_len >= 2048,
    }


def compile_hybrid(extraction: dict) -> str:
    lines = []
    for hub in extraction.get("event_hubs", []):
        lines.append(f"\nEVENT: {hub.get('action', '')} ({hub.get('source_sentence', '?')})")
        for edge in hub.get("edges", []):
            lines.append(f"  {edge.get('type', '?')} -> {edge.get('target', '?')} [{edge.get('source', '?')}]")
    for ent in extraction.get("entities", []):
        attrs = ent.get("attributes", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
        lines.append(f"\nENTITY: {ent.get('name', '')} ({attr_str})")
        if ent.get("aliases"):
            lines.append(f"  Aliases: {', '.join(ent['aliases'])}")
    for eu in extraction.get("explicit_unknowns", []):
        lines.append(f"\nEXPLICITLY UNKNOWN: {eu.get('about', '')} [{eu.get('source', '?')}]")
    return "\n".join(lines)


def compile_stories(extraction: dict) -> str:
    lines = []
    for story in extraction.get("stories", []):
        lines.append(f"\n[{story.get('thread', '')}]")
        lines.append(story.get("story", ""))
        if story.get("explicit_unknowns"):
            lines.append(f"NOT STATED: {', '.join(story['explicit_unknowns'])}")
    return "\n".join(lines)


def load_extraction(src_id: str, fmt: str) -> dict | None:
    path = EXTRACTION_DIR / f"extraction_{src_id}_{fmt}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if "_parse_error" not in data:
                return data
    return None


def load_test_data() -> list[dict]:
    resp_file = REPO_ROOT / "datasets/benchmarks/ragtruth/response.jsonl"
    source_file = REPO_ROOT / "datasets/benchmarks/ragtruth/source_info.jsonl"

    sources = {}
    with open(source_file) as f:
        for line in f:
            item = json.loads(line)
            sources[item["source_id"]] = item.get("source_info", "")

    test_cases = []
    target_ids = {10, 24, 26, 28, 29, 44, 86, 141, 146, 180}

    with open(resp_file) as f:
        for line in f:
            item = json.loads(line)
            rid = int(item["id"]) if str(item.get("id", "")).isdigit() else -1
            if rid not in target_ids:
                continue

            src_id = item["source_id"]
            src_text = sources.get(src_id, "")
            labels = item.get("labels", [])
            response = item.get("response", "")

            if labels:
                for lab in labels:
                    hall_text = lab.get("text", "")
                    hall_sentence = hall_text
                    for sent in re.split(r"(?<=[.!?])\s+", response):
                        if hall_text in sent:
                            hall_sentence = sent.strip()
                            break
                    test_cases.append({
                        "id": f"ragtruth_{rid}",
                        "source_id": src_id,
                        "source_text": src_text,
                        "source_len": len(src_text),
                        "sentence": hall_sentence,
                        "ground_truth": "hallucinated",
                        "hall_type": lab.get("label_type", ""),
                    })
            else:
                first_sent = re.split(r"(?<=[.!?])\s+", response)[0].strip()
                test_cases.append({
                    "id": f"ragtruth_{rid}",
                    "source_id": src_id,
                    "source_text": src_text,
                    "source_len": len(src_text),
                    "sentence": first_sent,
                    "ground_truth": "grounded",
                    "hall_type": "clean",
                })

    return test_cases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_data()

    print(f"Loaded {len(test_cases)} test cases")
    print(f"  Hallucinated: {sum(1 for tc in test_cases if tc['ground_truth'] == 'hallucinated')}")
    print(f"  Clean: {sum(1 for tc in test_cases if tc['ground_truth'] == 'grounded')}")

    unique_sources = {}
    for tc in test_cases:
        if tc["source_id"] not in unique_sources:
            unique_sources[tc["source_id"]] = tc["source_text"]

    print(f"\n{'='*70}")
    print("PHASE 1: LOAD CACHED EXTRACTIONS FROM EXP 38d")
    print(f"{'='*70}")

    extractions = {}
    for src_id in unique_sources:
        extractions[src_id] = {}
        for fmt in ["hybrid", "isolated", "overlapping"]:
            ext = load_extraction(src_id, fmt)
            if ext:
                print(f"  Source {src_id} — {fmt}: cached ✓")
                extractions[src_id][fmt] = ext
            else:
                print(f"  Source {src_id} — {fmt}: MISSING ✗")

    print(f"\n{'='*70}")
    print("PHASE 2: LOAD MINICHECK")
    print(f"{'='*70}")
    _load_minicheck()

    print(f"\n{'='*70}")
    print(f"PHASE 3: VERIFY {len(test_cases)} CLAIMS × 4 FORMATS")
    print(f"  Verifier: MiniCheck Flan-T5-Large (local)")
    print(f"  Order: raw → hybrid → isolated → overlapping")
    print(f"{'='*70}")

    formats = ["raw", "hybrid", "isolated", "overlapping"]
    all_results = {fmt: [] for fmt in formats}

    for tc in test_cases:
        src_id = tc["source_id"]
        src_text = tc["source_text"]

        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        raw_numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

        compiled = {"raw": raw_numbered}
        if src_id in extractions:
            if "hybrid" in extractions[src_id]:
                compiled["hybrid"] = compile_hybrid(extractions[src_id]["hybrid"])
            if "isolated" in extractions[src_id]:
                compiled["isolated"] = compile_stories(extractions[src_id]["isolated"])
            if "overlapping" in extractions[src_id]:
                compiled["overlapping"] = compile_stories(extractions[src_id]["overlapping"])

        print(f"\n--- {tc['id']} ({tc['hall_type']}) ---")
        print(f"  Claim: {tc['sentence'][:80]}...")

        for fmt in formats:
            if fmt not in compiled:
                all_results[fmt].append({"_skipped": True, "_test_id": tc["id"]})
                continue

            start = time.time()
            result = minicheck_verify(tc["sentence"], compiled[fmt])
            elapsed = round(time.time() - start, 3)

            result["_elapsed"] = elapsed
            result["_test_id"] = tc["id"]
            result["_expected"] = tc["ground_truth"]
            result["_hall_type"] = tc["hall_type"]
            result["_source_chars"] = len(compiled[fmt])

            verdict = result["verdict"]
            correct = (
                (verdict == "HALLUCINATED" and tc["ground_truth"] == "hallucinated")
                or (verdict == "GROUNDED" and tc["ground_truth"] == "grounded")
            )
            result["_correct"] = correct

            status = "✓" if correct else "✗"
            trunc = " [TRUNC]" if result.get("truncated") else ""
            print(f"  {fmt:<12} {status} verdict={verdict:<13} conf={result['confidence']:.3f} "
                  f"prob_sup={result['prob_supported']:.3f} "
                  f"[{elapsed:.1f}s, {result['input_tokens']}tok{trunc}]")

            all_results[fmt].append(result)

    print(f"\n{'='*70}")
    print("PHASE 4: RESULTS")
    print(f"{'='*70}")

    print(f"\n  Gemini 2.5 Flash (Exp 38d): 11/11 on all 4 formats")
    print(f"\n  MiniCheck Flan-T5-Large (this experiment):")
    print(f"\n{'Format':<14} {'Correct':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FP':<4} {'FN':<4}")
    print("-" * 60)

    for fmt in formats:
        results = [r for r in all_results[fmt] if not r.get("_skipped")]
        if not results:
            print(f"{fmt:<14} SKIPPED")
            continue

        correct = sum(1 for r in results if r["_correct"])
        total = len(results)

        tp = sum(1 for r in results if r["_expected"] == "hallucinated" and r["verdict"] == "HALLUCINATED")
        fp = sum(1 for r in results if r["_expected"] == "grounded" and r["verdict"] == "HALLUCINATED")
        fn = sum(1 for r in results if r["_expected"] == "hallucinated" and r["verdict"] == "GROUNDED")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{fmt:<14} {correct}/{total:<8} {precision:<8.1%} {recall:<8.1%} {f1:<8.3f} {fp:<4} {fn:<4}")

    print(f"\n{'Test Case':<20} {'Type':<18} ", end="")
    for fmt in formats:
        print(f"{fmt:<16} ", end="")
    print()
    print("-" * 100)
    for i, tc in enumerate(test_cases):
        print(f"{tc['id']:<20} {tc['hall_type'][:16]:<18} ", end="")
        for fmt in formats:
            r = all_results[fmt][i]
            if r.get("_skipped"):
                print(f"{'SKIP':<16} ", end="")
                continue
            status = "✓" if r["_correct"] else "✗"
            v = r["verdict"][:5]
            print(f"{status} {v} {r['confidence']:.2f}     ", end="")
        print()

    print(f"\n{'='*70}")
    print("PHASE 5: CONFIDENCE DISTRIBUTION")
    print(f"{'='*70}")

    for fmt in formats:
        results = [r for r in all_results[fmt] if not r.get("_skipped")]
        if not results:
            continue
        hall_probs = [r["prob_supported"] for r in results if r["_expected"] == "hallucinated"]
        clean_probs = [r["prob_supported"] for r in results if r["_expected"] == "grounded"]

        print(f"\n  {fmt}:")
        if hall_probs:
            print(f"    Hallucinated: prob_sup min={min(hall_probs):.3f} max={max(hall_probs):.3f} avg={sum(hall_probs)/len(hall_probs):.3f}")
        if clean_probs:
            print(f"    Clean:        prob_sup min={min(clean_probs):.3f} max={max(clean_probs):.3f} avg={sum(clean_probs)/len(clean_probs):.3f}")
        if hall_probs and clean_probs:
            gap = min(clean_probs) - max(hall_probs)
            print(f"    Separation: {gap:+.3f} ({'clean > hall' if gap > 0 else 'OVERLAP'})")

    print(f"\n{'='*70}")
    print("PHASE 6: TRUNCATION ANALYSIS")
    print(f"{'='*70}")
    for fmt in formats:
        results = [r for r in all_results[fmt] if not r.get("_skipped")]
        if not results:
            continue
        truncated = sum(1 for r in results if r.get("truncated"))
        print(f"  {fmt:<14} {truncated}/{len(results)} truncated at 2048 tokens")

    print(f"\n{'='*70}")
    print("PHASE 7: ERROR ANALYSIS")
    print(f"{'='*70}")
    for fmt in formats:
        errors = [r for r in all_results[fmt] if not r.get("_skipped") and not r["_correct"]]
        if errors:
            print(f"\n  {fmt}: {len(errors)} error(s)")
            for r in errors:
                expected = "should flag" if r["_expected"] == "hallucinated" else "should clear"
                print(f"    {r['_test_id']} [{r['_hall_type']}]: {expected} but {r['verdict']} "
                      f"(prob_sup={r['prob_supported']:.3f}, {r['_source_chars']}ch, {r['input_tokens']}tok)")
        else:
            print(f"\n  {fmt}: 0 errors ✓")

    print(f"\n{'='*70}")
    print("PHASE 8: GEMINI vs MINICHECK")
    print(f"{'='*70}")
    print(f"\n  {'Format':<14} {'Gemini (38d)':<14} {'MiniCheck (39)':<14} {'Delta':<10}")
    print(f"  {'-'*52}")
    for fmt in formats:
        results = [r for r in all_results[fmt] if not r.get("_skipped")]
        if not results:
            print(f"  {fmt:<14} {'11/11':<14} {'SKIP':<14}")
            continue
        mc = sum(1 for r in results if r["_correct"])
        delta = mc - 11
        print(f"  {fmt:<14} {'11/11':<14} {mc}/{len(results):<12} {delta:+d}")

    out_path = OUTPUT_DIR / "exp39_results.json"
    with open(out_path, "w") as f:
        json.dump({"formats": formats, "per_format": all_results}, f, indent=2, default=str)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
