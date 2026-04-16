"""
Experiment 40: Gemma 4 as L3 Reasoning Verifier (v4)

v4 changes:
  - No system_instruction (may not be supported for Gemma via Gemini API)
  - JSON instruction placed FIRST in prompt (before source)
  - Narrative fallback parser extracts verdict from text
  - Detailed error logging shows exactly what went wrong

Usage:
    poetry run python experiments/exp40_gemma4_verifier.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import httpx

GEMMA_MODELS = [
    ("gemma-4-31b-it", "Gemma 4 31B Dense"),
    ("gemma-4-26b-a4b-it", "Gemma 4 26B MoE (4B active)"),
]
GEMINI_TIMEOUT = 180.0
MAX_RETRIES = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path("experiments/exp40_results")
EXTRACTION_DIR = REPO_ROOT / "experiments/exp38d_results"


def call_model(prompt: str, model: str, *, temperature: float = 0.0) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "topP": 1.0},
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code != 200:
                    print(f"    ⚠ HTTP {resp.status_code}: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()

            # Handle multiple parts (thinking mode)
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            if not parts:
                finish = data.get("candidates", [{}])[0].get("finishReason", "unknown")
                print(f"    ⚠ No parts in response. finishReason={finish}")
                return ""

            # Concatenate all text parts (skip thinking channel)
            texts = []
            for part in parts:
                text = part.get("text", "")
                if text:
                    texts.append(text)
            return "\n".join(texts)

        except (httpx.ReadTimeout, httpx.ConnectTimeout) as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"    ⚠ Attempt {attempt} timeout, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
        except httpx.HTTPStatusError as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"    ⚠ Attempt {attempt} HTTP error, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_response(raw: str) -> dict:
    """Parse JSON from response, with narrative text fallback."""
    if not raw or not raw.strip():
        return {"verdict": "", "confidence": 0, "_empty_response": True}

    # Strip thinking tags
    cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.DOTALL).strip()

    # Strategy 1: direct JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Strategy 2: code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Strategy 3: find { ... } block
    match = re.search(r"\{[^{}]*\"verdict\"[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Strategy 4: extract from narrative text
    return extract_verdict_from_text(raw)


def extract_verdict_from_text(raw: str) -> dict:
    """Extract GROUNDED/HALLUCINATED from narrative analysis."""
    upper = raw.upper()

    # Look for explicit verdict statement
    verdict_match = re.search(
        r"\b(HALLUCINATED|GROUNDED)\b", raw
    )

    hall_words = ["HALLUCINATED", "FABRICAT", "CONTRADICT", "NOT SUPPORTED",
                  "NOT GROUNDED", "DOES NOT MATCH", "INCORRECT", "INACCURATE",
                  "MISATTRIBUT", "NOT FOUND IN", "NOT STATED", "NOT MENTIONED IN THE SOURCE"]
    ground_words = ["GROUNDED", "SUPPORTED BY", "CONSISTENT WITH", "VERIFIED", "CONFIRMED",
                    "MATCHES THE SOURCE", "ACCURATELY"]

    hall_count = sum(1 for w in hall_words if w in upper)
    ground_count = sum(1 for w in ground_words if w in upper)

    if verdict_match:
        verdict = verdict_match.group(1).upper()
    elif hall_count > ground_count:
        verdict = "HALLUCINATED"
    elif ground_count > hall_count:
        verdict = "GROUNDED"
    else:
        verdict = ""

    conf_match = re.search(r"confidence[:\s]*([\d.]+)", raw, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.8

    return {
        "verdict": verdict,
        "confidence": confidence,
        "reasoning": raw[:300],
        "_parsed_from_text": True,
        "_h": hall_count,
        "_g": ground_count,
    }


VERIFICATION_PROMPT = """IMPORTANT: Respond with ONLY a JSON object. No markdown, no bullets, no analysis text.

Source document:
{compiled_source}

Claim: "{claim}"

Is the claim GROUNDED or HALLUCINATED? Check numbers, dates, names, roles, who-did-what. If any fact contradicts or is fabricated, verdict is HALLUCINATED.

Return ONLY:
{{"verdict":"GROUNDED" or "HALLUCINATED","confidence":0.0-1.0,"reasoning":"brief explanation"}}"""


def compile_hybrid(ext):
    lines = []
    for hub in ext.get("event_hubs", []):
        lines.append(f"\nEVENT: {hub.get('action', '')} ({hub.get('source_sentence', '?')})")
        for edge in hub.get("edges", []):
            lines.append(f"  {edge.get('type', '?')} -> {edge.get('target', '?')} [{edge.get('source', '?')}]")
    for ent in ext.get("entities", []):
        attrs = ent.get("attributes", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
        lines.append(f"\nENTITY: {ent.get('name', '')} ({attr_str})")
    for eu in ext.get("explicit_unknowns", []):
        lines.append(f"\nUNKNOWN: {eu.get('about', '')} [{eu.get('source', '?')}]")
    return "\n".join(lines)


def compile_stories(ext):
    lines = []
    for story in ext.get("stories", []):
        lines.append(f"\n[{story.get('thread', '')}]")
        lines.append(story.get("story", ""))
        if story.get("explicit_unknowns"):
            lines.append(f"NOT STATED: {', '.join(story['explicit_unknowns'])}")
    return "\n".join(lines)


def load_extraction(src_id, fmt):
    path = EXTRACTION_DIR / f"extraction_{src_id}_{fmt}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if "_parse_error" not in data:
                return data
    return None


def load_test_data():
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
                        "id": f"ragtruth_{rid}", "source_id": src_id,
                        "source_text": src_text, "source_len": len(src_text),
                        "sentence": hall_sentence, "ground_truth": "hallucinated",
                        "hall_type": lab.get("label_type", ""),
                    })
            else:
                first_sent = re.split(r"(?<=[.!?])\s+", response)[0].strip()
                test_cases.append({
                    "id": f"ragtruth_{rid}", "source_id": src_id,
                    "source_text": src_text, "source_len": len(src_text),
                    "sentence": first_sent, "ground_truth": "grounded",
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
    print(f"  Note: extractions by Gemini. Only verifier model changes.")
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

    formats = ["raw", "hybrid", "isolated", "overlapping"]
    all_model_results = {}

    for model_id, model_name in GEMMA_MODELS:
        print(f"\n{'='*70}")
        print(f"PHASE 2: {model_name.upper()}")
        print(f"  44 API calls · no system_instruction · narrative fallback ON")
        print(f"{'='*70}")

        all_results = {fmt: [] for fmt in formats}

        for tc in test_cases:
            src_id = tc["source_id"]
            src_text = tc["source_text"]

            sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
            raw_numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

            compiled = {"raw": raw_numbered}
            if src_id in extractions:
                for fmt_key in ["hybrid", "isolated", "overlapping"]:
                    if fmt_key in extractions[src_id]:
                        if fmt_key == "hybrid":
                            compiled[fmt_key] = compile_hybrid(extractions[src_id][fmt_key])
                        else:
                            compiled[fmt_key] = compile_stories(extractions[src_id][fmt_key])

            print(f"\n--- {tc['id']} ({tc['hall_type']}) ---")
            print(f"  Claim: {tc['sentence'][:80]}...")

            for fmt in formats:
                if fmt not in compiled:
                    all_results[fmt].append({"_skipped": True, "_test_id": tc["id"]})
                    continue

                start = time.time()
                try:
                    raw_resp = call_model(
                        VERIFICATION_PROMPT.format(
                            compiled_source=compiled[fmt], claim=tc["sentence"]
                        ),
                        model=model_id,
                    )
                    elapsed = round(time.time() - start, 1)
                    result = parse_response(raw_resp)
                except Exception as e:
                    elapsed = round(time.time() - start, 1)
                    print(f"    ✗ {fmt} EXCEPTION: {e}")
                    result = {"verdict": "ERROR", "confidence": 0, "_error": True,
                              "reasoning": str(e)[:200]}

                result["_elapsed"] = elapsed
                result["_test_id"] = tc["id"]
                result["_expected"] = tc["ground_truth"]
                result["_hall_type"] = tc["hall_type"]
                result["_source_chars"] = len(compiled[fmt])

                verdict = result.get("verdict", "").upper()
                correct = (
                    (verdict == "HALLUCINATED" and tc["ground_truth"] == "hallucinated")
                    or (verdict == "GROUNDED" and tc["ground_truth"] == "grounded")
                )
                result["_correct"] = correct

                status = "✓" if correct else "✗"
                conf = result.get("confidence", "?")
                tag = ""
                if result.get("_parsed_from_text"):
                    tag = f" [text h={result.get('_h',0)} g={result.get('_g',0)}]"
                elif result.get("_error"):
                    tag = " [ERR]"
                elif result.get("_empty_response"):
                    tag = " [EMPTY]"
                print(f"  {fmt:<12} {status} {verdict:<13} conf={conf} [{elapsed}s]{tag}")

                all_results[fmt].append(result)

        all_model_results[model_id] = all_results

        # Print summary for this model
        print(f"\n  {model_name} — Summary:")
        print(f"  {'Format':<14} {'Correct':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FP':<4} {'FN':<4}")
        print(f"  {'-'*60}")

        for fmt in formats:
            results = [r for r in all_results[fmt]
                       if not r.get("_skipped") and not r.get("_error") and not r.get("_empty_response")
                       and r.get("verdict", "").upper() in ("GROUNDED", "HALLUCINATED")]
            if not results:
                print(f"  {fmt:<14} NO VALID RESULTS")
                continue

            correct = sum(1 for r in results if r["_correct"])
            total = len(results)
            tp = sum(1 for r in results if r["_expected"] == "hallucinated" and r["verdict"].upper() == "HALLUCINATED")
            fp = sum(1 for r in results if r["_expected"] == "grounded" and r["verdict"].upper() == "HALLUCINATED")
            fn = sum(1 for r in results if r["_expected"] == "hallucinated" and r["verdict"].upper() == "GROUNDED")

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

            print(f"  {fmt:<14} {correct}/{total:<8} {prec:<8.1%} {rec:<8.1%} {f1:<8.3f} {fp:<4} {fn:<4}")

        # Error analysis
        print(f"\n  Error analysis:")
        for fmt in formats:
            errors = [r for r in all_results[fmt]
                      if not r.get("_skipped") and not r.get("_error")
                      and r.get("verdict", "").upper() in ("GROUNDED", "HALLUCINATED")
                      and not r["_correct"]]
            if errors:
                print(f"    {fmt}: {len(errors)} error(s)")
                for r in errors:
                    exp = "should flag" if r["_expected"] == "hallucinated" else "should clear"
                    print(f"      {r['_test_id']}: {exp} but {r['verdict']}")
            else:
                print(f"    {fmt}: 0 errors ✓")

    # Final comparison
    print(f"\n{'='*70}")
    print("PHASE 3: THREE-MODEL COMPARISON")
    print(f"{'='*70}")

    mc = {"raw": "7/11", "hybrid": "9/11", "isolated": "10/11", "overlapping": "7/11"}
    print(f"\n  {'Format':<14} {'Gemini':<12} {'MiniCheck':<12} ", end="")
    for _, n in GEMMA_MODELS:
        print(f"{n[:16]:<18} ", end="")
    print()
    print(f"  {'-'*76}")
    for fmt in formats:
        print(f"  {fmt:<14} {'11/11':<12} {mc[fmt]:<12} ", end="")
        for mid, _ in GEMMA_MODELS:
            res = [r for r in all_model_results.get(mid, {}).get(fmt, [])
                   if not r.get("_skipped") and not r.get("_error")
                   and r.get("verdict", "").upper() in ("GROUNDED", "HALLUCINATED")]
            c = sum(1 for r in res if r["_correct"])
            t = len(res)
            print(f"{c}/{t:<16} ", end="")
        print()

    out = OUTPUT_DIR / "exp40_results.json"
    with open(out, "w") as f:
        json.dump({
            "experiment": "Exp 40: Gemma 4 verifier",
            "note": "Extractions from 38d (Gemini). Only verifier changed.",
            "per_model": all_model_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
