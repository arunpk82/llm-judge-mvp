"""
Experiment 38c: Overlapping Small Stories

Hypothesis: The 38d stories format failed because stories were ISOLATED —
each covered one narrative thread independently. Details that span threads
(e.g., "Philadelphia woman" as an attribute of Thomas) got dropped when
the story covering that thread omitted them.

Fix: Create OVERLAPPING stories where each story shares 1-2 key context
sentences with adjacent/related stories. Like chunking with overlap —
no detail falls through the cracks because boundary facts appear in
multiple stories.

Test matrix: 11 test cases × 3 formats (raw, 38d isolated, 38c overlapping)
= 33 verification calls + 7 overlapping extractions = ~40 Gemini calls

Usage:
    poetry run python experiments/exp38c_overlapping_stories.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT = 120.0
OUTPUT_DIR = Path("experiments/exp38c_results")


def call_gemini(prompt: str, *, temperature: float = 0.0, json_mode: bool = True) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    gen_config = {"temperature": temperature, "topP": 1.0}
    if json_mode:
        gen_config["responseMimeType"] = "application/json"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }
    with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
        resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        data = resp.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]


def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {"_raw": raw, "_parse_error": True}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPT_ISOLATED_STORIES = """You are compiling a source document into small, self-contained stories.

SOURCE DOCUMENT:
{source}

INSTRUCTIONS:

Create small stories (MAXIMUM 5 sentences each) that capture ALL the source's information.
Each story should be:
- Self-contained (readable without the others)
- Factually precise (use exact names, dates, numbers from the source)
- Explicit about what is NOT known ("It is not clear when she was arrested")
- Cover one narrative thread

CRITICAL RULES:
- Do NOT add any information not in the source
- Do NOT infer dates, ages, or details — if the source doesn't say, don't include it
- If the source says something is unknown/unclear, EXPLICITLY state that
- Use exact numbers, names, dates from the source
- Every fact must be traceable to the source

Return JSON:
{{
  "stories": [
    {{
      "thread": "thread name",
      "story": "The complete story text, max 5 sentences.",
      "key_facts": ["fact1", "fact2"],
      "explicit_unknowns": ["what is not stated"]
    }}
  ]
}}
"""

PROMPT_OVERLAPPING_STORIES = """You are compiling a source document into overlapping small stories.

SOURCE DOCUMENT:
{source}

INSTRUCTIONS:

Create small stories (MAXIMUM 5 sentences each) that capture ALL the source's information.
Each story MUST OVERLAP with at least one other story — they share 1-2 sentences of
context so that entity attributes, locations, and key identifiers appear in MULTIPLE stories.

OVERLAP RULES:
- When a person is mentioned in a story, ALWAYS include their full identity
  (name, age, location, role) even if another story already established it.
- When an event references a person, include that person's key identifiers
  in the same story (e.g., "Keonna Thomas, 30, of Philadelphia" not just "Thomas").
- Each story should be understandable on its own WITH full entity context.
- Shared context sentences should appear verbatim across stories that need them.

CRITICAL RULES:
- Do NOT add any information not in the source
- Do NOT infer dates, ages, or details — if the source doesn't say, don't include it
- If the source says something is unknown/unclear, EXPLICITLY state that in EVERY
  story where the unknown information would be relevant
- Use exact numbers, names, dates from the source
- Every fact must be traceable to the source
- When the source says "a Philadelphia woman" and later identifies her as
  "Keonna Thomas, 30", EVERY story mentioning her must include BOTH
  "Philadelphia" AND "Keonna Thomas" AND "30"

Return JSON:
{{
  "stories": [
    {{
      "thread": "thread name",
      "story": "The complete story text, max 5 sentences.",
      "key_facts": ["fact1", "fact2"],
      "shared_context": ["facts repeated from other stories for completeness"],
      "explicit_unknowns": ["what is not stated"]
    }}
  ]
}}
"""

VERIFICATION_PROMPT = """You are a precise hallucination detector. Given a SOURCE REPRESENTATION and a CLAIM, determine if the claim is GROUNDED or HALLUCINATED.

SOURCE REPRESENTATION:
{compiled_source}

CLAIM TO VERIFY:
"{claim}"

RULES:
1. GROUNDED means every factual assertion in the claim is supported by the source.
2. HALLUCINATED means ANY factual assertion contradicts the source OR fabricates information not in the source.
3. If the source EXPLICITLY says something is unknown/unclear, and the claim provides a specific value for it, that is HALLUCINATION.
4. If the claim attributes an action/property to the wrong entity, that is HALLUCINATION.
5. Paraphrasing is acceptable. Changing facts is not.
6. Pay attention to: numbers, dates, names, roles, actions, and who did what.

Return JSON:
{{
  "verdict": "GROUNDED" or "HALLUCINATED",
  "confidence": 0.0 to 1.0,
  "reasoning": "step by step explanation",
  "specific_issues": [
    {{"claim_part": "the problematic text", "source_says": "what source says", "issue_type": "contradiction/fabrication/gap_filling/misattribution"}}
  ]
}}
"""


def compile_stories(extraction: dict) -> str:
    lines = []
    for story in extraction.get("stories", []):
        lines.append(f"\n[{story.get('thread', '')}]")
        lines.append(story.get("story", ""))
        if story.get("explicit_unknowns"):
            lines.append(f"NOT STATED: {', '.join(story['explicit_unknowns'])}")
    return "\n".join(lines)


def load_test_data() -> tuple[dict, list]:
    repo_root = Path(__file__).resolve().parent.parent
    resp_file = repo_root / "datasets/benchmarks/ragtruth/response.jsonl"
    source_file = repo_root / "datasets/benchmarks/ragtruth/source_info.jsonl"

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

    return sources, test_cases


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, test_cases = load_test_data()

    print(f"Loaded {len(test_cases)} test cases")
    print(f"  Hallucinated: {sum(1 for tc in test_cases if tc['ground_truth'] == 'hallucinated')}")
    print(f"  Clean: {sum(1 for tc in test_cases if tc['ground_truth'] == 'grounded')}")

    # Unique sources
    unique_sources = {}
    for tc in test_cases:
        if tc["source_id"] not in unique_sources:
            unique_sources[tc["source_id"]] = tc["source_text"]

    # ---- Phase 1: Extract overlapping stories ----
    print(f"\n{'='*70}")
    print(f"PHASE 1: EXTRACT {len(unique_sources)} SOURCES — overlapping vs isolated stories")
    print(f"{'='*70}")

    extractions = {}

    for src_id, src_text in unique_sources.items():
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))
        extractions[src_id] = {}

        # Isolated stories (38d original)
        print(f"\n  Source {src_id} ({len(src_text)} chars) — isolated stories...")
        start = time.time()
        raw = call_gemini(PROMPT_ISOLATED_STORIES.format(source=numbered))
        ext = parse_json(raw)
        ext["_elapsed"] = round(time.time() - start, 1)
        extractions[src_id]["isolated"] = ext
        n_stories = len(ext.get("stories", []))
        print(f"    {n_stories} stories in {ext['_elapsed']}s")

        # Overlapping stories (38c new)
        print(f"  Source {src_id} ({len(src_text)} chars) — overlapping stories...")
        start = time.time()
        raw = call_gemini(PROMPT_OVERLAPPING_STORIES.format(source=numbered))
        ext = parse_json(raw)
        ext["_elapsed"] = round(time.time() - start, 1)
        extractions[src_id]["overlapping"] = ext
        n_stories = len(ext.get("stories", []))
        print(f"    {n_stories} stories in {ext['_elapsed']}s")

    # Save extractions
    for src_id, formats in extractions.items():
        for fmt, ext in formats.items():
            path = OUTPUT_DIR / f"extraction_{src_id}_{fmt}.json"
            with open(path, "w") as f:
                json.dump(ext, f, indent=2)

    # ---- Phase 2: Compare compiled outputs ----
    print(f"\n{'='*70}")
    print("PHASE 2: COMPILED OUTPUT COMPARISON")
    print(f"{'='*70}")

    for src_id in unique_sources:
        iso_compiled = compile_stories(extractions[src_id]["isolated"])
        ovl_compiled = compile_stories(extractions[src_id]["overlapping"])
        print(f"\n  Source {src_id}:")
        print(f"    Isolated:    {len(iso_compiled):>5} chars, {len(extractions[src_id]['isolated'].get('stories', []))} stories")
        print(f"    Overlapping: {len(ovl_compiled):>5} chars, {len(extractions[src_id]['overlapping'].get('stories', []))} stories")
        print(f"    Ratio: {len(ovl_compiled)/max(len(iso_compiled),1):.2f}x")

    # ---- Phase 3: Verify ----
    print(f"\n{'='*70}")
    print(f"PHASE 3: VERIFY {len(test_cases)} CLAIMS × 3 FORMATS")
    print(f"{'='*70}")

    formats = ["raw", "isolated", "overlapping"]
    all_results = {fmt: [] for fmt in formats}

    for tc in test_cases:
        src_id = tc["source_id"]
        src_text = tc["source_text"]

        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        raw_numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

        compiled = {
            "raw": raw_numbered,
            "isolated": compile_stories(extractions[src_id]["isolated"]),
            "overlapping": compile_stories(extractions[src_id]["overlapping"]),
        }

        print(f"\n--- {tc['id']} ({tc['hall_type']}) ---")
        print(f"  Claim: {tc['sentence'][:80]}...")

        for fmt in formats:
            start = time.time()
            raw = call_gemini(VERIFICATION_PROMPT.format(
                compiled_source=compiled[fmt],
                claim=tc["sentence"],
            ))
            elapsed = round(time.time() - start, 1)
            result = parse_json(raw)
            result["_elapsed"] = elapsed
            result["_test_id"] = tc["id"]
            result["_expected"] = tc["ground_truth"]
            result["_hall_type"] = tc["hall_type"]

            verdict = result.get("verdict", "").upper()
            correct = (
                (verdict == "HALLUCINATED" and tc["ground_truth"] == "hallucinated")
                or (verdict == "GROUNDED" and tc["ground_truth"] == "grounded")
            )
            result["_correct"] = correct

            status = "✓" if correct else "✗"
            conf = result.get("confidence", "?")
            print(f"  {fmt:<12} {status} verdict={verdict:<13} conf={conf} [{elapsed}s]")

            all_results[fmt].append(result)

    # ---- Phase 4: Results ----
    print(f"\n{'='*70}")
    print("PHASE 4: RESULTS")
    print(f"{'='*70}")

    print(f"\n{'Format':<14} {'Correct':<10} {'Precision':<10} {'Recall':<10} {'FP':<4} {'FN':<4}")
    print("-" * 56)
    for fmt in formats:
        results = all_results[fmt]
        correct = sum(1 for r in results if r["_correct"])
        total = len(results)

        called_grounded = [r for r in results if r.get("verdict", "").upper() == "GROUNDED"]
        true_grounded = sum(1 for r in called_grounded if r["_expected"] == "grounded")
        precision = true_grounded / len(called_grounded) if called_grounded else 1.0

        actual_hall = [r for r in results if r["_expected"] == "hallucinated"]
        caught = sum(1 for r in actual_hall if r.get("verdict", "").upper() == "HALLUCINATED")
        recall = caught / len(actual_hall) if actual_hall else 0

        fp = sum(1 for r in results if r["_expected"] == "grounded" and not r["_correct"])
        fn = sum(1 for r in results if r["_expected"] == "hallucinated" and not r["_correct"])

        print(f"{fmt:<14} {correct}/{total:<8} {precision:<10.1%} {recall:<10.1%} {fp:<4} {fn:<4}")

    # Per-case breakdown
    print(f"\n{'Test Case':<20} {'Type':<18} ", end="")
    for fmt in formats:
        print(f"{fmt:<14} ", end="")
    print()
    print("-" * 86)
    for i, tc in enumerate(test_cases):
        print(f"{tc['id']:<20} {tc['hall_type'][:16]:<18} ", end="")
        for fmt in formats:
            r = all_results[fmt][i]
            status = "✓" if r["_correct"] else "✗"
            verdict = r.get("verdict", "?")[:5]
            conf = r.get("confidence", 0)
            print(f"{status} {verdict} {conf:.1f}   ", end="")
        print()

    # False positive detail
    print(f"\n{'='*70}")
    print("FALSE POSITIVE ANALYSIS")
    print(f"{'='*70}")
    for fmt in formats:
        fps = [r for r in all_results[fmt] if r["_expected"] == "grounded" and not r["_correct"]]
        if fps:
            print(f"\n  {fmt}: {len(fps)} false positive(s)")
            for r in fps:
                print(f"    {r['_test_id']}: {r.get('reasoning', '?')[:200]}")
        else:
            print(f"\n  {fmt}: 0 false positives ✓")

    # Key comparison: what does overlapping retain that isolated drops?
    print(f"\n{'='*70}")
    print("OVERLAP ANALYSIS — detail retention check")
    print(f"{'='*70}")
    # Check if "Philadelphia" appears in compiled stories for source 15596
    if "15596" in extractions:
        for fmt_name in ["isolated", "overlapping"]:
            compiled = compile_stories(extractions["15596"][fmt_name])
            has_philly = "Philadelphia" in compiled or "philadelphia" in compiled.lower()
            print(f"\n  Source 15596 ({fmt_name}):")
            print(f"    Contains 'Philadelphia': {'YES' if has_philly else 'NO ← this caused FP in 38d'}")
            print(f"    Compiled length: {len(compiled)} chars")

    # Save results
    out_path = OUTPUT_DIR / "exp38c_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "test_cases": [{k: v for k, v in tc.items() if k != "source_text"} for tc in test_cases],
            "per_format": all_results,
            "compiled_lengths": {
                src_id: {
                    fmt: len(compile_stories(ext))
                    for fmt, ext in fmts.items()
                }
                for src_id, fmts in extractions.items()
            },
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
