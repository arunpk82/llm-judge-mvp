"""
Experiment 38d: Order independence + stochastic variance verification
(Patched: 180s timeout, 3 retries with backoff, extraction caching for resume)

Format verification order: raw → hybrid → isolated → overlapping

Usage:
    poetry run python experiments/exp38d_order_test.py
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
GEMINI_TIMEOUT = 180.0
MAX_RETRIES = 3
OUTPUT_DIR = Path("experiments/exp38d_results")


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

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"    ⚠ Attempt {attempt} failed ({type(e).__name__}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ✗ All {MAX_RETRIES} attempts failed.")
                raise


def parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        return {"_raw": raw, "_parse_error": True}


PROMPT_HYBRID = """You are building a hybrid event-centered knowledge graph from a source document.

SOURCE DOCUMENT:
{source}

INSTRUCTIONS:

Build a graph where each EVENT is a central hub, and typed edges connect it to entities, times, places, and other events.

For each event hub:
- Identify the central action
- Add typed edges: agent_of, target_of, when, where, why, how, evidence_for, leads_to, enriched_by
- Connect events to each other with narrative edges: leads_to, evidence_for, explains, parallels, contradicts

For each entity:
- Track all attributes (name, age, aliases, location)
- Track which events they participate in and their role

CRITICAL:
- For every edge, note the source sentence.
- Mark any information the source EXPLICITLY says is unknown as "EXPLICITLY_UNKNOWN".
- Do NOT infer or assume — only include what the source states.

Return JSON:
{{
  "event_hubs": [
    {{
      "event_id": "EVT_1",
      "action": "description of action",
      "source_sentence": "which sentence",
      "edges": [
        {{"type": "edge_type", "target": "entity or value", "source": "sentence ref"}}
      ]
    }}
  ],
  "entities": [
    {{
      "name": "entity name",
      "aliases": ["other names"],
      "attributes": {{"key": "value"}},
      "participates_in": ["EVT_1"],
      "source_sentences": ["S0", "S3"]
    }}
  ],
  "inter_event_edges": [
    {{"from": "EVT_1", "to": "EVT_2", "type": "relationship", "source": "sentence ref"}}
  ],
  "explicit_unknowns": [
    {{"about": "what is unknown", "unknown_slots": ["when", "where"], "source": "sentence ref"}}
  ]
}}
"""

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


def load_cached_extraction(src_id: str, fmt: str) -> dict | None:
    path = OUTPUT_DIR / f"extraction_{src_id}_{fmt}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
            if "_parse_error" not in data:
                return data
    return None


def save_extraction(src_id: str, fmt: str, ext: dict):
    path = OUTPUT_DIR / f"extraction_{src_id}_{fmt}.json"
    with open(path, "w") as f:
        json.dump(ext, f, indent=2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _, test_cases = load_test_data()

    print(f"Loaded {len(test_cases)} test cases")
    print(f"  Hallucinated: {sum(1 for tc in test_cases if tc['ground_truth'] == 'hallucinated')}")
    print(f"  Clean: {sum(1 for tc in test_cases if tc['ground_truth'] == 'grounded')}")

    unique_sources = {}
    for tc in test_cases:
        if tc["source_id"] not in unique_sources:
            unique_sources[tc["source_id"]] = tc["source_text"]

    # ---- Phase 1: Extract (with caching for resume) ----
    print(f"\n{'='*70}")
    print(f"PHASE 1: EXTRACT {len(unique_sources)} SOURCES × 3 FORMATS")
    print(f"  (timeout: {GEMINI_TIMEOUT}s, retries: {MAX_RETRIES}, caching enabled)")
    print(f"{'='*70}")

    extractions = {}
    fmt_configs = [
        ("hybrid", PROMPT_HYBRID),
        ("isolated", PROMPT_ISOLATED_STORIES),
        ("overlapping", PROMPT_OVERLAPPING_STORIES),
    ]

    for src_id, src_text in unique_sources.items():
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))
        extractions[src_id] = {}

        for fmt_name, prompt_template in fmt_configs:
            cached = load_cached_extraction(src_id, fmt_name)
            if cached:
                print(f"\n  Source {src_id} — {fmt_name}: cached ✓")
                extractions[src_id][fmt_name] = cached
                continue

            print(f"\n  Source {src_id} ({len(src_text)} chars) — {fmt_name}...")
            start = time.time()
            raw = call_gemini(prompt_template.format(source=numbered))
            ext = parse_json(raw)
            ext["_elapsed"] = round(time.time() - start, 1)
            extractions[src_id][fmt_name] = ext

            if fmt_name != "hybrid":
                n = len(ext.get("stories", []))
                print(f"    {n} stories in {ext['_elapsed']}s")
            else:
                print(f"    Done in {ext['_elapsed']}s")

            save_extraction(src_id, fmt_name, ext)

    # ---- Phase 2: Detail retention ----
    print(f"\n{'='*70}")
    print("PHASE 2: DETAIL RETENTION — 'Philadelphia' check (source 15596)")
    print(f"{'='*70}")

    if "15596" in extractions:
        for fmt_name in ["hybrid", "isolated", "overlapping"]:
            ext = extractions["15596"][fmt_name]
            compiled = compile_hybrid(ext) if fmt_name == "hybrid" else compile_stories(ext)
            has_philly = "philadelphia" in compiled.lower()
            print(f"\n  {fmt_name}:")
            print(f"    Contains 'Philadelphia': {'YES' if has_philly else 'NO ← detail loss'}")
            print(f"    Compiled length: {len(compiled)} chars")

    # ---- Phase 3: Verify ----
    print(f"\n{'='*70}")
    print(f"PHASE 3: VERIFY {len(test_cases)} CLAIMS × 4 FORMATS")
    print(f"  Order: raw → hybrid → isolated → overlapping")
    print(f"{'='*70}")

    formats = ["raw", "hybrid", "isolated", "overlapping"]
    all_results = {fmt: [] for fmt in formats}

    for tc in test_cases:
        src_id = tc["source_id"]
        src_text = tc["source_text"]

        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        raw_numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

        compiled = {
            "raw": raw_numbered,
            "hybrid": compile_hybrid(extractions[src_id]["hybrid"]),
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
    print("-" * 100)
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

    # ---- Phase 5: Cross-run comparison ----
    print(f"\n{'='*70}")
    print("PHASE 5: STOCHASTIC VARIANCE — extraction run comparison")
    print(f"{'='*70}")
    print(f"\n  Source 15596 isolated stories 'Philadelphia' retention:")
    print(f"    38v2 (run 1): NO  → caused 2 FP")
    print(f"    38c  (run 2): YES → 0 FP")
    if "15596" in extractions:
        compiled = compile_stories(extractions["15596"]["isolated"])
        has_philly = "philadelphia" in compiled.lower()
        print(f"    38d  (run 3): {'YES' if has_philly else 'NO'} → {'0 FP' if has_philly else 'FP expected'}")
        print(f"\n  Result: Philadelphia retained in {'2' if has_philly else '1'} of 3 runs.")

    # Save results
    out_path = OUTPUT_DIR / "exp38d_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "test_cases": [{k: v for k, v in tc.items() if k != "source_text"} for tc in test_cases],
            "per_format": all_results,
            "verification_order": formats,
            "philadelphia_check": {
                fmt: "philadelphia" in (
                    compile_hybrid(extractions["15596"][fmt]) if fmt == "hybrid"
                    else compile_stories(extractions["15596"][fmt])
                ).lower()
                for fmt in ["hybrid", "isolated", "overlapping"]
            } if "15596" in extractions else {},
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
