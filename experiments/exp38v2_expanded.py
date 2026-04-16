"""
Experiment 38v2: Expanded Source Representation Test — 10 Cases, 7 Sources.

Expands exp38 from 4 test cases (1 source) to 10 test cases (7 sources).
Tests the two winning formats (38c hybrid, 38d small stories) + raw baseline.
Covers: count errors, entity misattribution, date reassignment, action errors,
gap-filling, negation, and clean paraphrases across sources from 1222 to 6227 chars.

Test matrix: 10 test cases × 3 formats = 30 verification calls
Plus 7 sources × 2 extraction formats = 14 extraction calls
Total: ~44 Gemini calls, ~$0.02-0.03

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python experiments/exp38v2_expanded.py
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
OUTPUT_DIR = Path("experiments/exp38v2_results")

# ---------------------------------------------------------------------------
# Gemini helpers (same as exp38)
# ---------------------------------------------------------------------------
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
# Load sources and test cases from RAGTruth
# ---------------------------------------------------------------------------
def load_test_data() -> tuple[dict, list]:
    """Load sources and build test cases from RAGTruth benchmark."""
    repo_root = Path(__file__).resolve().parent.parent
    resp_file = repo_root / "datasets/benchmarks/ragtruth/response.jsonl"
    source_file = repo_root / "datasets/benchmarks/ragtruth/source_info.jsonl"

    # Load sources
    sources = {}
    with open(source_file) as f:
        for line in f:
            item = json.loads(line)
            sources[item["source_id"]] = item.get("source_info", "")

    # Build test cases
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
                # Hallucinated — extract the specific hallucinated sentence
                for lab in labels:
                    hall_text = lab.get("text", "")
                    # Find the full sentence containing this hallucination
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
                        "meta": lab.get("meta", "")[:200],
                    })
            else:
                # Clean — take first sentence of response as test
                first_sent = re.split(r"(?<=[.!?])\s+", response)[0].strip()
                test_cases.append({
                    "id": f"ragtruth_{rid}",
                    "source_id": src_id,
                    "source_text": src_text,
                    "source_len": len(src_text),
                    "sentence": first_sent,
                    "ground_truth": "grounded",
                    "hall_type": "clean",
                    "meta": "",
                })

    return sources, test_cases


# ---------------------------------------------------------------------------
# Extraction prompts (winners from exp38 + raw baseline)
# ---------------------------------------------------------------------------

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

PROMPT_STORIES = """You are compiling a source document into small, self-contained stories.

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

VERIFICATION_PROMPT = """You are a precise hallucination detector. Given a SOURCE REPRESENTATION and a CLAIM, determine if the claim is GROUNDED or HALLUCINATED.

SOURCE REPRESENTATION:
{compiled_source}

CLAIM TO VERIFY:
"{claim}"

RULES:
1. GROUNDED means every factual assertion in the claim is supported by the source.
2. HALLUCINATED means ANY factual assertion contradicts the source OR fabricates information not in the source.
3. If the source EXPLICITLY says something is unknown/unclear, and the claim provides a specific value for it, that is HALLUCINATION.
4. If the claim attributes an action/property to the wrong entity (e.g., net worth of person A attributed to person B), that is HALLUCINATION.
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


# ---------------------------------------------------------------------------
# Format compilation
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources_data, test_cases = load_test_data()

    print(f"Loaded {len(test_cases)} test cases across {len(set(tc['source_id'] for tc in test_cases))} sources")
    print(f"  Hallucinated: {sum(1 for tc in test_cases if tc['ground_truth'] == 'hallucinated')}")
    print(f"  Clean: {sum(1 for tc in test_cases if tc['ground_truth'] == 'grounded')}")

    # Show test cases
    print(f"\n{'ID':<18} {'Type':<20} {'SrcLen':>6} {'Sentence':<60}")
    print("-" * 110)
    for tc in test_cases:
        print(f"{tc['id']:<18} {tc['hall_type']:<20} {tc['source_len']:>6} {tc['sentence'][:58]}")

    # ---- Phase 1: Extract unique sources ----
    unique_sources = {}
    for tc in test_cases:
        if tc["source_id"] not in unique_sources:
            unique_sources[tc["source_id"]] = tc["source_text"]

    print(f"\n{'='*70}")
    print(f"PHASE 1: EXTRACT {len(unique_sources)} UNIQUE SOURCES × 2 FORMATS")
    print(f"{'='*70}")

    extractions = {}  # source_id -> {format -> extraction}

    for src_id, src_text in unique_sources.items():
        # Number sentences for reference
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

        extractions[src_id] = {}

        # 38c Hybrid
        print(f"\n  Source {src_id} ({len(src_text)} chars) — Hybrid extraction...")
        start = time.time()
        raw = call_gemini(PROMPT_HYBRID.format(source=numbered))
        ext = parse_json(raw)
        ext["_elapsed"] = round(time.time() - start, 1)
        extractions[src_id]["hybrid"] = ext
        print(f"    Done in {ext['_elapsed']}s")

        # 38d Stories
        print(f"  Source {src_id} ({len(src_text)} chars) — Stories extraction...")
        start = time.time()
        raw = call_gemini(PROMPT_STORIES.format(source=numbered))
        ext = parse_json(raw)
        ext["_elapsed"] = round(time.time() - start, 1)
        extractions[src_id]["stories"] = ext
        print(f"    Done in {ext['_elapsed']}s")

    # Save extractions
    for src_id, formats in extractions.items():
        for fmt, ext in formats.items():
            path = OUTPUT_DIR / f"extraction_{src_id}_{fmt}.json"
            with open(path, "w") as f:
                json.dump(ext, f, indent=2)

    # ---- Phase 2: Compile and verify ----
    print(f"\n{'='*70}")
    print(f"PHASE 2: VERIFY {len(test_cases)} CLAIMS × 3 FORMATS")
    print(f"{'='*70}")

    formats = ["raw", "hybrid", "stories"]
    all_results = {fmt: [] for fmt in formats}

    for tc in test_cases:
        src_id = tc["source_id"]
        src_text = tc["source_text"]

        # Number sentences for raw format
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        raw_numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))

        compiled = {
            "raw": raw_numbered,
            "hybrid": compile_hybrid(extractions[src_id]["hybrid"]),
            "stories": compile_stories(extractions[src_id]["stories"]),
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
            print(f"  {fmt:<8} {status} verdict={verdict:<13} conf={conf} [{elapsed}s]")

            all_results[fmt].append(result)

    # ---- Phase 3: Results ----
    print(f"\n{'='*70}")
    print(f"PHASE 3: RESULTS")
    print(f"{'='*70}")

    # Summary table
    print(f"\n{'Format':<10} {'Correct':<10} {'Precision':<10} {'Recall':<10} {'Avg Conf':<10}")
    print("-" * 50)
    for fmt in formats:
        results = all_results[fmt]
        correct = sum(1 for r in results if r["_correct"])
        total = len(results)

        # Precision: of those we called GROUNDED, how many are actually grounded?
        called_grounded = [r for r in results if r.get("verdict", "").upper() == "GROUNDED"]
        true_grounded = sum(1 for r in called_grounded if r["_expected"] == "grounded")
        precision = true_grounded / len(called_grounded) if called_grounded else 0

        # Recall: of actual hallucinations, how many did we catch?
        actual_hall = [r for r in results if r["_expected"] == "hallucinated"]
        caught = sum(1 for r in actual_hall if r.get("verdict", "").upper() == "HALLUCINATED")
        recall = caught / len(actual_hall) if actual_hall else 0

        avg_conf = sum(r.get("confidence", 0) for r in results) / total if total else 0

        print(f"{fmt:<10} {correct}/{total:<8} {precision:<10.1%} {recall:<10.1%} {avg_conf:<10.3f}")

    # Per-case breakdown
    print(f"\n{'Test Case':<20} {'Type':<18} ", end="")
    for fmt in formats:
        print(f"{fmt:<10} ", end="")
    print()
    print("-" * 80)
    for i, tc in enumerate(test_cases):
        print(f"{tc['id']:<20} {tc['hall_type'][:16]:<18} ", end="")
        for fmt in formats:
            r = all_results[fmt][i]
            status = "✓" if r["_correct"] else "✗"
            verdict = r.get("verdict", "?")[:6]
            conf = r.get("confidence", 0)
            print(f"{status}{verdict} {conf:.1f}  ", end="")
        print()

    # False positive analysis (clean sentences marked as hallucinated)
    print(f"\n{'='*70}")
    print("FALSE POSITIVE ANALYSIS (clean sentences wrongly flagged)")
    print(f"{'='*70}")
    for fmt in formats:
        fps = [r for r in all_results[fmt] if r["_expected"] == "grounded" and not r["_correct"]]
        if fps:
            print(f"\n  {fmt}: {len(fps)} false positive(s)")
            for r in fps:
                print(f"    {r['_test_id']}: {r.get('reasoning', '?')[:150]}")
        else:
            print(f"\n  {fmt}: 0 false positives ✓")

    # False negative analysis (hallucinations missed)
    print(f"\n{'='*70}")
    print("FALSE NEGATIVE ANALYSIS (hallucinations missed)")
    print(f"{'='*70}")
    for fmt in formats:
        fns = [r for r in all_results[fmt] if r["_expected"] == "hallucinated" and not r["_correct"]]
        if fns:
            print(f"\n  {fmt}: {len(fns)} false negative(s)")
            for r in fns:
                print(f"    {r['_test_id']} [{r['_hall_type']}]: {r.get('reasoning', '?')[:150]}")
        else:
            print(f"\n  {fmt}: 0 false negatives ✓")

    # Save full results
    out_path = OUTPUT_DIR / "exp38v2_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "test_cases": [{k: v for k, v in tc.items() if k != "source_text"} for tc in test_cases],
            "per_format": {
                fmt: [{k: v for k, v in r.items()} for r in results]
                for fmt, results in all_results.items()
            },
            "extraction_times": {
                src_id: {fmt: ext.get("_elapsed", 0) for fmt, ext in fmts.items()}
                for src_id, fmts in extractions.items()
            },
        }, f, indent=2)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
