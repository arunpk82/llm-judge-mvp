"""
Experiment 33 — Optimized SLM Extraction Pipeline

Tests whether three optimization techniques can close the gap between
Gemma 4 31B (66.7% in Exp 32) and Gemini 2.5 Flash baseline.

Techniques:
  T1: spaCy NER pre-seeding — extract entities from source first, hint to model
  T2: Chained extraction — P1 entities feed into P2/P3 prompts
  T3: Self-verification — after P2, ask model to find missed events
  T4: JSON mode — force structured JSON output via API parameter

Phases:
  33a: Raw Gemma 31B (Exp 32 baseline = 66.7%)
  33b: T1 + T2 (spaCy pre-seed + chained P1→P2/P3)
  33c: T1 + T2 + T3 (add self-verification)

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python experiments/exp33_optimized_extraction.py --max-sources 2

Output: experiments/exp33_optimized_extraction_results.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ========================================================================
# TECHNIQUE 1: spaCy NER Pre-seeding
# ========================================================================

def spacy_pre_extract(source_text: str) -> dict:
    """Run spaCy NER on source to extract entities, numbers, dates."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return {"entities": [], "numbers": [], "dates": [], "sentences": []}

    doc = nlp(source_text)

    entities = []
    seen = set()
    for ent in doc.ents:
        key = (ent.text.lower().strip(), ent.label_)
        if key not in seen and ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "EVENT", "PRODUCT"):
            entities.append({"text": ent.text, "label": ent.label_})
            seen.add(key)

    numbers = []
    for ent in doc.ents:
        if ent.label_ in ("CARDINAL", "QUANTITY", "MONEY", "PERCENT", "ORDINAL"):
            numbers.append({"text": ent.text, "label": ent.label_})

    dates = []
    for ent in doc.ents:
        if ent.label_ in ("DATE", "TIME"):
            dates.append({"text": ent.text, "label": ent.label_})

    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    return {
        "entities": entities,
        "numbers": numbers,
        "dates": dates,
        "sentences": sentences,
        "n_sentences": len(sentences),
    }


def format_entity_hints(pre: dict) -> str:
    """Format spaCy entities as a hint block for prompts."""
    if not pre["entities"]:
        return ""

    lines = []
    for e in pre["entities"]:
        lines.append(f"  - {e['text']} ({e['label']})")

    hint = "MINIMUM ENTITY SET detected by pre-processing. Your extraction MUST include ALL of these AND more. This list is INCOMPLETE — there are many additional entities in the document that you MUST find:\n"
    hint += "\n".join(lines)
    hint += "\n\nIMPORTANT: The list above is a FLOOR, not a ceiling. You should extract at LEAST twice as many entities as shown above. Read every sentence and find entities the pre-processing missed."
    return hint


# ========================================================================
# PROMPTS — Three versions: Raw, Pre-seeded, Chained
# ========================================================================

# --- P1: Entity extraction (same for all phases, with optional pre-seed) ---
def prompt_p1(source: str, hints: str = "") -> str:
    hint_block = f"\n\n{hints}\n" if hints else ""
    return f"""You are an entity extraction engine. Read the source document and extract EVERY person, organization, place, and thing mentioned.
{hint_block}
For EACH entity, provide:
- name: exact name as used in the document
- type: PERSON / ORG / PLACE / PRODUCT / OTHER
- aliases: other names used for the same entity (nicknames, pronouns, titles)
- attributes: dict of all stated properties (age, role, title, description, condition, status)

CRITICAL: Extract EVERY entity, even minor ones. If the document says "female companion", that is an entity — do NOT change it to "wife" or "girlfriend".

Return ONLY valid JSON: {{"entities": [...]}}

Source:
\"\"\"{source}\"\"\""""


# --- P2: Event extraction — RAW vs CHAINED ---
def prompt_p2_raw(source: str) -> str:
    return f"""You are an event extraction engine. Read the source document and extract EVERY event or action that occurs.

For EACH event, provide:
- who: the actor (use exact name from document)
- action: the verb/action (use exact wording from document)
- target: who/what was acted upon (if applicable)
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)
- sequence: approximate order (1, 2, 3...)

CRITICAL: Extract EVERY action, including passive ones ("was arrested", "was hospitalized"). Include both active and passive voice events. Do not skip minor events.

Return ONLY valid JSON: {{"events": [...]}}

Source:
\"\"\"{source}\"\"\""""


def prompt_p2_chained(source: str, p1_entities: list) -> str:
    entity_list = ", ".join(e.get("name", "") for e in p1_entities[:40])
    return f"""You are an event extraction engine. Read the source document and extract EVERY event or action.

These entities were found in the document: [{entity_list}]

For EACH entity above AND any other actor in the document, extract EVERY action they performed or that happened to them:
- who: the actor (use exact name from document)
- action: the verb/action (use exact wording from document)
- target: who/what was acted upon (if applicable)
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)
- sequence: approximate order (1, 2, 3...)

CRITICAL: Go through the document SENTENCE BY SENTENCE. For each sentence, ask: "What action happened here?" Extract it. Include passive voice ("was arrested", "was hospitalized"). Do not skip ANY event.

Return ONLY valid JSON: {{"events": [...]}}

Source:
\"\"\"{source}\"\"\""""


# --- P2b: Self-verification (Technique 3) ---
def prompt_p2_verify(source: str, p2_events: list) -> str:
    event_summary = json.dumps(p2_events[:30], indent=0)[:2000]
    return f"""You are a verification engine. Below is a source document and a list of events already extracted from it.

Your job: Read the source document SENTENCE BY SENTENCE. For each sentence, check if the events list below captures what happened. If ANY action in ANY sentence is missing from the events list, add it.

Already extracted events:
{event_summary}

ONLY output NEW events that are MISSING from the list above. If nothing is missing, return {{"new_events": []}}

Return ONLY valid JSON: {{"new_events": [...]}}

Source:
\"\"\"{source}\"\"\""""


# --- P3: Relationship extraction — RAW vs CHAINED ---
def prompt_p3_raw(source: str) -> str:
    return f"""You are a relationship extraction engine. Read the source document and map ALL relationships between entities.

For EACH relationship, provide:
- entity1: first entity
- entity2: second entity
- relationship: type of connection (teammate, companion, employer, lawyer, victim, suspect, etc.)
- NOT_relationships: list of relationships that are explicitly NOT stated or would be INCORRECT

CRITICAL: Pay attention to the EXACT words used. If the document says "female companion", the relationship is "companion" — NOT "wife", NOT "girlfriend".

Return ONLY valid JSON: {{"relationships": [...]}}

Source:
\"\"\"{source}\"\"\""""


def prompt_p3_chained(source: str, p1_entities: list) -> str:
    entity_list = ", ".join(e.get("name", "") for e in p1_entities[:40])
    return f"""You are a relationship extraction engine. Map ALL relationships between entities in the document.

These entities were found: [{entity_list}]

For EVERY PAIR of entities that are connected, provide:
- entity1: first entity
- entity2: second entity
- relationship: type of connection (teammate, companion, employer, lawyer, victim, suspect, etc.)
- NOT_relationships: list of relationships that are explicitly NOT stated or would be INCORRECT

CRITICAL: Consider EVERY possible pair from the entity list. Pay attention to EXACT words — "female companion" is "companion", NOT "wife".

Return ONLY valid JSON: {{"relationships": [...]}}

Source:
\"\"\"{source}\"\"\""""


# --- P4: Numbers (same for all phases) ---
def prompt_p4(source: str, pre_numbers: list | None = None) -> str:
    hint = ""
    if pre_numbers:
        nums = ", ".join(n["text"] for n in pre_numbers[:20])
        hint = f"\nNumbers detected by pre-processing: [{nums}]. Make sure ALL of these are captured with their context.\n"
    return f"""You are a numerical fact extraction engine. Read the source document and extract EVERY number, date, quantity, and measurement.
{hint}
For EACH numerical fact, provide:
- entity: who or what the number describes
- number: the exact value (as stated in document)
- unit: what the number measures (years, people, dollars, etc.)
- describes: what aspect this number refers to
- context: the sentence where this number appears

CRITICAL: Capture the EXACT number as stated. "15 years" is 15, unit="years". "Three women" is 3, unit="people".

Return ONLY valid JSON: {{"numerical_facts": [...], "temporal_facts": [...]}}

Source:
\"\"\"{source}\"\"\""""


# --- P5: Negations (same for all phases) ---
def prompt_p5(source: str) -> str:
    return f"""You are a negation and boundary extraction engine. Read the source document and identify everything that is explicitly denied, negated, absent, or bounded.

Extract:
1. explicit_negations: Statements that use "not", "no", "never", "denied", "neither"
   - statement: what is negated
   - context: surrounding text

2. absent_information: Important information that is notably NOT provided
   - what: what information is missing
   - why_notable: why its absence matters

3. boundaries: Limits or qualifications on stated facts
   - fact: the stated fact
   - boundary: the limit or qualification

4. corrections: Where the document corrects a potential misconception
   - wrong: what might be incorrectly assumed
   - right: what the document actually says

Return ONLY valid JSON: {{"explicit_negations": [...], "absent_information": [...], "boundaries": [...], "corrections": [...]}}

Source:
\"\"\"{source}\"\"\""""


# ========================================================================
# API CALLER — Google AI Studio with JSON mode
# ========================================================================

def call_gemma(prompt: str, model_id: str = "gemma-4-31b-it",
               timeout: int = 300, json_mode: bool = True) -> tuple[dict | None, float]:
    """Call Google AI Studio API with optional JSON mode."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, 0.0

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )

    gen_config: dict = {"temperature": 0.0, "topP": 1.0}
    if json_mode:
        gen_config["responseMimeType"] = "application/json"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": gen_config,
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=float(timeout)) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

            result = json.loads(raw)
            if isinstance(result, list):
                result = {"entities": result}
            return result, time.time() - t0
    except json.JSONDecodeError as e:
        print(f"        JSON parse error: {str(e)[:60]}")
        return None, time.time() - t0
    except Exception as e:
        print(f"        API error: {str(e)[:80]}")
        return None, time.time() - t0


# ========================================================================
# METRICS
# ========================================================================

def _safe_len(data: dict | None, key: str) -> int:
    if not data:
        return 0
    val = data.get(key, [])
    return len(val) if isinstance(val, list) else 0


def measure_extraction(passes: dict) -> dict:
    p1 = passes.get("P1_entities")
    p2 = passes.get("P2_events")
    p3 = passes.get("P3_relationships")
    p4 = passes.get("P4_numbers")
    p5 = passes.get("P5_negations")

    n_entities = _safe_len(p1, "entities")
    n_events = _safe_len(p2, "events")
    n_relationships = _safe_len(p3, "relationships")
    n_numerical = _safe_len(p4, "numerical_facts")
    n_temporal = _safe_len(p4, "temporal_facts")
    n_negations = _safe_len(p5, "explicit_negations")
    n_absent = _safe_len(p5, "absent_information")
    n_corrections = _safe_len(p5, "corrections")
    n_boundaries = _safe_len(p5, "boundaries")

    n_passes_ok = sum(1 for p in [p1, p2, p3, p4, p5] if p is not None)

    return {
        "entities": n_entities,
        "events": n_events,
        "relationships": n_relationships,
        "numerical_facts": n_numerical,
        "temporal_facts": n_temporal,
        "negations": n_negations,
        "absent_info": n_absent,
        "corrections": n_corrections,
        "boundaries": n_boundaries,
        "total_items": (
            n_entities + n_events + n_relationships + n_numerical
            + n_temporal + n_negations + n_absent + n_corrections + n_boundaries
        ),
        "passes_ok": n_passes_ok,
    }


# ========================================================================
# PHASES
# ========================================================================

def run_phase_a_raw(source: str, source_idx: int) -> dict:
    """Phase 33a: Raw extraction (same as Exp 32)."""
    passes = {}
    times = {}

    prompts = [
        ("P1_entities", prompt_p1(source)),
        ("P2_events", prompt_p2_raw(source)),
        ("P3_relationships", prompt_p3_raw(source)),
        ("P4_numbers", prompt_p4(source)),
        ("P5_negations", prompt_p5(source)),
    ]

    for pass_name, prompt in prompts:
        data, elapsed = call_gemma(prompt)
        times[pass_name] = round(elapsed, 2)
        if data:
            passes[pass_name] = data
            counts = {k: len(v) if isinstance(v, list) else 1 for k, v in data.items()}
            print(f"      ✓ {pass_name}: {counts} ({elapsed:.1f}s)")
        else:
            passes[pass_name] = None
            print(f"      ✗ {pass_name}: FAILED ({elapsed:.1f}s)")

    return {"passes": passes, "times": times, "metrics": measure_extraction(passes)}


def run_phase_b_chained(source: str, source_idx: int, pre: dict) -> dict:
    """Phase 33b: spaCy pre-seeded + chained P1→P2/P3."""
    passes = {}
    times = {}
    hints = format_entity_hints(pre)

    # Step 1: P1 with entity hints
    print("      [P1 with spaCy hints]")
    data, elapsed = call_gemma(prompt_p1(source, hints))
    times["P1_entities"] = round(elapsed, 2)
    if data:
        passes["P1_entities"] = data
        p1_entities = data.get("entities", [])
        print(f"      ✓ P1_entities: {len(p1_entities)} entities ({elapsed:.1f}s)")
    else:
        passes["P1_entities"] = None
        p1_entities = []
        print(f"      ✗ P1_entities: FAILED ({elapsed:.1f}s)")

    # Step 2: P2 CHAINED with P1 entities
    print("      [P2 chained with P1 entities]")
    data, elapsed = call_gemma(prompt_p2_chained(source, p1_entities))
    times["P2_events"] = round(elapsed, 2)
    if data:
        passes["P2_events"] = data
        n_events = len(data.get("events", []))
        print(f"      ✓ P2_events: {n_events} events ({elapsed:.1f}s)")
    else:
        # Retry with truncated source (first 4000 chars)
        print(f"      ✗ P2_events: FAILED ({elapsed:.1f}s) — retrying with truncated source...")
        truncated = source[:4000]
        data, elapsed2 = call_gemma(prompt_p2_chained(truncated, p1_entities))
        times["P2_events_retry"] = round(elapsed2, 2)
        if data:
            passes["P2_events"] = data
            n_events = len(data.get("events", []))
            print(f"      ✓ P2_events (retry): {n_events} events ({elapsed2:.1f}s)")
        else:
            # Final fallback: raw P2 without chaining (simpler prompt may work)
            print(f"      ✗ P2_events retry FAILED ({elapsed2:.1f}s) — fallback to raw P2...")
            data, elapsed3 = call_gemma(prompt_p2_raw(truncated))
            times["P2_events_fallback"] = round(elapsed3, 2)
            if data:
                passes["P2_events"] = data
                n_events = len(data.get("events", []))
                print(f"      ✓ P2_events (fallback): {n_events} events ({elapsed3:.1f}s)")
            else:
                passes["P2_events"] = None
                print(f"      ✗ P2_events: ALL ATTEMPTS FAILED")

    # Step 3: P3 CHAINED with P1 entities
    print("      [P3 chained with P1 entities]")
    data, elapsed = call_gemma(prompt_p3_chained(source, p1_entities))
    times["P3_relationships"] = round(elapsed, 2)
    if data:
        passes["P3_relationships"] = data
        n_rels = len(data.get("relationships", []))
        print(f"      ✓ P3_relationships: {n_rels} relationships ({elapsed:.1f}s)")
    else:
        passes["P3_relationships"] = None
        print(f"      ✗ P3_relationships: FAILED ({elapsed:.1f}s)")

    # Step 4: P4 with number hints from spaCy
    print("      [P4 with spaCy number hints]")
    data, elapsed = call_gemma(prompt_p4(source, pre.get("numbers")))
    times["P4_numbers"] = round(elapsed, 2)
    if data:
        passes["P4_numbers"] = data
        counts = {k: len(v) if isinstance(v, list) else 1 for k, v in data.items()}
        print(f"      ✓ P4_numbers: {counts} ({elapsed:.1f}s)")
    else:
        passes["P4_numbers"] = None
        print(f"      ✗ P4_numbers: FAILED ({elapsed:.1f}s)")

    # Step 5: P5 (same as raw)
    data, elapsed = call_gemma(prompt_p5(source))
    times["P5_negations"] = round(elapsed, 2)
    if data:
        passes["P5_negations"] = data
        counts = {k: len(v) if isinstance(v, list) else 1 for k, v in data.items()}
        print(f"      ✓ P5_negations: {counts} ({elapsed:.1f}s)")
    else:
        passes["P5_negations"] = None
        print(f"      ✗ P5_negations: FAILED ({elapsed:.1f}s)")

    return {"passes": passes, "times": times, "metrics": measure_extraction(passes)}


def run_phase_c_verified(source: str, source_idx: int, pre: dict) -> dict:
    """Phase 33c: Chained + self-verification loop."""
    # Run phase B first
    result = run_phase_b_chained(source, source_idx, pre)
    passes = result["passes"]
    times = result["times"]

    # Step 6: Self-verification on P2 events
    p2_data = passes.get("P2_events")
    if p2_data and p2_data.get("events"):
        print("      [P2b self-verification]")
        verify_data, elapsed = call_gemma(
            prompt_p2_verify(source, p2_data["events"])
        )
        times["P2b_verify"] = round(elapsed, 2)

        if verify_data:
            new_events = verify_data.get("new_events", [])
            if new_events:
                # Merge new events into P2
                original_count = len(p2_data["events"])
                p2_data["events"].extend(new_events)
                passes["P2_events"] = p2_data
                print(f"      ✓ P2b_verify: +{len(new_events)} new events "
                      f"(total: {original_count}→{len(p2_data['events'])}) ({elapsed:.1f}s)")
            else:
                print(f"      ✓ P2b_verify: 0 new events (extraction was complete) ({elapsed:.1f}s)")
        else:
            print(f"      ✗ P2b_verify: FAILED ({elapsed:.1f}s)")
    else:
        print("      ⊘ P2b_verify: skipped (no P2 events to verify)")

    # Recalculate metrics after verification
    result["metrics"] = measure_extraction(passes)
    return result


# ========================================================================
# MAIN
# ========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp 33: Optimized SLM Extraction")
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--phases", nargs="+", default=["b", "c"],
                        help="Phases to run: a (raw), b (chained), c (verified)")
    parser.add_argument("--baseline-path", default="experiments/exp31_multipass_fact_tables.json")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 33 — Optimized SLM Extraction Pipeline")
    print("=" * 70)
    print("  Model: Gemma 4 31B via Google AI Studio")
    print("  T1: spaCy NER pre-seeding")
    print("  T2: Chained P1→P2/P3 extraction")
    print("  T3: Self-verification loop (P2b)")
    print("  T4: JSON mode (responseMimeType)")
    print("=" * 70)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    # Load RAGTruth sources
    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))

    source_map = {}
    for case in cases:
        source = "\n".join(case.request.source_context or []) if case.request.source_context else ""
        source_hash = hash(source)
        if source_hash not in source_map:
            source_map[source_hash] = {"source": source, "case_ids": []}
        source_map[source_hash]["case_ids"].append(case.case_id)

    sources = list(source_map.values())
    if args.max_sources:
        sources = sources[:args.max_sources]

    # Load Gemini baseline for comparison
    baseline_metrics = []
    if os.path.exists(args.baseline_path):
        with open(args.baseline_path) as f:
            baseline_data = json.load(f)
        # Compute baseline metrics per source
        for src_data in sources:
            primary_id = src_data["case_ids"][0]
            if primary_id in baseline_data:
                bl_passes = baseline_data[primary_id].get("passes", {})
                baseline_metrics.append(measure_extraction(bl_passes))
            else:
                baseline_metrics.append(None)
        print(f"  Baseline loaded: {sum(1 for b in baseline_metrics if b)} sources")
    else:
        print(f"  WARNING: No baseline at {args.baseline_path}")

    print(f"  Sources: {len(sources)}")
    print(f"  Phases: {args.phases}")

    # Pre-process all sources with spaCy
    print(f"\n  Running spaCy NER pre-processing...")
    t_spacy = time.time()
    pre_extractions = []
    for si, src_data in enumerate(sources):
        pre = spacy_pre_extract(src_data["source"][:6000])
        pre_extractions.append(pre)
        print(f"    Source {si}: {len(pre['entities'])} entities, "
              f"{len(pre['numbers'])} numbers, {len(pre['dates'])} dates, "
              f"{pre['n_sentences']} sentences")
    print(f"  spaCy pre-processing: {time.time() - t_spacy:.1f}s")

    # Run phases
    all_results = {}
    t0 = time.time()

    phase_runners = {
        "a": ("33a: Raw (Exp 32 baseline)", run_phase_a_raw),
        "b": ("33b: spaCy + Chained P1→P2/P3", None),  # handled below
        "c": ("33c: Chained + Self-verification", None),
    }

    for phase_key in args.phases:
        phase_name = {
            "a": "33a: Raw (no optimization)",
            "b": "33b: spaCy pre-seed + chained P1→P2/P3",
            "c": "33c: Chained + self-verification loop",
        }.get(phase_key, f"Phase {phase_key}")

        print(f"\n{'='*60}")
        print(f"  PHASE {phase_name}")
        print(f"{'='*60}")

        phase_results = []

        for si, src_data in enumerate(sources):
            source_text = src_data["source"][:6000]
            case_ids = src_data["case_ids"]
            pre = pre_extractions[si]

            print(f"\n    [{si+1}/{len(sources)}] Source for "
                  f"{', '.join(case_ids[:3])}{'...' if len(case_ids) > 3 else ''}")

            if phase_key == "a":
                result = run_phase_a_raw(source_text, si)
            elif phase_key == "b":
                result = run_phase_b_chained(source_text, si, pre)
            elif phase_key == "c":
                result = run_phase_c_verified(source_text, si, pre)
            else:
                continue

            phase_results.append({
                "source_idx": si,
                "case_ids": case_ids,
                "metrics": result["metrics"],
                "times": result["times"],
            })

            # Print comparison to baseline
            m = result["metrics"]
            bl = baseline_metrics[si] if si < len(baseline_metrics) else None
            if bl:
                pct = round(m["total_items"] / bl["total_items"] * 100, 1) if bl["total_items"] > 0 else 0
                print(f"    → Total: {m['total_items']} items "
                      f"(vs Gemini baseline: {bl['total_items']}, {pct}%)")
                print(f"      Entities: {m['entities']} vs {bl['entities']} "
                      f"({round(m['entities']/bl['entities']*100,1) if bl['entities'] else 0}%)")
                print(f"      Events:   {m['events']} vs {bl['events']} "
                      f"({round(m['events']/bl['events']*100,1) if bl['events'] else 0}%)")
                print(f"      Rels:     {m['relationships']} vs {bl['relationships']} "
                      f"({round(m['relationships']/bl['relationships']*100,1) if bl['relationships'] else 0}%)")
            else:
                print(f"    → Total: {m['total_items']} items (no baseline)")

        all_results[phase_key] = phase_results

    elapsed = time.time() - t0

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print(f"\n{'='*80}")
    print("COMPARISON: Optimization Techniques vs Gemini Baseline")
    print(f"{'='*80}")

    # Aggregate baseline
    bl_totals = [b["total_items"] for b in baseline_metrics if b]
    bl_entities = [b["entities"] for b in baseline_metrics if b]
    bl_events = [b["events"] for b in baseline_metrics if b]
    bl_rels = [b["relationships"] for b in baseline_metrics if b]

    bl_avg_total = sum(bl_totals) / len(bl_totals) if bl_totals else 1
    bl_avg_ent = sum(bl_entities) / len(bl_entities) if bl_entities else 1
    bl_avg_evt = sum(bl_events) / len(bl_events) if bl_events else 1
    bl_avg_rel = sum(bl_rels) / len(bl_rels) if bl_rels else 1

    print(f"\n  {'Phase':<40s} {'Total':>6s} {'vs BL':>6s} "
          f"{'Ent':>5s} {'Evt':>5s} {'Rel':>5s}")
    print(f"  {'-'*70}")
    print(f"  {'Gemini 2.5 Flash (BASELINE)':<40s} {bl_avg_total:>6.1f} {'BASE':>6s} "
          f"{bl_avg_ent:>5.1f} {bl_avg_evt:>5.1f} {bl_avg_rel:>5.1f}")

    # Exp 32 raw result (hardcoded from previous run)
    print(f"  {'Exp 32: Gemma 31B raw':<40s} {'137.5':>6s} {'66.7%':>6s} "
          f"{'61.0':>5s} {'34.0':>5s} {'11.5':>5s}")

    for phase_key, phase_results in all_results.items():
        if not phase_results:
            continue
        phase_name = {
            "a": "33a: Raw + JSON mode",
            "b": "33b: spaCy + Chained",
            "c": "33c: Chained + Verified",
        }.get(phase_key, f"Phase {phase_key}")

        avg_total = sum(r["metrics"]["total_items"] for r in phase_results) / len(phase_results)
        avg_ent = sum(r["metrics"]["entities"] for r in phase_results) / len(phase_results)
        avg_evt = sum(r["metrics"]["events"] for r in phase_results) / len(phase_results)
        avg_rel = sum(r["metrics"]["relationships"] for r in phase_results) / len(phase_results)

        pct = round(avg_total / bl_avg_total * 100, 1) if bl_avg_total > 0 else 0

        print(f"  {phase_name:<40s} {avg_total:>6.1f} {pct:>5.1f}% "
              f"{avg_ent:>5.1f} {avg_evt:>5.1f} {avg_rel:>5.1f}")

    print(f"  {'-'*70}")

    # Decision
    print(f"\n  TECHNIQUE IMPACT:")
    if "b" in all_results and all_results["b"]:
        b_avg = sum(r["metrics"]["total_items"] for r in all_results["b"]) / len(all_results["b"])
        b_pct = round(b_avg / bl_avg_total * 100, 1)
        delta = round(b_pct - 66.7, 1)
        print(f"    T1+T2 (spaCy + chaining): {b_pct}% of baseline "
              f"({'+'if delta>0 else ''}{delta}pp vs raw)")

    if "c" in all_results and all_results["c"]:
        c_avg = sum(r["metrics"]["total_items"] for r in all_results["c"]) / len(all_results["c"])
        c_pct = round(c_avg / bl_avg_total * 100, 1)
        delta = round(c_pct - 66.7, 1)
        print(f"    T1+T2+T3 (+ verification): {c_pct}% of baseline "
              f"({'+'if delta>0 else ''}{delta}pp vs raw)")

        if c_pct >= 90:
            print(f"\n  ✅ SCIENCE GATE: PASS — {c_pct}% ≥ 90% threshold")
            print(f"     → Proceed to local deployment (Gemma E4B / Qwen 14B)")
        elif c_pct >= 80:
            print(f"\n  ⚠️  SCIENCE GATE: PARTIAL — {c_pct}% (80-90% range)")
            print(f"     → Consider hybrid: SLM for P1+P4, Gemini for P2+P3+P5")
        else:
            print(f"\n  ❌ SCIENCE GATE: FAIL — {c_pct}% < 80% threshold")
            print(f"     → Techniques insufficient. Consider fine-tuning or keep Gemini.")

    # Save
    output = {
        "experiment": "exp33_optimized_extraction",
        "model": "gemma-4-31b-it",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1),
        "n_sources": len(sources),
        "phases_run": args.phases,
        "techniques": {
            "T1": "spaCy NER pre-seeding",
            "T2": "Chained P1→P2/P3 extraction",
            "T3": "Self-verification loop (P2b)",
            "T4": "JSON mode (responseMimeType)",
        },
        "baseline_avg": {
            "total_items": round(bl_avg_total, 1),
            "entities": round(bl_avg_ent, 1),
            "events": round(bl_avg_evt, 1),
            "relationships": round(bl_avg_rel, 1),
        },
        "results": {k: v for k, v in all_results.items()},
    }

    output_path = "experiments/exp33_optimized_extraction_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
