"""
Experiment 38: Source Representation Formats for L3 Semantic Verification.

Tests 4 source compilation formats using Gemini, then evaluates each format's
ability to detect known hallucinations via an LLM verification call.

The 4 formats:
  38a: Event-question graph (WHO/WHAT/WHEN/WHERE/WHY/HOW per event, "not stated" tracking)
  38b: Forward + backward narrative pass ("what next" + "why this")
  38c: Hybrid mind-map + knowledge graph (event hubs with typed edges)
  38d: Small stories (≤5 sentence natural language summaries per thread)

Test source: ragtruth_24 (shared by cases 24-29, 16 sentences about Keonna Thomas)
Test hallucinations:
  - ragtruth_28 S0: "Three women including Thomas were charged with attempting to join ISIS"
    (EVIDENT CONFLICT: only Thomas charged, others arrested separately)
  - ragtruth_29 S1: "She was arrested on March 26"
    (EVIDENT CONFLICT: S8 says arrest details unknown, March 26 is ticket purchase date)

Usage:
    export GEMINI_API_KEY=your_key
    python experiments/exp38_source_representations.py

Each format is extracted once, then used for verification of both hallucinations.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_TIMEOUT = 120.0
OUTPUT_DIR = Path("experiments/exp38_results")

# ---------------------------------------------------------------------------
# Source & hallucination test cases
# ---------------------------------------------------------------------------
SOURCE_TEXT = """S0: The FBI charged a Philadelphia woman on Thursday with trying to travel overseas to fight for ISIS.
S1: She's one of three women arrested this week on terror charges.
S2: Two New York women were also taken into custody.
S3: An FBI complaint cites numerous social media messages dating back to August 2013 that were sent by Keonna Thomas, 30, also known as "Young Lioness" and "Fatayat Al Khilafah."
S4: One Twitter message said, "If we truly knew the realities ... we all would be rushing to join our brothers in the front lines pray ALLAH accept us as shuhada [martyrs]."
S5: Another said, "When you're a mujahid [violent jihadi fighter] your death becomes a wedding."
S6: The FBI said Thomas purchased an electronic visa to Turkey on March 23.
S7: Turkey is known as the easiest place from which to enter Syria and join ISIS.
S8: An ISIS manual advises recruits to buy round-trip tickets to vacation spots such as Spain and then purchase tickets for their real destination once they arrive overseas, the FBI said.
S9: On March 26, Thomas purchased a ticket to Barcelona, with a March 29 departure and an April 15 return to the United States, the complaint said.
S10: It's not clear when or where she was arrested.
S11: She was charged with knowingly attempting to provide material support and resources to a designated foreign terrorist organization.
S12: She could be sentenced to 15 years in prison.
S13: On Thursday, Noelle Velentzas, 28, and her former roommate, Asia Siddiqui, 31, were arrested in New York and accused of planning to build an explosive device for attacks in the United States, federal prosecutors said.
S14: In the past 18 months, the Justice Department's National Security Division has prosecuted or is prosecuting more than 30 cases of people attempting to travel abroad to join or provide support to terrorist groups.
S15: Of those cases, 18 allegedly involve support to ISIS.
S16: "The terrorist threat is more decentralized, more diffuse, more complicated," Homeland Security Secretary Jeh Johnson told reporters Thursday.
S17: "It involves the potential lone wolf actor, it involves the effective use of social media, the Internet."
"""

HALLUCINATION_TESTS = [
    {
        "id": "ragtruth_28_S0",
        "sentence": "Three women, including Keonna Thomas of Philadelphia, were charged with attempting to join ISIS this week.",
        "ground_truth": "hallucinated",
        "type": "EVIDENT CONFLICT",
        "explanation": "Only Thomas was charged. The other two (Velentzas, Siddiqui) were arrested and accused, not charged with the same offense. Source says 'charged' for Thomas (S0) but 'arrested and accused' for the NY women (S13).",
    },
    {
        "id": "ragtruth_29_S1",
        "sentence": "She was arrested on March 26 and could face 15 years in prison.",
        "ground_truth": "hallucinated",
        "type": "EVIDENT CONFLICT",
        "explanation": "S10 explicitly says 'It's not clear when or where she was arrested.' March 26 is the date Thomas purchased a ticket to Barcelona (S9), not her arrest date. Date reassigned from PURCHASE to ARREST event.",
    },
    {
        "id": "ragtruth_24_S0_clean",
        "sentence": "The FBI has charged a Philadelphia woman, Keonna Thomas, with trying to travel overseas to fight for ISIS.",
        "ground_truth": "grounded",
        "type": "clean",
        "explanation": "Accurate paraphrase of S0 + S3 identity resolution.",
    },
    {
        "id": "ragtruth_26_S2_clean",
        "sentence": "She purchased an electronic visa for Turkey on March 23 and later bought a round-trip ticket to Barcelona on March 26.",
        "ground_truth": "grounded",
        "type": "clean",
        "explanation": "Accurate combination of S6 (visa, March 23) and S9 (ticket, March 26).",
    },
]


# ---------------------------------------------------------------------------
# Gemini API helper
# ---------------------------------------------------------------------------
def call_gemini(prompt: str, *, temperature: float = 0.0) -> str:
    """Call Gemini API. Returns text response."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 1.0,
            "responseMimeType": "application/json",
        },
    }

    with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
        resp = client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()

    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return text


def call_gemini_text(prompt: str, *, temperature: float = 0.0) -> str:
    """Call Gemini API without JSON mode. Returns raw text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "topP": 1.0},
    }

    with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
        resp = client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()

    return data["candidates"][0]["content"]["parts"][0]["text"]


# ===================================================================
# FORMAT 38a: Event-Question Graph
# ===================================================================

PROMPT_38A = """You are building an event-question knowledge graph from a source document.

SOURCE DOCUMENT (each sentence has an ID):
{source}

INSTRUCTIONS:
Process each sentence sequentially (S0, S1, S2, ...). For each sentence:

1. Identify if it contains an EVENT (something happened), ENRICHMENT (adds detail to a previous event/entity), CONTEXT (broader pattern), BACKGROUND (general knowledge), EXPLICIT_UNKNOWN (source says it doesn't know), or QUOTE (attributed statement).

2. For each EVENT, fill these slots from ONLY what the sentence says:
   - WHO_actor, WHO_target, WHAT (action), WHEN, WHERE, WHERE_TO, WHERE_FROM, WHY, HOW, RESULT, STATUS (completed/attempting/unknown)
   - For slots the sentence does NOT provide, write "NOT_STATED"
   - For slots the source EXPLICITLY says are unknown, write "EXPLICITLY_UNKNOWN" and note the source sentence

3. For ENRICHMENT sentences, note which previous event/entity they enrich and what new information they add.

4. Track entity resolutions (e.g., "Philadelphia woman" from S0 = "Keonna Thomas" from S3).

5. After processing all sentences, list all EXPLICITLY_UNKNOWN slots — these are hallucination traps.

Return JSON with this structure:
{{
  "events": [
    {{
      "event_id": "EVT_1",
      "source_sentence": "S0",
      "description": "FBI charged Philadelphia woman",
      "slots": {{
        "WHO_actor": {{"value": "FBI", "source": "S0"}},
        "WHO_target": {{"value": "a Philadelphia woman", "source": "S0"}},
        "WHAT": {{"value": "charged", "source": "S0"}},
        "WHEN": {{"value": "Thursday", "source": "S0"}},
        "WHERE": "NOT_STATED",
        "WHY": {{"value": "trying to travel overseas to fight for ISIS", "source": "S0"}},
        "HOW": "NOT_STATED",
        "RESULT": "NOT_STATED",
        "STATUS": {{"value": "attempting (travel not completed)", "source": "S0"}}
      }}
    }}
  ],
  "enrichments": [
    {{
      "source_sentence": "S3",
      "enriches": "EVT_1.WHO_target",
      "new_info": "Name: Keonna Thomas, Age: 30, Aliases: Young Lioness, Fatayat Al Khilafah"
    }}
  ],
  "entity_resolutions": [
    {{"from": "a Philadelphia woman", "to": "Keonna Thomas", "resolved_at": "S3"}}
  ],
  "explicit_unknowns": [
    {{"slot": "arrest_time", "description": "when she was arrested", "source": "S10"}},
    {{"slot": "arrest_location", "description": "where she was arrested", "source": "S10"}}
  ],
  "sentence_coverage": {{
    "covered": ["S0", "S1", "S3"],
    "uncovered": ["S14", "S15"]
  }}
}}
"""


# ===================================================================
# FORMAT 38b: Forward + Backward Narrative Pass
# ===================================================================

PROMPT_38B_FORWARD = """You are building a narrative graph by tracing "what happens next" for each entity/event.

SOURCE DOCUMENT (each sentence has an ID):
{source}

INSTRUCTIONS — FORWARD PASS ("What happens next?"):

1. Start with S0. Identify the main entities and events.
2. For each entity/event in S0, trace forward through the remaining sentences:
   - What is the NEXT thing that happens to this entity?
   - What is the NEXT piece of evidence related to this event?
3. Continue until no more sentences connect to this thread.
4. Record which sentences are reached and which are not.

Return JSON:
{{
  "threads": [
    {{
      "thread_name": "Keonna Thomas thread",
      "starting_entity": "Philadelphia woman / Keonna Thomas",
      "chain": [
        {{"sentence": "S0", "what": "FBI charged her", "connection_to_next": "identity revealed in S3"}},
        {{"sentence": "S3", "what": "Named as Keonna Thomas, 30", "connection_to_next": "her actions detailed in S6"}},
        {{"sentence": "S6", "what": "Purchased visa to Turkey March 23", "connection_to_next": "second purchase in S9"}}
      ]
    }},
    {{
      "thread_name": "NY women thread",
      "starting_entity": "Two New York women",
      "chain": [...]
    }}
  ],
  "sentences_reached": ["S0", "S1", "S2", "S3"],
  "sentences_unreached": ["S14", "S15"]
}}
"""

PROMPT_38B_BACKWARD = """You are building a narrative graph by tracing "why did this happen" backwards.

SOURCE DOCUMENT (each sentence has an ID):
{source}

FORWARD PASS RESULTS (already completed):
{forward_result}

INSTRUCTIONS — BACKWARD PASS ("Why did this happen?"):

1. Start from the UNREACHED sentences and from the LAST events in each thread.
2. For each, ask "WHY did this happen?" and trace backward to find the causal chain.
3. This should connect the unreached sentences to the main narrative.
4. Also trace backward from key events to find their motivations and causes.

Return JSON:
{{
  "causal_chains": [
    {{
      "chain_name": "Why was Thomas charged?",
      "chain": [
        {{"sentence": "S11", "fact": "charged with material support", "caused_by": "S0 charge event"}},
        {{"sentence": "S0", "fact": "FBI charged her", "caused_by": "evidence of travel plans"}},
        {{"sentence": "S6", "fact": "purchased visa", "caused_by": "radicalization"}},
        {{"sentence": "S3", "fact": "social media since 2013", "caused_by": "root cause of case"}}
      ]
    }}
  ],
  "newly_connected_sentences": ["S14", "S15"],
  "still_unreached": [],
  "total_coverage": 18
}}
"""


# ===================================================================
# FORMAT 38c: Hybrid Mind-Map + Knowledge Graph
# ===================================================================

PROMPT_38C = """You are building a hybrid event-centered knowledge graph from a source document.

SOURCE DOCUMENT (each sentence has an ID):
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

CRITICAL: For every edge, note the source sentence ID. For connections between events, note the relationship type.

Return JSON:
{{
  "event_hubs": [
    {{
      "event_id": "EVT_1",
      "action": "charged",
      "source_sentence": "S0",
      "edges": [
        {{"type": "agent_of", "target": "FBI", "source": "S0"}},
        {{"type": "target_of", "target": "Keonna Thomas", "source": "S0,S3"}},
        {{"type": "when", "target": "Thursday", "source": "S0"}},
        {{"type": "reason", "target": "TRAVEL_ATTEMPT", "source": "S0"}},
        {{"type": "specified_by", "target": "EVT_CHARGE_DETAIL", "source": "S11"}}
      ]
    }}
  ],
  "entities": [
    {{
      "name": "Keonna Thomas",
      "aliases": ["Philadelphia woman", "Young Lioness", "Fatayat Al Khilafah"],
      "attributes": {{"age": 30, "location": "Philadelphia"}},
      "participates_in": ["EVT_1", "EVT_2", "EVT_3"],
      "source_sentences": ["S0", "S3", "S6", "S9"]
    }}
  ],
  "inter_event_edges": [
    {{"from": "EVT_2_visa", "to": "EVT_3_ticket", "type": "leads_to", "source": "S6,S9"}},
    {{"from": "EVT_2_visa", "to": "EVT_1_charge", "type": "evidence_for", "source": "S6"}}
  ],
  "explicit_unknowns": [
    {{"about": "Keonna Thomas arrest", "unknown_slots": ["when", "where"], "source": "S10"}}
  ]
}}
"""


# ===================================================================
# FORMAT 38d: Small Stories (≤5 sentences per thread)
# ===================================================================

PROMPT_38D = """You are compiling a source document into small, self-contained stories.

SOURCE DOCUMENT (each sentence has an ID):
{source}

INSTRUCTIONS:

Create small stories (MAXIMUM 5 sentences each) that capture the source's information.
Each story should be:
- Self-contained (readable without the others)
- Factually precise (use exact names, dates, numbers from the source)
- Explicit about what is NOT known ("It is not clear when she was arrested")
- Cover one narrative thread

Group by thread:
- Thomas's case (charges, evidence, travel plans)
- NY women's case (Velentzas, Siddiqui)
- Broader context (prosecution statistics, official statements)

CRITICAL RULES:
- Do NOT add any information not in the source
- Do NOT infer dates or details — if the source says "not clear", say "not clear"
- Use exact numbers (30 years old, March 23, March 26, 15 years, 3 women)
- Each story must note which source sentences (S0-S17) it draws from

Return JSON:
{{
  "stories": [
    {{
      "thread": "Thomas case",
      "story": "The FBI charged Keonna Thomas, a 30-year-old Philadelphia woman, on Thursday with attempting to provide material support to ISIS. Thomas, also known as 'Young Lioness,' had sent social media messages supporting ISIS since August 2013. She purchased an electronic visa to Turkey on March 23 and a round-trip ticket to Barcelona on March 26. It is not clear when or where she was arrested. She could face up to 15 years in prison.",
      "source_sentences": ["S0", "S3", "S6", "S9", "S10", "S11", "S12"],
      "key_facts": ["charged on Thursday", "age 30", "visa March 23", "ticket March 26", "arrest details unknown", "15 years max"]
    }}
  ]
}}
"""


# ===================================================================
# Verification prompt — same for all formats
# ===================================================================

VERIFICATION_PROMPT = """You are a hallucination detector. Given a COMPILED SOURCE and a CLAIM, determine if the claim is GROUNDED (supported by the source) or HALLUCINATED (contradicts or fabricates beyond the source).

COMPILED SOURCE:
{compiled_source}

CLAIM TO VERIFY:
"{claim}"

INSTRUCTIONS:
1. For each factual assertion in the claim, check if the source supports it.
2. If the source EXPLICITLY says something is unknown, and the claim provides a specific value, that is HALLUCINATION.
3. If the claim combines facts in a way the source does not support (e.g., attributing action X to person Y when the source attributes X to person Z), that is HALLUCINATION.
4. If the claim accurately paraphrases the source, it is GROUNDED.

Return JSON:
{{
  "verdict": "GROUNDED" or "HALLUCINATED",
  "confidence": 0.0 to 1.0,
  "reasoning": "step by step explanation",
  "specific_issues": [
    {{"claim_part": "the specific part", "source_says": "what source actually says", "issue": "contradiction/fabrication/gap_filling"}}
  ]
}}
"""


# ===================================================================
# Main experiment runner
# ===================================================================
def run_extraction(format_id: str, prompt: str) -> dict:
    """Run a single extraction format and return parsed JSON."""
    print(f"\n{'='*60}")
    print(f"EXTRACTING FORMAT {format_id}")
    print(f"{'='*60}")

    start = time.time()
    raw = call_gemini(prompt.format(source=SOURCE_TEXT))
    elapsed = time.time() - start

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            print(f"  PARSE ERROR. Raw response saved.")
            result = {"_raw": raw, "_parse_error": True}

    result["_elapsed_seconds"] = round(elapsed, 1)
    print(f"  Extracted in {elapsed:.1f}s")
    return result


def format_for_verification(format_id: str, extraction: dict) -> str:
    """Convert extraction result to a text format for the verification prompt."""
    if format_id == "38a":
        # Event-question format: list events with slots
        lines = []
        for evt in extraction.get("events", []):
            lines.append(f"\nEVENT ({evt.get('source_sentence', '?')}): {evt.get('description', '')}")
            for slot, val in evt.get("slots", {}).items():
                if isinstance(val, dict):
                    lines.append(f"  {slot}: {val.get('value', '?')} [from {val.get('source', '?')}]")
                else:
                    lines.append(f"  {slot}: {val}")
        for eu in extraction.get("explicit_unknowns", []):
            lines.append(f"\nEXPLICITLY UNKNOWN: {eu.get('description', '')} [source: {eu.get('source', '?')}]")
        for er in extraction.get("entity_resolutions", []):
            lines.append(f"ENTITY: '{er.get('from', '')}' = '{er.get('to', '')}' (resolved at {er.get('resolved_at', '?')})")
        return "\n".join(lines)

    elif format_id == "38b":
        # Forward + backward: list threads and causal chains
        lines = []
        for thread in extraction.get("threads", []):
            lines.append(f"\nTHREAD: {thread.get('thread_name', '')}")
            for step in thread.get("chain", []):
                lines.append(f"  {step.get('sentence', '?')}: {step.get('what', '')}")
        for chain in extraction.get("causal_chains", []):
            lines.append(f"\nCAUSAL: {chain.get('chain_name', '')}")
            for step in chain.get("chain", []):
                lines.append(f"  {step.get('sentence', '?')}: {step.get('fact', '')} <- {step.get('caused_by', '')}")
        return "\n".join(lines)

    elif format_id == "38c":
        # Hybrid: event hubs + entities + edges
        lines = []
        for hub in extraction.get("event_hubs", []):
            lines.append(f"\nEVENT HUB: {hub.get('action', '')} ({hub.get('source_sentence', '?')})")
            for edge in hub.get("edges", []):
                lines.append(f"  {edge.get('type', '?')} -> {edge.get('target', '?')} [{edge.get('source', '?')}]")
        for ent in extraction.get("entities", []):
            attrs = ent.get("attributes", {})
            lines.append(f"\nENTITY: {ent.get('name', '')} (age={attrs.get('age','?')}, loc={attrs.get('location','?')})")
            lines.append(f"  Aliases: {', '.join(ent.get('aliases', []))}")
        for eu in extraction.get("explicit_unknowns", []):
            lines.append(f"\nEXPLICITLY UNKNOWN: {eu.get('about', '')} - {eu.get('unknown_slots', [])}")
        return "\n".join(lines)

    elif format_id == "38d":
        # Small stories: just concatenate the stories
        lines = []
        for story in extraction.get("stories", []):
            lines.append(f"\n[{story.get('thread', '')}]")
            lines.append(story.get("story", ""))
            lines.append(f"Key facts: {', '.join(story.get('key_facts', []))}")
        return "\n".join(lines)

    return json.dumps(extraction, indent=2)[:3000]


def run_verification(format_id: str, compiled_source: str, test_case: dict) -> dict:
    """Run verification of a claim against a compiled source."""
    prompt = VERIFICATION_PROMPT.format(
        compiled_source=compiled_source,
        claim=test_case["sentence"],
    )

    start = time.time()
    raw = call_gemini(prompt)
    elapsed = time.time() - start

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL)
        if match:
            result = json.loads(match.group(1))
        else:
            result = {"verdict": "ERROR", "reasoning": raw[:500]}

    result["_elapsed_seconds"] = round(elapsed, 1)
    result["_test_id"] = test_case["id"]
    result["_expected"] = test_case["ground_truth"]
    result["_correct"] = (
        (result.get("verdict", "").upper() == "HALLUCINATED" and test_case["ground_truth"] == "hallucinated")
        or (result.get("verdict", "").upper() == "GROUNDED" and test_case["ground_truth"] == "grounded")
    )

    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    # ---- Phase 1: Extract all 4 formats ----
    print("\n" + "=" * 70)
    print("PHASE 1: SOURCE EXTRACTION — 4 Formats")
    print("=" * 70)

    extractions = {}

    # 38a: Event-question graph
    extractions["38a"] = run_extraction("38a", PROMPT_38A)

    # 38b: Forward pass first, then backward
    print(f"\n{'='*60}")
    print(f"EXTRACTING FORMAT 38b (forward pass)")
    print(f"{'='*60}")
    start = time.time()
    fwd_raw = call_gemini(PROMPT_38B_FORWARD.format(source=SOURCE_TEXT))
    fwd_elapsed = time.time() - start
    try:
        fwd_result = json.loads(fwd_raw)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", fwd_raw, re.DOTALL)
        fwd_result = json.loads(match.group(1)) if match else {"_raw": fwd_raw}
    print(f"  Forward pass: {fwd_elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"EXTRACTING FORMAT 38b (backward pass)")
    print(f"{'='*60}")
    start = time.time()
    bwd_raw = call_gemini(
        PROMPT_38B_BACKWARD.format(
            source=SOURCE_TEXT,
            forward_result=json.dumps(fwd_result, indent=2)[:3000],
        )
    )
    bwd_elapsed = time.time() - start
    try:
        bwd_result = json.loads(bwd_raw)
    except json.JSONDecodeError:
        match = re.search(r"```json\s*(.*?)\s*```", bwd_raw, re.DOTALL)
        bwd_result = json.loads(match.group(1)) if match else {"_raw": bwd_raw}
    print(f"  Backward pass: {bwd_elapsed:.1f}s")

    extractions["38b"] = {
        **fwd_result,
        **bwd_result,
        "_elapsed_seconds": round(fwd_elapsed + bwd_elapsed, 1),
    }

    # 38c: Hybrid mind-map + KG
    extractions["38c"] = run_extraction("38c", PROMPT_38C)

    # 38d: Small stories
    extractions["38d"] = run_extraction("38d", PROMPT_38D)

    # Save raw extractions
    for fmt_id, ext in extractions.items():
        out_path = OUTPUT_DIR / f"extraction_{fmt_id}.json"
        with open(out_path, "w") as f:
            json.dump(ext, f, indent=2)
        print(f"  Saved {out_path}")

    # ---- Phase 2: Compile each format for verification ----
    print("\n" + "=" * 70)
    print("PHASE 2: COMPILE FORMATS FOR VERIFICATION")
    print("=" * 70)

    compiled = {}
    for fmt_id, ext in extractions.items():
        compiled[fmt_id] = format_for_verification(fmt_id, ext)
        # Save compiled version
        out_path = OUTPUT_DIR / f"compiled_{fmt_id}.txt"
        with open(out_path, "w") as f:
            f.write(compiled[fmt_id])
        print(f"  {fmt_id}: {len(compiled[fmt_id])} chars")

    # Also test raw source as baseline
    compiled["raw"] = SOURCE_TEXT
    print(f"  raw: {len(SOURCE_TEXT)} chars (baseline)")

    # ---- Phase 3: Verify hallucinations against each format ----
    print("\n" + "=" * 70)
    print("PHASE 3: HALLUCINATION VERIFICATION — 4 test cases × 5 formats")
    print("=" * 70)

    results_table = []

    for fmt_id in ["raw", "38a", "38b", "38c", "38d"]:
        fmt_results = []
        print(f"\n--- Format: {fmt_id} ---")
        for test in HALLUCINATION_TESTS:
            result = run_verification(fmt_id, compiled[fmt_id], test)
            fmt_results.append(result)
            status = "✓" if result["_correct"] else "✗"
            print(
                f"  {status} {test['id']}: "
                f"verdict={result.get('verdict', '?')} "
                f"(expected={test['ground_truth']}) "
                f"confidence={result.get('confidence', '?')} "
                f"[{result['_elapsed_seconds']}s]"
            )

        all_results[fmt_id] = fmt_results
        correct = sum(1 for r in fmt_results if r["_correct"])
        results_table.append(
            {
                "format": fmt_id,
                "correct": correct,
                "total": len(HALLUCINATION_TESTS),
                "accuracy": f"{correct}/{len(HALLUCINATION_TESTS)}",
                "avg_confidence": round(
                    sum(r.get("confidence", 0) for r in fmt_results) / len(fmt_results), 3
                ),
            }
        )

    # ---- Phase 4: Results comparison ----
    print("\n" + "=" * 70)
    print("PHASE 4: RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Format':<10} {'Correct':<10} {'Accuracy':<10} {'Avg Conf':<10}")
    print("-" * 40)
    for row in results_table:
        print(f"{row['format']:<10} {row['accuracy']:<10} {'':<10} {row['avg_confidence']:<10}")

    # Detailed per-case breakdown
    print(f"\n{'Test Case':<25} ", end="")
    for fmt_id in ["raw", "38a", "38b", "38c", "38d"]:
        print(f"{fmt_id:<8} ", end="")
    print()
    print("-" * 70)

    for i, test in enumerate(HALLUCINATION_TESTS):
        print(f"{test['id']:<25} ", end="")
        for fmt_id in ["raw", "38a", "38b", "38c", "38d"]:
            r = all_results[fmt_id][i]
            status = "✓" if r["_correct"] else "✗"
            verdict = r.get("verdict", "?")[:5]
            print(f"{status}{verdict:<7} ", end="")
        print()

    # Save full results
    out_path = OUTPUT_DIR / "exp38_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "summary": results_table,
                "per_format": {
                    fmt_id: [
                        {k: v for k, v in r.items() if not k.startswith("_")}
                        | {"test_id": r["_test_id"], "expected": r["_expected"], "correct": r["_correct"]}
                        for r in results
                    ]
                    for fmt_id, results in all_results.items()
                },
                "extraction_times": {
                    fmt_id: ext.get("_elapsed_seconds", 0)
                    for fmt_id, ext in extractions.items()
                },
            },
            f,
            indent=2,
        )
    print(f"\nFull results saved to {out_path}")

    # ---- Phase 5: Analysis ----
    print("\n" + "=" * 70)
    print("PHASE 5: ANALYSIS — Which format detected which hallucination?")
    print("=" * 70)

    for fmt_id in ["raw", "38a", "38b", "38c", "38d"]:
        print(f"\n--- {fmt_id} ---")
        for r in all_results[fmt_id]:
            test_id = r["_test_id"]
            if r["_expected"] == "hallucinated":
                if r["_correct"]:
                    issues = r.get("specific_issues", [])
                    issue_summary = "; ".join(
                        f"{i.get('claim_part', '?')}: {i.get('issue', '?')}"
                        for i in issues[:2]
                    )
                    print(f"  CAUGHT {test_id}: {issue_summary}")
                else:
                    print(
                        f"  MISSED {test_id}: verdict={r.get('verdict', '?')}, "
                        f"reasoning={r.get('reasoning', '?')[:100]}"
                    )


if __name__ == "__main__":
    main()
