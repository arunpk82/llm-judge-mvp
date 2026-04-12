"""
Experiment 31 — Stage 1: Multi-Pass Ensemble Extraction

5 focused extraction passes + 1 verification pass per source document.
Each pass asks a different question, producing complementary fact tables.

Pass 1: Entity profiles (who, attributes, descriptions)
Pass 2: Event timeline (what happened, chronologically)
Pass 3: Relationships (how entities connect, including NOT)
Pass 4: Numbers and dates (every quantity with context)
Pass 5: Negations and boundaries (what is NOT said/denied)
Pass 6: Verification (compare source against passes 1-5, find gaps)

Output: experiments/exp31_multipass_fact_tables.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

# ========================================================================
# 5 EXTRACTION PROMPTS — each asks a different question
# ========================================================================

PASS_PROMPTS = {
    "P1_entities": """You are an entity extraction engine. Read the source document and extract EVERY person, organization, place, and thing mentioned.

For EACH entity, provide:
- name: exact name as used in the document
- type: PERSON / ORG / PLACE / PRODUCT / OTHER
- aliases: other names used for the same entity (nicknames, pronouns, titles)
- attributes: dict of all stated properties (age, role, title, description, condition, status)

CRITICAL: Extract EVERY entity, even minor ones. If the document says "female companion", that is an entity — do NOT change it to "wife" or "girlfriend".

Return ONLY valid JSON: {"entities": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
    "P2_events": """You are an event extraction engine. Read the source document and extract EVERY event or action that occurs.

For EACH event, provide:
- who: the actor (use exact name from document)
- action: the verb/action (use exact wording from document)
- target: who/what was acted upon (if applicable)
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)
- sequence: approximate order (1, 2, 3...)

CRITICAL: Extract EVERY action, including passive ones ("was arrested", "was hospitalized"). Include both active and passive voice events. Do not skip minor events.

Return ONLY valid JSON: {"events": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
    "P3_relationships": """You are a relationship extraction engine. Read the source document and map ALL relationships between entities.

For EACH relationship, provide:
- entity1: first entity
- entity2: second entity
- relationship: type of connection (teammate, companion, employer, lawyer, victim, suspect, etc.)
- NOT_relationships: list of relationships that are explicitly NOT stated or would be INCORRECT
  (e.g., if document says "companion", then NOT_relationships: ["wife", "girlfriend"])

CRITICAL: Pay attention to the EXACT words used. If the document says "female companion", the relationship is "companion" — NOT "wife", NOT "girlfriend". List these incorrect alternatives in NOT_relationships.

Also extract:
- team memberships
- organizational affiliations
- family connections
- professional relationships
- adversarial relationships (attacker/victim, plaintiff/defendant)

Return ONLY valid JSON: {"relationships": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
    "P4_numbers": """You are a numerical fact extraction engine. Read the source document and extract EVERY number, date, quantity, and measurement.

For EACH numerical fact, provide:
- entity: who or what the number describes
- number: the exact value (as stated in document)
- unit: what the number measures (years, people, dollars, etc.)
- describes: what aspect this number refers to
- context: the sentence where this number appears

Also extract:
- Dates and times (exact as stated)
- Ages
- Counts of people or things
- Distances, weights, measurements
- Monetary amounts
- Percentages
- Ordinals (first, second, etc.)

CRITICAL: Capture the EXACT number as stated. "15 years" is 15, unit="years". "Three women" is 3, unit="people". "53-year-old" is 53, unit="age".

Return ONLY valid JSON: {"numerical_facts": [...], "temporal_facts": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
    "P5_negations": """You are a negation and boundary extraction engine. Read the source document and identify everything that is explicitly denied, negated, absent, or bounded.

Extract:
1. explicit_negations: Statements that use "not", "no", "never", "denied", "neither"
   - statement: what is negated
   - context: surrounding text
   
2. absent_information: Important information that is notably NOT provided
   - what: what information is missing
   - why_notable: why its absence matters
   (e.g., "suspect has not been named" — name is absent)

3. boundaries: Limits or qualifications on stated facts
   - fact: the stated fact
   - boundary: the limit or qualification
   (e.g., "only one plant was shut down" — boundary is "only one")

4. corrections: Where the document corrects a potential misconception
   - wrong: what might be incorrectly assumed
   - right: what the document actually says
   (e.g., might assume "wife" but document says "companion")

Return ONLY valid JSON: {"explicit_negations": [...], "absent_information": [...], "boundaries": [...], "corrections": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
}

VERIFY_PROMPT = """You are a fact verification engine. Below is a source document and 5 extractions from different angles. Your job is to find GAPS — facts in the source that were missed by ALL 5 extractions.

Source document:
\"\"\"{source}\"\"\"

Extraction 1 (Entities): {p1}
Extraction 2 (Events): {p2}
Extraction 3 (Relationships): {p3}
Extraction 4 (Numbers): {p4}
Extraction 5 (Negations): {p5}

List ANY facts from the source that are NOT captured in any of the 5 extractions above. For each missed fact, provide:
- fact: the missed information
- category: which extraction should have caught it (entities/events/relationships/numbers/negations)
- source_text: the relevant text from the source

Return ONLY valid JSON: {"missed_facts": [...]}
If nothing is missed, return: {"missed_facts": []}

JSON:"""


def call_gemini(
    prompt: str, case_id: str, pass_name: str, timeout: int = 90
) -> dict | None:
    """Call Gemini and parse JSON response."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    try:
        with httpx.Client(timeout=float(timeout)) as client:
            resp = client.post(
                url, json=payload, headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()

            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()

            return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"      {pass_name}: JSON parse error: {str(e)[:60]}")
        return None
    except Exception as e:
        print(f"      {pass_name}: error: {str(e)[:80]}")
        return None


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 31 — STAGE 1: Multi-Pass Ensemble Extraction")
    print("=" * 70)
    print("  5 extraction passes + 1 verification pass per source")
    print("  P1: Entities  P2: Events  P3: Relationships")
    print("  P4: Numbers   P5: Negations  P6: Verification")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))

    # Deduplicate sources (same source shared across multiple cases)
    source_map = {}
    for case in cases:
        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )
        source_hash = hash(source)
        if source_hash not in source_map:
            source_map[source_hash] = {"source": source, "case_ids": []}
        source_map[source_hash]["case_ids"].append(case.case_id)

    print(f"\n  Cases: {len(cases)}")
    print(f"  Unique sources: {len(source_map)} (deduplicating shared sources)")
    print(f"  Estimated calls: {len(source_map) * 6} ({len(source_map)} × 6 passes)")
    print(f"  Estimated cost: ~${len(source_map) * 0.006:.3f}")

    all_extractions = {}
    t0 = time.time()

    for si, (src_hash, src_data) in enumerate(source_map.items()):
        source = src_data["source"][:6000]
        case_ids = src_data["case_ids"]
        primary_id = case_ids[0]

        print(
            f"\n  [{si+1}/{len(source_map)}] Source for {', '.join(case_ids[:3])}{'...' if len(case_ids) > 3 else ''}"
        )

        passes = {}
        pass_times = {}

        # Run 5 extraction passes
        for pass_name, prompt_template in PASS_PROMPTS.items():
            t_start = time.time()
            prompt = prompt_template.replace("{source}", source)
            result = call_gemini(prompt, primary_id, pass_name)
            elapsed = time.time() - t_start
            pass_times[pass_name] = round(elapsed, 1)

            if result:
                passes[pass_name] = result
                # Count extracted items
                counts = {
                    k: len(v) if isinstance(v, list) else 1 for k, v in result.items()
                }
                print(f"    ✓ {pass_name}: {counts} ({elapsed:.1f}s)")
            else:
                passes[pass_name] = None
                print(f"    ✗ {pass_name}: FAILED ({elapsed:.1f}s)")

        # Run verification pass
        t_start = time.time()
        verify_prompt = (
            VERIFY_PROMPT.replace("{source}", source)
            .replace("{p1}", json.dumps(passes.get("P1_entities", {}))[:2000])
            .replace("{p2}", json.dumps(passes.get("P2_events", {}))[:2000])
            .replace("{p3}", json.dumps(passes.get("P3_relationships", {}))[:2000])
            .replace("{p4}", json.dumps(passes.get("P4_numbers", {}))[:2000])
            .replace("{p5}", json.dumps(passes.get("P5_negations", {}))[:2000])
        )
        verify_result = call_gemini(verify_prompt, primary_id, "P6_verify")
        verify_elapsed = time.time() - t_start
        pass_times["P6_verify"] = round(verify_elapsed, 1)

        n_missed = 0
        if verify_result:
            n_missed = len(verify_result.get("missed_facts", []))
            passes["P6_verify"] = verify_result
            print(
                f"    ✓ P6_verify: {n_missed} missed facts found ({verify_elapsed:.1f}s)"
            )
        else:
            passes["P6_verify"] = None
            print(f"    ✗ P6_verify: FAILED ({verify_elapsed:.1f}s)")

        # Store for all case_ids sharing this source
        for cid in case_ids:
            all_extractions[cid] = {
                "source_len": len(source),
                "pass_times": pass_times,
                "passes": passes,
                "n_missed_facts": n_missed,
            }

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*70}")

    n_complete = sum(
        1
        for v in all_extractions.values()
        if all(v["passes"].get(p) for p in PASS_PROMPTS)
    )
    n_verified = sum(
        1 for v in all_extractions.values() if v["passes"].get("P6_verify")
    )

    print(f"  Cases: {len(all_extractions)}")
    print(f"  All 5 passes complete: {n_complete}/{len(all_extractions)}")
    print(f"  Verification pass complete: {n_verified}/{len(all_extractions)}")

    # Aggregate stats per pass
    for pass_name in list(PASS_PROMPTS.keys()) + ["P6_verify"]:
        times = [
            v["pass_times"].get(pass_name, 0)
            for v in all_extractions.values()
            if v["pass_times"].get(pass_name)
        ]
        # Deduplicate (same source = same time)
        unique_times = list(
            {
                id(v): v["pass_times"].get(pass_name, 0)
                for v in all_extractions.values()
                if v["pass_times"].get(pass_name)
            }.values()
        )
        if unique_times:
            print(f"  {pass_name:16s}: avg {sum(unique_times)/len(unique_times):.1f}s")

    # Missed facts stats
    missed = [v["n_missed_facts"] for v in all_extractions.values()]
    print(
        f"\n  Missed facts found by verification: avg {sum(missed)/len(missed):.1f} per case"
    )

    # Save
    output_path = "experiments/exp31_multipass_fact_tables.json"
    with open(output_path, "w") as f:
        json.dump(all_extractions, f, indent=2)
    fsize = os.path.getsize(output_path) / 1024
    print(f"\nSaved: {output_path} ({fsize:.1f} KB)")
    print("\nRun Stage 2: poetry run python experiments/exp31_stage2_ensemble.py")


if __name__ == "__main__":
    main()
