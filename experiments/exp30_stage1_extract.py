"""
Experiment 30 — Stage 1: Extract Fact Tables

Calls Gemini ONCE per source document to extract structured fact tables.
Saves all 50 fact tables to a JSON file.

Stage 2 (comparison) runs separately against this file.
If extraction needs improvement, rerun only this stage.
If comparison logic needs improvement, rerun only Stage 2.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

_EXTRACT_PROMPT = """You are a fact extraction engine. Read the following source document and extract ALL factual information into a structured JSON format.

Extract:
1. entities: Every person, organization, place mentioned. Include name, type (PERSON/ORG/PLACE/OTHER), aliases (other names used for same entity), and key attributes as a dict (role, age, title, description, condition, etc).
2. relationships: How entities relate. Include entity1, entity2, relationship (e.g. "teammate", "wife", "companion", "lawyer", "employer").
3. numerical_facts: Every number with context. Include entity (who/what it describes), number, unit, describes (what the number measures).
4. events: What happened. Include who (actor), action (verb), target (who/what was acted upon), where, when.
5. temporal_facts: Dates, times, temporal references. Include entity, date, event (what happened).
6. negations: Facts explicitly stated as NOT true or denied. Include statement and context.

CRITICAL RULES:
- Extract ONLY what the document states. Do not infer or add information.
- If the document says "female companion", record exactly that — NOT "wife" or "girlfriend".
- If a name is not given, record as "unnamed" with the description.
- Capture exact numbers, dates, quantities as stated.
- Include ALL entities even minor ones.
- Return ONLY valid JSON. No markdown, no explanation, no preamble.

Source document:
\"\"\"
{source}
\"\"\"

JSON:"""


def extract_fact_table(source: str, case_id: str) -> dict | None:
    """Use Gemini to extract structured fact table from source."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = _EXTRACT_PROMPT.format(source=source[:6000])

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    try:
        with httpx.Client(timeout=60.0) as client:
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
        print(f"    {case_id}: JSON parse error: {str(e)[:60]}")
        print(f"    Raw (first 200): {raw[:200]}")
        return None
    except Exception as e:
        print(f"    {case_id}: Gemini error: {str(e)[:80]}")
        return None


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    print("=" * 70)
    print("EXPERIMENT 30 — STAGE 1: Extract Fact Tables")
    print("=" * 70)
    print("  Gemini extracts structured fact table from each source document.")
    print("  Output: experiments/exp30_fact_tables.json")
    print("=" * 70)

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))
    print(f"\nCases: {len(cases)}")

    fact_tables = {}
    errors = 0
    t0 = time.time()

    for ci, case in enumerate(cases):
        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )

        t_start = time.time()
        ft = extract_fact_table(source, case.case_id)
        t_elapsed = time.time() - t_start

        if ft:
            n_ent = len(ft.get("entities", []))
            n_evt = len(ft.get("events", []))
            n_num = len(ft.get("numerical_facts", []))
            n_rel = len(ft.get("relationships", []))
            n_neg = len(ft.get("negations", []))
            n_tmp = len(ft.get("temporal_facts", []))

            fact_tables[case.case_id] = {
                "source_len": len(source),
                "extraction_time_s": round(t_elapsed, 2),
                "fact_table": ft,
                "stats": {
                    "entities": n_ent,
                    "events": n_evt,
                    "numerical_facts": n_num,
                    "relationships": n_rel,
                    "negations": n_neg,
                    "temporal_facts": n_tmp,
                },
            }
            print(
                f"  [{ci+1:2d}/50] {case.case_id}: {n_ent} entities, {n_evt} events, "
                f"{n_num} numbers, {n_rel} rels, {n_neg} negations ({t_elapsed:.1f}s)"
            )
        else:
            errors += 1
            fact_tables[case.case_id] = {
                "source_len": len(source),
                "extraction_time_s": round(t_elapsed, 2),
                "fact_table": None,
                "stats": None,
                "error": True,
            }
            print(f"  [{ci+1:2d}/50] {case.case_id}: FAILED ({t_elapsed:.1f}s)")

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE — {elapsed:.0f}s total")
    print(f"{'='*70}")
    print(f"  Successful: {len(cases) - errors} / {len(cases)}")
    print(f"  Failed: {errors}")

    # Aggregate stats
    all_stats = [v["stats"] for v in fact_tables.values() if v.get("stats")]
    if all_stats:
        print("\n  Aggregate extraction stats:")
        for key in [
            "entities",
            "events",
            "numerical_facts",
            "relationships",
            "negations",
            "temporal_facts",
        ]:
            vals = [s[key] for s in all_stats]
            print(
                f"    {key:20s}: avg={sum(vals)/len(vals):.1f}, min={min(vals)}, max={max(vals)}, total={sum(vals)}"
            )

    # Average extraction time
    times = [v["extraction_time_s"] for v in fact_tables.values()]
    print(
        f"\n  Extraction time: avg={sum(times)/len(times):.1f}s, total={sum(times):.0f}s"
    )
    print(f"  Estimated cost: ~${len(cases) * 0.001:.3f}")

    # Save
    output_path = "experiments/exp30_fact_tables.json"
    with open(output_path, "w") as f:
        json.dump(fact_tables, f, indent=2)
    print(f"\nSaved: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print("\nRun Stage 2: poetry run python experiments/exp30_stage2_compare.py")


if __name__ == "__main__":
    main()
