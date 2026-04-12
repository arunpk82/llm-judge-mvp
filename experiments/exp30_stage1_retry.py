"""
Experiment 30 — Stage 1 RETRY: Extract failed fact tables

Retries the 9 failed cases with 120s timeout (was 60s).
Updates the existing exp30_fact_tables.json in place.
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
2. relationships: How entities relate. Include entity1, entity2, relationship (e.g. "teammate", "wife", "companion", "lawyer", "employer"). Include NOT field listing relationships explicitly NOT stated.
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
        with httpx.Client(timeout=120.0) as client:  # 120s timeout (was 60s)
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
        return None
    except Exception as e:
        print(f"    {case_id}: Gemini error: {str(e)[:80]}")
        return None


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    fact_tables_path = "experiments/exp30_fact_tables.json"
    if not os.path.exists(fact_tables_path):
        print(f"ERROR: {fact_tables_path} not found. Run Stage 1 first.")
        sys.exit(1)

    with open(fact_tables_path) as f:
        fact_tables = json.load(f)

    failed = [cid for cid, v in fact_tables.items() if not v.get("fact_table")]
    print(f"Retrying {len(failed)} failed cases with 120s timeout: {', '.join(failed)}")

    adapter = RAGTruthAdapter()
    cases = {c.case_id: c for c in adapter.load_cases(max_cases=50)}

    fixed = 0
    for cid in failed:
        case = cases.get(cid)
        if not case:
            print(f"  {cid}: case not found in adapter, skipping")
            continue

        source = (
            "\n".join(case.request.source_context or [])
            if case.request.source_context
            else ""
        )
        print(
            f"  {cid}: extracting (source={len(source)} chars)...", end=" ", flush=True
        )

        t_start = time.time()
        ft = extract_fact_table(source, cid)
        t_elapsed = time.time() - t_start

        if ft:
            n_ent = len(ft.get("entities", []))
            n_evt = len(ft.get("events", []))
            n_num = len(ft.get("numerical_facts", []))
            n_rel = len(ft.get("relationships", []))
            n_neg = len(ft.get("negations", []))

            fact_tables[cid] = {
                "source_len": len(source),
                "extraction_time_s": round(t_elapsed, 2),
                "fact_table": ft,
                "stats": {
                    "entities": n_ent,
                    "events": n_evt,
                    "numerical_facts": n_num,
                    "relationships": n_rel,
                    "negations": n_neg,
                    "temporal_facts": len(ft.get("temporal_facts", [])),
                },
            }
            fixed += 1
            print(f"OK ({n_ent} entities, {n_evt} events, {t_elapsed:.1f}s)")
        else:
            print(f"FAILED again ({t_elapsed:.1f}s)")

    # Save updated file
    with open(fact_tables_path, "w") as f:
        json.dump(fact_tables, f, indent=2)

    n_valid = sum(1 for v in fact_tables.values() if v.get("fact_table"))
    print(f"\nFixed: {fixed}/{len(failed)}")
    print(f"Total valid: {n_valid}/50")
    print(f"Updated: {fact_tables_path}")
    print("\nRerun Stage 2: poetry run python experiments/exp30_stage2_compare.py")


if __name__ == "__main__":
    main()
