"""
Exp 31 Stage 1 Retry: Fix 2 failed passes
- Source 4 (ragtruth_66): P1_entities JSON parse error → retry
- Source 8 (ragtruth_180): P2_events timeout → retry with 120s
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

RETRY_PROMPTS = {
    "ragtruth_180": {
        "pass": "P2_events",
        "prompt": """You are an event extraction engine. Read the source document and extract EVERY event or action that occurs.

For EACH event, provide:
- who: the actor (use exact name from document)
- action: the verb/action (use exact wording)
- target: who/what was acted upon (if applicable)
- where: location (if stated)
- when: time/date (if stated)
- sequence: approximate order (1, 2, 3...)

CRITICAL: Extract EVERY action including passive ones. Do not skip minor events.

Return ONLY valid JSON with no markdown. The response must start with a curly brace.

Source:
\"\"\"{source}\"\"\"

JSON:""",
    }
}


def call_gemini(prompt, case_id, pass_name, timeout=120):
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
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
    except Exception as e:
        print(f"    {case_id} {pass_name}: {str(e)[:80]}")
        return None


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    ft_path = "experiments/exp31_multipass_fact_tables.json"
    with open(ft_path) as f:
        all_data = json.load(f)

    adapter = RAGTruthAdapter()
    cases = {c.case_id: c for c in adapter.load_cases(max_cases=50)}

    # Find which case_ids share the same source as each retry target
    source_groups = {}
    for retry_id in RETRY_PROMPTS:
        case = cases[retry_id]
        source = "\n".join(case.request.source_context or [])
        matching = [
            cid
            for cid, c in cases.items()
            if "\n".join(c.request.source_context or []) == source
        ]
        source_groups[retry_id] = matching

    for retry_id, info in RETRY_PROMPTS.items():
        case = cases[retry_id]
        source = "\n".join(case.request.source_context or [])[:6000]
        pass_name = info["pass"]
        prompt = info["prompt"].replace("{source}", source)
        affected = source_groups[retry_id]

        print(
            f"Retrying {retry_id} {pass_name} (affects {len(affected)} cases)...",
            end=" ",
            flush=True,
        )
        t0 = time.time()
        result = call_gemini(prompt, retry_id, pass_name, timeout=180)
        elapsed = time.time() - t0

        if result:
            # Handle case where Gemini returns a list instead of dict
            if isinstance(result, list):
                result = (
                    {"entities": result}
                    if pass_name == "P1_entities"
                    else {"events": result}
                )
            counts = {
                k: len(v) if isinstance(v, list) else 1 for k, v in result.items()
            }
            print(f"OK {counts} ({elapsed:.1f}s)")
            for cid in affected:
                if cid in all_data:
                    all_data[cid]["passes"][pass_name] = result
                    all_data[cid]["pass_times"][pass_name] = round(elapsed, 2)
        else:
            print(f"FAILED ({elapsed:.1f}s)")

    # Check completeness
    n_complete = sum(
        1
        for v in all_data.values()
        if all(
            v["passes"].get(f"P{i}")
            for i in [
                "1_entities",
                "2_events",
                "3_relationships",
                "4_numbers",
                "5_negations",
            ]
        )
    )
    print(f"\nAll 5 passes complete: {n_complete}/{len(all_data)}")

    with open(ft_path, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Updated: {ft_path}")


if __name__ == "__main__":
    main()
