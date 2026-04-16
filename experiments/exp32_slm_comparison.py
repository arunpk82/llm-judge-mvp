"""
Experiment 32 — SLM Comparison for Knowledge Graph Extraction

Tests multiple SLMs via hosted APIs against the Gemini 2.5 Flash baseline.
Uses the SAME 5 extraction prompts from Exp 31.
Compares graph completeness, JSON reliability, latency, and cost.

Providers:
  - Google AI Studio: Gemma 4 E4B, Gemma 4 26B MoE (free tier)
  - Groq:             Llama 3.3 70B, Qwen 2.5 32B (free tier, OpenAI-compatible)
  - Baseline:         Gemini 2.5 Flash (existing results from exp31)

Usage:
    # Set API keys
    export GEMINI_API_KEY=your_key           # Google AI Studio (Gemma + Gemini)
    export GROQ_API_KEY=your_key             # Groq (Llama, Qwen)

    # Run full comparison (all models, all sources)
    poetry run python experiments/exp32_slm_comparison.py

    # Run single model for testing
    poetry run python experiments/exp32_slm_comparison.py --models gemma-4-e4b

    # Run on fewer sources for quick test
    poetry run python experiments/exp32_slm_comparison.py --max-sources 2

Output: experiments/exp32_slm_comparison_results.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ========================================================================
# 5 EXTRACTION PROMPTS — identical to Exp 31
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

3. boundaries: Limits or qualifications on stated facts
   - fact: the stated fact
   - boundary: the limit or qualification

4. corrections: Where the document corrects a potential misconception
   - wrong: what might be incorrectly assumed
   - right: what the document actually says

Return ONLY valid JSON: {"explicit_negations": [...], "absent_information": [...], "boundaries": [...], "corrections": [...]}

Source:
\"\"\"{source}\"\"\"

JSON:""",
}

# ========================================================================
# MODEL DEFINITIONS — hosted API configurations
# ========================================================================

MODELS = {
    # Google AI Studio (Gemini API) — free tier
    "gemma-4-31b": {
        "provider": "google",
        "model_id": "gemma-4-31b-it",
        "display_name": "Gemma 4 31B Dense",
        "env_key": "GEMINI_API_KEY",
        "cost_per_1m_input": 0.0,  # Free via AI Studio
        "cost_per_1m_output": 0.0,
    },
    "gemma-4-26b-moe": {
        "provider": "google",
        "model_id": "gemma-4-26b-a4b-it",
        "display_name": "Gemma 4 26B MoE (3.8B active)",
        "env_key": "GEMINI_API_KEY",
        "cost_per_1m_input": 0.0,
        "cost_per_1m_output": 0.0,
    },
    # Groq — free tier, OpenAI-compatible
    "llama-3.3-70b": {
        "provider": "groq",
        "model_id": "llama-3.3-70b-versatile",
        "display_name": "Llama 3.3 70B (Groq)",
        "env_key": "GROQ_API_KEY",
        "cost_per_1m_input": 0.59,
        "cost_per_1m_output": 0.79,
    },
    "qwen3-32b": {
        "provider": "groq",
        "model_id": "qwen/qwen3-32b",
        "display_name": "Qwen3 32B (Groq)",
        "env_key": "GROQ_API_KEY",
        "cost_per_1m_input": 0.29,
        "cost_per_1m_output": 0.39,
    },
    # Baseline — existing Gemini 2.5 Flash results (no API call needed)
    "gemini-baseline": {
        "provider": "baseline",
        "model_id": "gemini-2.5-flash",
        "display_name": "Gemini 2.5 Flash (BASELINE)",
        "env_key": "GEMINI_API_KEY",
        "cost_per_1m_input": 0.15,
        "cost_per_1m_output": 0.60,
    },
}


# ========================================================================
# API CALLERS — one per provider
# ========================================================================

def _strip_markdown_fences(raw: str) -> str:
    """Remove ```json ... ``` fences from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()
    # Also handle ```json prefix
    if raw.startswith("json"):
        raw = raw[4:].strip()
    return raw


def call_google(prompt: str, model_id: str, timeout: int = 300) -> tuple[dict | None, float]:
    """Call Google AI Studio API (Gemini + Gemma models)."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, 0.0

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=float(timeout)) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"]
            raw = _strip_markdown_fences(raw)
            result = json.loads(raw)
            # Handle list returns (some models return list instead of dict)
            if isinstance(result, list):
                result = {"entities": result}  # best guess
            return result, time.time() - t0
    except json.JSONDecodeError as e:
        print(f"        JSON parse error: {str(e)[:60]}")
        return None, time.time() - t0
    except Exception as e:
        print(f"        API error: {str(e)[:80]}")
        return None, time.time() - t0


def call_groq(prompt: str, model_id: str, timeout: int = 300) -> tuple[dict | None, float]:
    """Call Groq API (OpenAI-compatible)."""
    import httpx

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, 0.0

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    t0 = time.time()
    try:
        with httpx.Client(timeout=float(timeout)) as client:
            resp = client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            raw = _strip_markdown_fences(raw)
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


def call_model(prompt: str, model_key: str, timeout: int = 300) -> tuple[dict | None, float]:
    """Route to the correct API caller based on provider."""
    model_def = MODELS[model_key]
    provider = model_def["provider"]
    model_id = model_def["model_id"]

    if provider == "google":
        return call_google(prompt, model_id, timeout)
    elif provider == "groq":
        return call_groq(prompt, model_id, timeout)
    else:
        return None, 0.0


# ========================================================================
# GRAPH COMPLETENESS METRICS
# ========================================================================

def _safe_len(data: dict | None, key: str) -> int:
    """Safely count items in an extraction result."""
    if not data:
        return 0
    val = data.get(key, [])
    if isinstance(val, list):
        return len(val)
    return 0


def measure_extraction(passes: dict) -> dict:
    """Measure completeness metrics from 5 extraction passes."""
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
        "passes_total": 5,
    }


# ========================================================================
# CROSS-VALIDATION CHECKS
# ========================================================================

def cross_validate(passes: dict) -> dict:
    """Cross-validate consistency across 5 extraction passes.

    Checks:
    - Every P2 actor should exist in P1 entities
    - Every P3 entity should exist in P1 entities
    - Every P4 entity should exist in P1 entities
    - P2 event counts should be consistent with P4 numbers
    """
    p1 = passes.get("P1_entities")
    p2 = passes.get("P2_events")
    p3 = passes.get("P3_relationships")
    p4 = passes.get("P4_numbers")
    p5 = passes.get("P5_negations")  # noqa: F841

    gaps = []
    stats = {
        "p2_actors_in_p1": 0, "p2_actors_total": 0,
        "p3_entities_in_p1": 0, "p3_entities_total": 0,
        "p4_entities_in_p1": 0, "p4_entities_total": 0,
    }

    if not p1:
        return {"gaps": ["P1 entities missing — cannot cross-validate"], "stats": stats}

    # Build P1 entity set (lowercased, with aliases)
    p1_entities = set()
    p1_aliases = {}
    for ent in (p1.get("entities") or []):
        name = (ent.get("name") or "").lower().strip()
        if name:
            p1_entities.add(name)
            for alias in (ent.get("aliases") or []):
                al = alias.lower().strip() if isinstance(alias, str) else ""
                if al:
                    p1_entities.add(al)
                    p1_aliases[al] = name

    def _entity_found(name: str) -> bool:
        nl = name.lower().strip()
        if nl in p1_entities:
            return True
        # Partial match: any P1 entity contains this name or vice versa
        return any(nl in e or e in nl for e in p1_entities if len(nl) > 2 and len(e) > 2)

    # Check 1: P2 actors in P1
    if p2:
        for evt in (p2.get("events") or []):
            who = evt.get("who", "")
            actors = [who] if isinstance(who, str) else (who if isinstance(who, list) else [])
            for actor in actors:
                if not actor or not actor.strip():
                    continue
                stats["p2_actors_total"] += 1
                if _entity_found(actor):
                    stats["p2_actors_in_p1"] += 1
                else:
                    gaps.append(f"P2→P1: Actor '{actor}' not in P1 entities")

    # Check 2: P3 entities in P1
    if p3:
        for rel in (p3.get("relationships") or []):
            for key in ["entity1", "entity2"]:
                ent = (rel.get(key) or "").strip()
                if not ent:
                    continue
                stats["p3_entities_total"] += 1
                if _entity_found(ent):
                    stats["p3_entities_in_p1"] += 1
                else:
                    gaps.append(f"P3→P1: Entity '{ent}' not in P1 entities")

    # Check 3: P4 entities in P1
    if p4:
        for nf in (p4.get("numerical_facts") or []):
            ent = (nf.get("entity") or "").strip()
            if not ent:
                continue
            stats["p4_entities_in_p1"] += 1
            if _entity_found(ent):
                stats["p4_entities_in_p1"] += 1
            else:
                gaps.append(f"P4→P1: Entity '{ent}' not in P1 entities")

    # Compute resolution rates
    stats["p2_resolution_rate"] = (
        round(stats["p2_actors_in_p1"] / stats["p2_actors_total"], 3)
        if stats["p2_actors_total"] > 0 else None
    )
    stats["p3_resolution_rate"] = (
        round(stats["p3_entities_in_p1"] / stats["p3_entities_total"], 3)
        if stats["p3_entities_total"] > 0 else None
    )

    return {
        "gaps": gaps[:20],  # Cap for readability
        "n_gaps": len(gaps),
        "stats": stats,
    }


# ========================================================================
# MAIN EXPERIMENT
# ========================================================================

@dataclass
class SourceResult:
    source_idx: int
    source_len: int
    case_ids: list[str]
    passes: dict = field(default_factory=dict)
    pass_times: dict = field(default_factory=dict)
    pass_failures: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    cross_validation: dict = field(default_factory=dict)
    total_time_s: float = 0.0


@dataclass
class ModelResult:
    model_key: str
    display_name: str
    provider: str
    sources: list[SourceResult] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)


def load_baseline(baseline_path: str) -> dict:
    """Load existing Gemini baseline from exp31_multipass_fact_tables.json."""
    if not os.path.exists(baseline_path):
        print(f"  WARNING: Baseline file not found: {baseline_path}")
        return {}
    with open(baseline_path) as f:
        return json.load(f)


def run_model_on_sources(
    model_key: str,
    sources: list[dict],
    baseline_data: dict | None = None,
) -> ModelResult:
    """Run a single model on all sources and collect metrics."""
    model_def = MODELS[model_key]
    result = ModelResult(
        model_key=model_key,
        display_name=model_def["display_name"],
        provider=model_def["provider"],
    )

    # Check API key availability
    env_key = model_def["env_key"]
    if model_def["provider"] != "baseline" and not os.environ.get(env_key):
        print(f"\n  SKIP {model_key}: {env_key} not set")
        return result

    print(f"\n{'='*60}")
    print(f"  MODEL: {model_def['display_name']}")
    print(f"  Provider: {model_def['provider']} | Model ID: {model_def['model_id']}")
    print(f"{'='*60}")

    for si, src_data in enumerate(sources):
        source_text = src_data["source"][:6000]
        case_ids = src_data["case_ids"]
        primary_id = case_ids[0]

        print(f"\n  [{si+1}/{len(sources)}] Source for {', '.join(case_ids[:3])}"
              f"{'...' if len(case_ids) > 3 else ''} ({len(source_text)} chars)")

        sr = SourceResult(
            source_idx=si,
            source_len=len(source_text),
            case_ids=case_ids,
        )

        # For baseline, load from existing data
        if model_def["provider"] == "baseline":
            if baseline_data and primary_id in baseline_data:
                bl = baseline_data[primary_id]
                sr.passes = bl.get("passes", {})
                sr.pass_times = bl.get("pass_times", {})
                sr.total_time_s = sum(sr.pass_times.values())
                sr.metrics = measure_extraction(sr.passes)
                sr.cross_validation = cross_validate(sr.passes)
                print(f"    ✓ Loaded from baseline ({sr.metrics['total_items']} items)")
            else:
                print(f"    ✗ Not found in baseline for {primary_id}")
                sr.pass_failures.append("baseline_missing")
            result.sources.append(sr)
            continue

        # Run 5 extraction passes
        t_source_start = time.time()
        for pass_name, prompt_template in PASS_PROMPTS.items():
            prompt = prompt_template.replace("{source}", source_text)

            # Rate limiting for Groq free tier (30 RPM)
            if model_def["provider"] == "groq":
                time.sleep(2.5)

            data, elapsed = call_model(prompt, model_key)
            sr.pass_times[pass_name] = round(elapsed, 2)

            if data:
                sr.passes[pass_name] = data
                # Count extracted items
                counts = {}
                for k, v in data.items():
                    counts[k] = len(v) if isinstance(v, list) else 1
                print(f"    ✓ {pass_name}: {counts} ({elapsed:.1f}s)")
            else:
                sr.passes[pass_name] = None
                sr.pass_failures.append(pass_name)
                print(f"    ✗ {pass_name}: FAILED ({elapsed:.1f}s)")

        sr.total_time_s = time.time() - t_source_start
        sr.metrics = measure_extraction(sr.passes)
        sr.cross_validation = cross_validate(sr.passes)
        result.sources.append(sr)

        # Cross-validation summary
        cv = sr.cross_validation
        n_gaps = cv.get("n_gaps", 0)
        p2_rate = cv.get("stats", {}).get("p2_resolution_rate")
        print(f"    → {sr.metrics['total_items']} total items, "
              f"{sr.metrics['passes_ok']}/5 passes OK, "
              f"{n_gaps} cross-val gaps"
              f"{f', P2→P1 resolution: {p2_rate:.0%}' if p2_rate is not None else ''}")

    # Compute aggregate metrics
    if result.sources:
        all_metrics = [s.metrics for s in result.sources if s.metrics]
        if all_metrics:
            result.aggregate = {
                "avg_entities": round(sum(m["entities"] for m in all_metrics) / len(all_metrics), 1),
                "avg_events": round(sum(m["events"] for m in all_metrics) / len(all_metrics), 1),
                "avg_relationships": round(sum(m["relationships"] for m in all_metrics) / len(all_metrics), 1),
                "avg_numerical": round(sum(m["numerical_facts"] for m in all_metrics) / len(all_metrics), 1),
                "avg_total_items": round(sum(m["total_items"] for m in all_metrics) / len(all_metrics), 1),
                "avg_passes_ok": round(sum(m["passes_ok"] for m in all_metrics) / len(all_metrics), 2),
                "total_time_s": round(sum(s.total_time_s for s in result.sources), 1),
                "avg_time_per_source_s": round(
                    sum(s.total_time_s for s in result.sources) / len(result.sources), 1
                ),
                "n_failures": sum(len(s.pass_failures) for s in result.sources),
                "json_parse_success_rate": round(
                    sum(m["passes_ok"] for m in all_metrics)
                    / sum(m["passes_total"] for m in all_metrics),
                    3,
                ),
                # Cross-validation averages
                "avg_cross_val_gaps": round(
                    sum(s.cross_validation.get("n_gaps", 0) for s in result.sources)
                    / len(result.sources), 1
                ),
            }

    return result


def print_comparison_table(results: list[ModelResult], baseline_key: str = "gemini-baseline"):
    """Print a comparison table of all models vs baseline."""
    baseline = next((r for r in results if r.model_key == baseline_key), None)
    baseline_total = baseline.aggregate.get("avg_total_items", 1) if baseline else 1

    print(f"\n{'='*90}")
    print("COMPARISON TABLE — SLM vs Gemini Baseline")
    print(f"{'='*90}")

    header = (
        f"{'Model':<30s} {'Items':>6s} {'vs BL':>6s} "
        f"{'Ent':>5s} {'Evt':>5s} {'Rel':>5s} {'Num':>5s} "
        f"{'JSON%':>6s} {'Time':>6s} {'Gaps':>5s}"
    )
    print(header)
    print("-" * 90)

    for r in results:
        a = r.aggregate
        if not a:
            print(f"  {r.display_name:<28s} — NO DATA (API key missing?)")
            continue

        vs_baseline = round(a["avg_total_items"] / baseline_total * 100, 1) if baseline_total > 0 else 0
        is_baseline = r.model_key == baseline_key

        row = (
            f"{'→ ' if is_baseline else '  '}"
            f"{r.display_name:<28s} "
            f"{a['avg_total_items']:>6.1f} "
            f"{'BASE' if is_baseline else f'{vs_baseline:5.1f}%':>6s} "
            f"{a['avg_entities']:>5.1f} "
            f"{a['avg_events']:>5.1f} "
            f"{a['avg_relationships']:>5.1f} "
            f"{a['avg_numerical']:>5.1f} "
            f"{a['json_parse_success_rate']*100:>5.1f}% "
            f"{a['avg_time_per_source_s']:>5.1f}s "
            f"{a['avg_cross_val_gaps']:>5.1f}"
        )
        print(row)

    print("-" * 90)

    # Decision criteria
    print(f"\nDECISION CRITERIA:")
    print(f"  ≥90% of baseline items → VIABLE replacement ($0.00/source)")
    print(f"  80-90% of baseline      → PARTIAL (use SLM for P1-P4, Gemini for P5+P6)")
    print(f"  <80% of baseline        → NOT VIABLE (keep Gemini)")
    print(f"  JSON success rate <90%   → UNRELIABLE (needs retry logic)")

    for r in results:
        if r.model_key == baseline_key or not r.aggregate:
            continue
        a = r.aggregate
        vs = round(a["avg_total_items"] / baseline_total * 100, 1)
        json_ok = a["json_parse_success_rate"] * 100

        if vs >= 90 and json_ok >= 90:
            verdict = "✅ VIABLE"
        elif vs >= 80 and json_ok >= 80:
            verdict = "⚠️  PARTIAL"
        else:
            verdict = "❌ NOT VIABLE"

        print(f"  {r.display_name}: {vs:.1f}% completeness, "
              f"{json_ok:.1f}% JSON reliability → {verdict}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Exp 32: SLM Comparison for KG Extraction")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Specific models to test (default: all available)")
    parser.add_argument("--max-sources", type=int, default=None,
                        help="Limit number of sources (for quick testing)")
    parser.add_argument("--baseline-path", default="experiments/exp31_multipass_fact_tables.json",
                        help="Path to Gemini baseline results")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 32 — SLM Comparison for Knowledge Graph Extraction")
    print("=" * 70)
    print("  Same 5 extraction prompts as Exp 31.")
    print("  Compare SLM completeness vs Gemini 2.5 Flash baseline.")
    print("=" * 70)

    # Check available API keys
    available_keys = {}
    for key in ["GEMINI_API_KEY", "GROQ_API_KEY"]:
        available_keys[key] = bool(os.environ.get(key))
        status = "✓ SET" if available_keys[key] else "✗ NOT SET"
        print(f"  {key}: {status}")

    # Determine which models to test
    if args.models:
        model_keys = args.models
    else:
        model_keys = []
        # Always include baseline
        model_keys.append("gemini-baseline")
        # Add models whose API keys are available
        for mk, md in MODELS.items():
            if mk == "gemini-baseline":
                continue
            if available_keys.get(md["env_key"], False):
                model_keys.append(mk)

    print(f"\n  Models to test: {', '.join(model_keys)}")

    # Load RAGTruth sources
    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))

    # Deduplicate sources
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

    print(f"  Cases: {len(cases)}")
    print(f"  Unique sources: {len(sources)}")
    print(f"  Estimated API calls: {len(sources) * 5} per model "
          f"({len(model_keys) - 1} models × {len(sources)} sources × 5 passes)")

    # Load baseline data
    baseline_data = load_baseline(args.baseline_path)
    if baseline_data:
        print(f"  Baseline loaded: {len(baseline_data)} cases from {args.baseline_path}")
    else:
        print(f"  WARNING: No baseline data — comparison will be limited")

    # Run each model
    t0 = time.time()
    all_results = []

    for mk in model_keys:
        model_result = run_model_on_sources(mk, sources, baseline_data)
        all_results.append(model_result)

    elapsed = time.time() - t0

    # Print comparison table
    print_comparison_table(all_results)

    # Print cross-validation detail for best SLM
    non_baseline = [r for r in all_results if r.model_key != "gemini-baseline" and r.aggregate]
    if non_baseline:
        best = max(non_baseline, key=lambda r: r.aggregate.get("avg_total_items", 0))
        print(f"\n{'='*70}")
        print(f"CROSS-VALIDATION DETAIL — {best.display_name}")
        print(f"{'='*70}")
        for sr in best.sources:
            cv = sr.cross_validation
            n_gaps = cv.get("n_gaps", 0)
            if n_gaps > 0:
                print(f"\n  Source {sr.source_idx} ({sr.case_ids[0]}): {n_gaps} gaps")
                for gap in cv.get("gaps", [])[:5]:
                    print(f"    • {gap}")

    # Save results
    output = {
        "experiment": "exp32_slm_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1),
        "n_sources": len(sources),
        "models_tested": model_keys,
        "results": [],
    }

    for r in all_results:
        model_data = {
            "model_key": r.model_key,
            "display_name": r.display_name,
            "provider": r.provider,
            "aggregate": r.aggregate,
            "sources": [
                {
                    "source_idx": s.source_idx,
                    "source_len": s.source_len,
                    "case_ids": s.case_ids,
                    "metrics": s.metrics,
                    "cross_validation": s.cross_validation,
                    "pass_times": s.pass_times,
                    "pass_failures": s.pass_failures,
                    "total_time_s": s.total_time_s,
                    # Don't save raw passes (too large) — only metrics
                }
                for s in r.sources
            ],
        }
        output["results"].append(model_data)

    output_path = "experiments/exp32_slm_comparison_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    fsize = os.path.getsize(output_path) / 1024
    print(f"\nSaved: {output_path} ({fsize:.1f} KB)")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
