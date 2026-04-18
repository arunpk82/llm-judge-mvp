"""
L2 Knowledge Graph Extraction — Multi-Pass Fact Table Generation.

Extracts structured fact tables from source documents using 5 focused
Gemini passes + 1 verification pass. Each pass asks a different question
(entities, events, relationships, numbers, negations) producing
complementary knowledge graphs.

Used by the benchmark prerequisite system to provision the L2 graph
cache before running evaluations. Ensures all benchmark sources have
cached fact tables so L2 never silently skips.

Design:
  - 6 API calls per unique source (~$0.006/source at Flash pricing)
  - Sources are deduplicated via SHA-256 hash (50 cases → 9 sources)
  - Extracted tables are stored in GraphCache for runtime lookups
  - Extraction happens once at registration time, never in the hot path

See also: ADR-0018, ADR-0025, graph_cache.py, hallucination_graphs.py.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# =====================================================================
# 5 Extraction Prompts + 1 Verification
# =====================================================================

PASS_PROMPTS: dict[str, str] = {
    "P1_entities": (
        "You are an entity extraction engine. Read the source document "
        "and extract EVERY person, organization, place, and thing mentioned.\n\n"
        "For EACH entity, provide:\n"
        "- name: exact name as used in the document\n"
        "- type: PERSON / ORG / PLACE / PRODUCT / OTHER\n"
        "- aliases: other names used for the same entity\n"
        "- attributes: dict of all stated properties\n\n"
        "CRITICAL: Extract EVERY entity, even minor ones. Use exact wording "
        "from the document.\n\n"
        'Return ONLY valid JSON: {{"entities": [...]}}\n\n'
        'Source:\n"""{source}"""\n\nJSON:'
    ),
    "P2_events": (
        "You are an event extraction engine. Read the source document "
        "and extract EVERY event or action that occurs.\n\n"
        "For EACH event, provide:\n"
        "- who: the actor (exact name)\n"
        "- action: the verb/action (exact wording)\n"
        "- target: who/what was acted upon\n"
        "- where: location (if stated)\n"
        "- when: time/date (if stated)\n"
        "- result: outcome (if stated)\n"
        "- sequence: approximate order\n\n"
        "CRITICAL: Extract EVERY action, including passive ones. "
        "Do not skip minor events.\n\n"
        'Return ONLY valid JSON: {{"events": [...]}}\n\n'
        'Source:\n"""{source}"""\n\nJSON:'
    ),
    "P3_relationships": (
        "You are a relationship extraction engine. Read the source document "
        "and map ALL relationships between entities.\n\n"
        "For EACH relationship, provide:\n"
        "- entity1: first entity\n"
        "- entity2: second entity\n"
        "- relationship: type of connection\n"
        "- NOT_relationships: list of relationships that would be INCORRECT\n\n"
        "CRITICAL: Pay attention to EXACT words. If document says "
        '"companion", the relationship is "companion" — NOT "wife".\n\n'
        'Return ONLY valid JSON: {{"relationships": [...]}}\n\n'
        'Source:\n"""{source}"""\n\nJSON:'
    ),
    "P4_numbers": (
        "You are a numerical fact extraction engine. Extract EVERY number, "
        "date, quantity, and measurement from the source document.\n\n"
        "For EACH fact, provide:\n"
        "- entity: who/what the number describes\n"
        "- number: the exact value\n"
        "- unit: what it measures\n"
        "- describes: what aspect\n"
        "- context: the sentence where it appears\n\n"
        "CRITICAL: Capture the EXACT number as stated.\n\n"
        'Return ONLY valid JSON: {{"numerical_facts": [...], "temporal_facts": [...]}}\n\n'
        'Source:\n"""{source}"""\n\nJSON:'
    ),
    "P5_negations": (
        "You are a negation and boundary extraction engine. Identify "
        "everything explicitly denied, negated, absent, or bounded.\n\n"
        "Extract:\n"
        '1. explicit_negations: Statements using "not", "no", "never"\n'
        "2. absent_information: Important info NOT provided\n"
        "3. boundaries: Limits or qualifications on stated facts\n"
        "4. corrections: Where document corrects potential misconceptions\n\n"
        "Return ONLY valid JSON: "
        '{{"explicit_negations": [...], "absent_information": [...], '
        '"boundaries": [...], "corrections": [...]}}\n\n'
        'Source:\n"""{source}"""\n\nJSON:'
    ),
}

VERIFY_PROMPT = (
    "You are a fact verification engine. Below is a source document and "
    "5 extractions from different angles. Find GAPS — facts in the source "
    "missed by ALL 5 extractions.\n\n"
    'Source document:\n"""{source}"""\n\n'
    "Extraction 1 (Entities): {p1}\n"
    "Extraction 2 (Events): {p2}\n"
    "Extraction 3 (Relationships): {p3}\n"
    "Extraction 4 (Numbers): {p4}\n"
    "Extraction 5 (Negations): {p5}\n\n"
    "List ANY facts from the source NOT captured in any extraction. "
    "For each missed fact, provide:\n"
    "- fact: the missed information\n"
    "- category: which extraction should have caught it\n"
    "- source_text: relevant text from source\n\n"
    'Return ONLY valid JSON: {{"missed_facts": []}}\n\nJSON:'
)


# =====================================================================
# Gemini API Call
# =====================================================================


def _call_gemini(
    prompt: str,
    *,
    model: str | None = None,
    timeout: int = 90,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """Call Gemini API and parse JSON response.

    Returns parsed JSON dict or None on failure.
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=float(timeout)) as client:
                resp = client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )
                resp.raise_for_status()
                data = resp.json()
                raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()

                # Strip markdown code fences
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    raw = raw.strip()

                return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("kg_extraction.json_parse_error", extra={"attempt": attempt})
        except Exception as e:
            logger.warning(
                "kg_extraction.api_error",
                extra={"attempt": attempt, "error": str(e)[:80]},
            )
            if attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))

    return None


# =====================================================================
# Multi-Pass Extraction
# =====================================================================


def extract_fact_tables(
    source_text: str,
    *,
    model: str | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Run 6-pass extraction on a source document.

    Args:
        source_text: Full source document text.
        model: Gemini model name (default from GEMINI_MODEL env).
        label: Human-readable label for logging (e.g. source_id).

    Returns:
        Dict with ``passes`` key containing P1-P6 extraction results.
    """
    passes: dict[str, Any] = {}
    pass_times: dict[str, float] = {}

    # Passes 1-5: focused extraction
    for pass_name, template in PASS_PROMPTS.items():
        t0 = time.time()
        prompt = template.replace("{source}", source_text)
        result = _call_gemini(prompt, model=model)
        elapsed = time.time() - t0

        if result is not None:
            passes[pass_name] = result
            pass_times[pass_name] = round(elapsed, 1)
            logger.info(
                "kg_extraction.pass_complete",
                extra={"pass": pass_name, "label": label, "elapsed": elapsed},
            )
        else:
            passes[pass_name] = {}
            pass_times[pass_name] = round(elapsed, 1)
            logger.warning(
                "kg_extraction.pass_failed",
                extra={"pass": pass_name, "label": label},
            )

    # Pass 6: verification
    t0 = time.time()
    verify_prompt = (
        VERIFY_PROMPT.replace("{source}", source_text)
        .replace("{p1}", json.dumps(passes.get("P1_entities", {})))
        .replace("{p2}", json.dumps(passes.get("P2_events", {})))
        .replace("{p3}", json.dumps(passes.get("P3_relationships", {})))
        .replace("{p4}", json.dumps(passes.get("P4_numbers", {})))
        .replace("{p5}", json.dumps(passes.get("P5_negations", {})))
    )
    verify_result = _call_gemini(verify_prompt, model=model)
    elapsed = time.time() - t0

    passes["P6_verify"] = verify_result or {"missed_facts": []}
    pass_times["P6_verify"] = round(elapsed, 1)

    return {
        "source_len": len(source_text),
        "pass_times": pass_times,
        "passes": passes,
        "n_missed_facts": len(
            passes.get("P6_verify", {}).get("missed_facts", [])
        ),
    }


# =====================================================================
# Prerequisite Provisioning
# =====================================================================


def check_benchmark_prerequisites(
    benchmark_path: str,
    data_dir: str = "datasets/benchmarks/ragtruth",
) -> dict[str, Any]:
    """Check if all benchmark sources have cached fact tables.

    Returns:
        Dict with ``ready`` (bool), ``total_sources``, ``cached``,
        ``missing`` (list of source_ids), and ``source_texts`` for
        missing sources.
    """
    from pathlib import Path

    from llm_judge.calibration.graph_cache import compute_source_hash, get_graph_cache

    # Load benchmark definition
    with open(benchmark_path, encoding="utf-8") as f:
        bm = json.load(f)
    benchmark_ids = set(bm["response_ids"])

    # Load source texts for benchmark cases
    source_info: dict[str, dict[str, Any]] = {}
    source_path = Path(data_dir) / "source_info.jsonl"
    with open(source_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            source_info[row["source_id"]] = row

    response_path = Path(data_dir) / "response.jsonl"
    source_ids_needed: dict[str, str] = {}  # source_id → source_text
    with open(response_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if str(row["id"]) not in benchmark_ids:
                continue
            sid = row.get("source_id", "")
            if sid and sid not in source_ids_needed:
                source = source_info.get(sid, {})
                source_text = _extract_source_text(source)
                if source_text:
                    source_ids_needed[sid] = source_text

    # Check cache coverage
    cache = get_graph_cache()
    cached: list[str] = []
    missing: list[str] = []
    missing_texts: dict[str, str] = {}

    for sid, text in source_ids_needed.items():
        h = compute_source_hash(text)
        if cache.get_by_hash(h) is not None:
            cached.append(sid)
        else:
            missing.append(sid)
            missing_texts[sid] = text

    return {
        "ready": len(missing) == 0,
        "total_sources": len(source_ids_needed),
        "cached": sorted(cached),
        "missing": sorted(missing),
        "missing_texts": missing_texts,
    }


def provision_missing_sources(
    missing_texts: dict[str, str],
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """Extract fact tables for missing sources and store in cache.

    Args:
        missing_texts: Mapping of source_id → source document text.
        model: Gemini model to use for extraction.

    Returns:
        Summary with extracted count, errors, and cost estimate.
    """
    from llm_judge.calibration.graph_cache import get_graph_cache

    cache = get_graph_cache()
    extracted = 0
    errors: list[str] = []
    total_calls = len(missing_texts) * 6

    print(f"\n  Provisioning L2 cache: {len(missing_texts)} sources × 6 passes = {total_calls} API calls")
    print(f"  Estimated cost: ~${len(missing_texts) * 0.006:.3f}")

    for i, (sid, text) in enumerate(sorted(missing_texts.items())):
        print(f"  [{i + 1}/{len(missing_texts)}] Extracting source {sid}...")
        t0 = time.time()

        try:
            tables = extract_fact_tables(text, model=model, label=sid)
            cache.put(text, {"passes": tables.get("passes", tables)})
            elapsed = time.time() - t0
            print(f"    Done in {elapsed:.1f}s — {tables['n_missed_facts']} gaps found by P6")
            extracted += 1
        except Exception as e:
            errors.append(f"{sid}: {str(e)[:80]}")
            logger.error(
                "kg_extraction.provision_error",
                extra={"source_id": sid, "error": str(e)[:80]},
            )

    return {
        "sources_extracted": extracted,
        "sources_failed": len(errors),
        "errors": errors,
        "api_calls": extracted * 6,
    }


def _extract_source_text(source: dict[str, Any]) -> str:
    """Extract source context text from RAGTruth source_info."""
    source_info = source.get("source_info", "")
    if isinstance(source_info, str):
        return source_info
    if isinstance(source_info, dict):
        parts = []
        if "passages" in source_info:
            parts.append(str(source_info["passages"]))
        if "question" in source_info:
            parts.append(str(source_info["question"]))
        for key, val in source_info.items():
            if key not in ("question", "passages"):
                parts.append(f"{key}: {val}")
        return "\n".join(parts)
    return str(source_info)
