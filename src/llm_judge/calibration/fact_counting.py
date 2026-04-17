"""
L3 Fact-Counting Verification — ADR-0027.

Ported from experiments/exp43_end_to_end.py (Phase 2).

The fact-counting approach decomposes each sentence into individual
facts, classifies each against the source document using a five-status
taxonomy, and computes a ``supported / total`` ratio. This derived
confidence is superior to both MiniCheck's prob_supported and LLM
self-reported confidence (Learning L61).

Key result from Exp 43 (303 sentences, RAGTruth-50):
  - Ratio >= 0.80: 230 sentences auto-cleared, 0 false negatives (100% grounding precision)
  - Maximum hallucination ratio: 0.750 (no hallucinated sentence reaches 0.80)
  - 76% of sentences resolved without L4, saving API cost and latency

Five-status taxonomy:
  SUPPORTED    — fact explicitly stated in source (cite the sentence)
  NOT_FOUND    — fact absent from source
  CONTRADICTED — fact conflicts with source
  SHIFTED      — words subtly change meaning vs source
  INFERRED     — reasonable inference from context

Design decisions:
  - Model is configurable via FACT_COUNTING_MODEL env var (default: gemma-4-31b-it)
  - Uses the same Gemini API pattern as _l4_gemini_check in hallucination.py
  - Robust JSON parser handles markdown fences, thinking tokens, partial JSON
  - Falls back gracefully: API error → ratio=0.0, sentence escalates to L4
  - Timeout/retry logic matches production patterns (3 attempts, exponential backoff)

See also: ADR-0027 (L3 method selection), ADR-0022 (fact-counting design),
          28 Metrics Eval Reference v5.2 Experiment 43.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# =====================================================================
# Prompt — from Exp 43 (validated on 303 sentences)
# =====================================================================

FACT_COUNTING_PROMPT = """Decompose the CLAIM into individual facts, then check each against the SOURCE.

SOURCE:
{source}

CLAIM: "{claim}"

For EACH fact in the claim:
- SUPPORTED: fact is explicitly stated in the source (cite the sentence)
- NOT_FOUND: fact is absent from the source
- CONTRADICTED: fact conflicts with what the source says
- SHIFTED: words used subtly change the meaning vs the source
- INFERRED: fact is a reasonable inference from context (e.g., "last November" → "2015")

Return ONLY this JSON:
{{"facts": [{{"fact": "...", "status": "SUPPORTED|NOT_FOUND|CONTRADICTED|SHIFTED|INFERRED", "source_ref": "which sentence or n/a"}}], "supported": 0, "not_found": 0, "contradicted": 0, "shifted": 0, "inferred": 0, "total": 0, "verdict": "GROUNDED or HALLUCINATED"}}"""


# =====================================================================
# Result dataclass
# =====================================================================


@dataclass
class FactCountResult:
    """Result from fact-counting verification of a single sentence."""

    supported: int = 0
    not_found: int = 0
    contradicted: int = 0
    shifted: int = 0
    inferred: int = 0
    total: int = 0
    verdict: str = ""  # "GROUNDED" or "HALLUCINATED" (from LLM)
    ratio: float = 0.0  # supported / total (the derived confidence)
    auto_clear: bool = False  # ratio >= threshold → safe to clear
    facts: list[dict[str, str]] = field(default_factory=list)
    error: str = ""  # non-empty on API/parse failure
    model: str = ""
    latency_ms: int = 0

    def to_evidence_dict(self) -> dict[str, Any]:
        """Flat dict for sentence_detail embedding in judgments.jsonl."""
        return {
            "fc_supported": self.supported,
            "fc_not_found": self.not_found,
            "fc_contradicted": self.contradicted,
            "fc_shifted": self.shifted,
            "fc_inferred": self.inferred,
            "fc_total": self.total,
            "fc_ratio": self.ratio,
            "fc_auto_clear": self.auto_clear,
            "fc_verdict": self.verdict,
            "fc_model": self.model,
            "fc_latency_ms": self.latency_ms,
        }


# =====================================================================
# JSON parser — robust, handles LLM output quirks
# (Ported from exp43_end_to_end.py parse_json_safe)
# =====================================================================


def _parse_json_safe(raw: str) -> dict[str, Any]:
    """Parse JSON from LLM output, handling markdown fences and thinking tokens."""
    # Strip thinking tokens (Gemma 4 can emit these)
    cleaned = re.sub(
        r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.DOTALL
    ).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fence
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try extracting any JSON object
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"_raw": raw[:500], "_parse_error": True}


# =====================================================================
# LLM caller — reuses the same Gemini API pattern as _l4_gemini_check
# =====================================================================

_MAX_RETRIES = 3
_TIMEOUT_SECONDS = 60.0
_RETRY_BASE_WAIT = 10


def _call_llm(prompt: str, *, model: str | None = None) -> str:
    """Call Gemini/Gemma API with retry logic.

    Uses GEMINI_API_KEY env var. Model defaults to FACT_COUNTING_MODEL
    env var, then to "gemma-4-31b-it" (validated in Exp 43).

    Returns raw response text. Raises on exhausted retries.
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    if model is None:
        model = os.environ.get("FACT_COUNTING_MODEL", "gemma-4-31b-it")

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=_TIMEOUT_SECONDS) as client:
                resp = client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )
                resp.raise_for_status()
                data = resp.json()

            parts = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [])
            )
            if not parts:
                return ""
            return "\n".join(p.get("text", "") for p in parts if p.get("text"))

        except Exception as e:
            last_error = e
            if attempt < _MAX_RETRIES:
                wait = attempt * _RETRY_BASE_WAIT
                logger.warning(
                    "fact_counting.retry",
                    extra={
                        "attempt": attempt,
                        "error": str(e)[:80],
                        "wait_s": wait,
                    },
                )
                time.sleep(wait)

    raise RuntimeError(
        f"fact_counting: all {_MAX_RETRIES} attempts failed: {last_error}"
    )


# =====================================================================
# Core function — fact-count a single sentence
# =====================================================================


def check_fact_counting(
    sentence: str,
    source_doc: str,
    *,
    threshold: float = 0.80,
    model: str | None = None,
    source_max_chars: int = 6000,
) -> FactCountResult:
    """
    Fact-count a single sentence against a source document.

    Decomposes the sentence into facts, classifies each against the
    source using the five-status taxonomy, and computes the supported/total
    ratio. If ratio >= threshold, the sentence is auto-cleared as grounded.

    Args:
        sentence: The response sentence to verify.
        source_doc: The source document to verify against.
        threshold: Auto-clear threshold (default 0.80 from Exp 43).
        model: LLM model to use (default from env FACT_COUNTING_MODEL).
        source_max_chars: Truncate source beyond this (default 6000).

    Returns:
        FactCountResult with ratio, auto_clear flag, and per-fact evidence.
    """
    resolved_model = model or os.environ.get(
        "FACT_COUNTING_MODEL", "gemma-4-31b-it"
    )
    start_ms = int(time.time() * 1000)

    try:
        prompt = FACT_COUNTING_PROMPT.format(
            source=source_doc[:source_max_chars],
            claim=sentence,
        )
        raw = _call_llm(prompt, model=resolved_model)
        parsed = _parse_json_safe(raw)
    except Exception as e:
        logger.warning(
            "fact_counting.error",
            extra={"error": str(e)[:120], "sentence": sentence[:60]},
        )
        elapsed = int(time.time() * 1000) - start_ms
        return FactCountResult(
            error=str(e)[:200],
            model=resolved_model,
            latency_ms=elapsed,
        )

    elapsed = int(time.time() * 1000) - start_ms

    if parsed.get("_parse_error"):
        return FactCountResult(
            error=f"json_parse_error: {parsed.get('_raw', '')[:100]}",
            model=resolved_model,
            latency_ms=elapsed,
        )

    # Extract counts from parsed JSON
    supported = int(parsed.get("supported", 0))
    not_found = int(parsed.get("not_found", 0))
    contradicted = int(parsed.get("contradicted", 0))
    shifted = int(parsed.get("shifted", 0))
    inferred = int(parsed.get("inferred", 0))
    total = int(parsed.get("total", 0))

    # Recompute total if LLM's count is wrong
    computed_total = supported + not_found + contradicted + shifted + inferred
    if computed_total > 0 and computed_total != total:
        total = computed_total

    # Compute ratio (the derived confidence — L61)
    ratio = round(supported / total, 3) if total > 0 else 0.0

    # Auto-clear decision
    auto_clear = ratio >= threshold

    verdict = str(parsed.get("verdict", "")).upper()
    if verdict not in ("GROUNDED", "HALLUCINATED"):
        # Derive verdict from ratio if LLM verdict is unclear
        verdict = "GROUNDED" if auto_clear else "HALLUCINATED"

    facts = parsed.get("facts", [])
    if not isinstance(facts, list):
        facts = []

    return FactCountResult(
        supported=supported,
        not_found=not_found,
        contradicted=contradicted,
        shifted=shifted,
        inferred=inferred,
        total=total,
        verdict=verdict,
        ratio=ratio,
        auto_clear=auto_clear,
        facts=facts,
        model=resolved_model,
        latency_ms=elapsed,
    )
