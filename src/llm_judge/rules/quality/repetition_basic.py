from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal

from llm_judge.rules.registry import register
from llm_judge.rules.types import Flag, RuleContext, RuleResult


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


Severity = Literal["weak", "strong"]


@register("quality.repetition_basic")
def quality_repetition_basic(ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
    text = (ctx.candidate or "").strip()
    if not text:
        return RuleResult(flags=[])

    n = int(params.get("ngram_n", 3))
    min_rep = int(params.get("min_repeated_ngram_count", 3))
    min_dup_line_ratio = float(params.get("min_duplicate_line_ratio", 0.35))

    # Duplicate lines
    lines = [ln.strip().lower() for ln in re.split(r"\r?\n+", text) if ln.strip()]
    dup_line_ratio = 0.0
    if lines:
        c = Counter(lines)
        dup_lines = sum(v for v in c.values() if v > 1)
        dup_line_ratio = dup_lines / max(1, len(lines))

    # Repeated n-grams
    toks = re.findall(r"[a-z0-9]+", text.lower())
    grams = _ngrams(toks, n)
    rep_ngram_count = 0
    top_ngram: str | None = None
    if grams:
        gc = Counter(grams)
        rep = [(g, v) for (g, v) in gc.items() if v >= min_rep]
        rep_ngram_count = len(rep)
        if rep:
            rep.sort(key=lambda x: (-x[1], x[0]))
            top_ngram = " ".join(rep[0][0])

    # Unigram diversity: low unique-token ratio = word-level repetition
    max_unique_ratio = float(params.get("max_unique_ratio", 0.4))
    min_token_count = int(params.get("min_token_count_for_diversity", 8))
    unique_ratio = len(set(toks)) / max(1, len(toks))

    triggers: list[str] = []
    if dup_line_ratio >= min_dup_line_ratio:
        triggers.append("duplicate_lines")
    if rep_ngram_count > 0:
        triggers.append("repeated_ngrams")
    if len(toks) >= min_token_count and unique_ratio <= max_unique_ratio:
        triggers.append("low_diversity")

    if not triggers:
        return RuleResult(flags=[])

    severity: Severity = (
        "strong"
        if (
            dup_line_ratio >= (min_dup_line_ratio * 1.5)
            or rep_ngram_count >= 2
            or unique_ratio <= max_unique_ratio
        )
        else "weak"
    )

    evidence: list[str] = []
    if top_ngram:
        evidence.append(f"top_ngram={top_ngram}")
    if lines:
        evidence.append(lines[0][:160])

    return RuleResult(
        flags=[
            Flag(
                id="quality.repetition_basic",
                severity=severity,
                details={
                    "dup_line_ratio": round(dup_line_ratio, 3),
                    "repeated_ngram_types": rep_ngram_count,
                    "unique_token_ratio": round(unique_ratio, 3),
                    "ngram_n": n,
                    "triggers": triggers,
                },
                evidence=evidence,
            )
        ]
    )
