from __future__ import annotations

import re
from typing import Any, Literal

from llm_judge.rules.registry import register
from llm_judge.rules.types import Flag, RuleContext, RuleResult

Severity = Literal["weak", "strong"]


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(1 for ch in text if ch.isalpha())
    return alpha / max(1, len(text))


def _max_repeat_run(text: str) -> int:
    if not text:
        return 0
    max_run = 1
    run = 1
    prev = text[0]
    for ch in text[1:]:
        if ch == prev:
            run += 1
            max_run = max(max_run, run)
        else:
            prev = ch
            run = 1
    return max_run


def _gibberish_token_ratio(text: str) -> float:
    toks = re.findall(r"[A-Za-z]{2,}", text)
    if not toks:
        return 1.0
    gib = 0
    for t in toks:
        t_low = t.lower()
        # Heuristic: very low vowel density or extreme consonant runs tend to be gibberish.
        vowels = sum(1 for ch in t_low if ch in "aeiou")
        if vowels == 0:
            gib += 1
            continue
        vowel_ratio = vowels / len(t_low)
        if vowel_ratio < 0.20:
            gib += 1
            continue
        if re.search(r"[^aeiou]{6,}", t_low):
            gib += 1
            continue
    return gib / max(1, len(toks))


@register("quality.nonsense_basic")
def quality_nonsense_basic(ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
    text = (ctx.candidate or "").strip()
    if not text:
        return RuleResult(flags=[])

    min_alpha_ratio = float(params.get("min_alpha_ratio", 0.55))
    min_word_count = int(params.get("min_word_count", 3))
    max_repeat_char_run = int(params.get("max_repeat_char_run", 6))
    min_gib_ratio = float(params.get("min_gibberish_token_ratio", 0.55))

    alpha_ratio = _alpha_ratio(text)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    repeat_run = _max_repeat_run(text)
    gib_ratio = _gibberish_token_ratio(text)

    triggers: list[str] = []
    if alpha_ratio < min_alpha_ratio:
        triggers.append("low_alpha_ratio")
    if word_count < min_word_count:
        triggers.append("too_short")
    if repeat_run >= max_repeat_char_run:
        triggers.append("char_run")
    if gib_ratio >= min_gib_ratio:
        triggers.append("gibberish_tokens")

    if not triggers:
        return RuleResult(flags=[])

    severity: Severity = "strong" if len(triggers) >= 2 else "weak"
    return RuleResult(
        flags=[
            Flag(
                id="quality.nonsense_basic",
                severity=severity,
                details={
                    "alpha_ratio": round(alpha_ratio, 3),
                    "word_count": word_count,
                    "max_repeat_run": repeat_run,
                    "gibberish_token_ratio": round(gib_ratio, 3),
                    "triggers": triggers,
                },
                evidence=[text[:160]],
            )
        ]
    )
