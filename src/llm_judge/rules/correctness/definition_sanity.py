from __future__ import annotations

import re
from typing import Any, Literal

from llm_judge.rules.registry import register
from llm_judge.rules.types import Flag, RuleContext, RuleResult

Severity = Literal["weak", "strong"]


_DEF_CUES = (
    " is ",
    " means ",
    " refers to ",
    " defined as ",
    " is called ",
)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _normalize(text: str) -> str:
    # keep it deterministic and simple: lowercase, trim, collapse whitespace
    return re.sub(r"\s+", " ", (text or "").strip().lower())


@register("correctness.definition_sanity")
def correctness_definition_sanity(ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
    """Catch obvious definition failures.

    Goal: close false negatives where the assistant produces definition-shaped text
    that is empty / self-referential / placeholder / vacuous.

    Policy intent:
    - emits STRONG for self-referential circular definitions (high confidence)
    - emits STRONG when 2+ independent triggers match
    - otherwise emits WEAK (signal, not a guaranteed fail)
    """

    user = _normalize(getattr(ctx, "user_text", ""))
    ans_raw = (ctx.candidate or "").strip()
    ans = _normalize(ans_raw)

    if not ans:
        return RuleResult(flags=[])

    # Activate mainly for definition-seeking prompts.
    # (Keep conservative to reduce false positives.)
    if not re.search(r"\b(what is|define|definition of|meaning of)\b", user):
        return RuleResult(flags=[])

    min_wc = int(params.get("min_definition_word_count", 8))
    wc = _word_count(ans_raw)

    # Optional: require at least one definition cue phrase in the answer to proceed.
    # Defaults to False for backwards compatibility.
    require_def_cue = bool(params.get("require_definition_cue", False))
    has_def_cue = any(cue.strip() in ans for cue in _DEF_CUES)

    if require_def_cue and not has_def_cue:
        return RuleResult(flags=[])

    triggers: list[str] = []

    # Too short to be a meaningful definition.
    if wc < min_wc:
        triggers.append("too_short")

    # Self-referential (circular) definition:
    # "X is X", "X means X", "X refers to X", "X defined as X"
    # Allow light punctuation around term.
    # Keep term length bounded to avoid matching entire paragraphs.
    # NOTE: This trigger is considered HIGH SIGNAL -> STRONG.
    m = re.search(
        r"\b([a-z][a-z0-9_\- ]{1,50})\b\s+"
        r"(is|means|refers to|defined as|is called)\s+"
        r"\b\1\b",
        ans,
    )
    self_ref_term: str | None = None
    if m:
        triggers.append("self_referential")
        self_ref_term = m.group(1).strip()

    # Generic filler / vacuous definition (2+ filler cues)
    filler = (
        "something",
        "things",
        "various",
        "etc",
        "and so on",
        "kind of",
        "sort of",
        "basically",
        "in general",
    )
    filler_hits = sum(1 for f in filler if f in ans)
    if filler_hits >= 2:
        triggers.append("vacuous_filler")

    # Unresolved placeholder patterns
    if re.search(r"\b(\[\s*insert\s*\]|<\s*insert\s*>|todo|tbd)\b", ans):
        triggers.append("placeholder")

    if not triggers:
        return RuleResult(flags=[])

    # Severity policy tuning (params override defaults)
    strong_on_self_ref = bool(params.get("strong_on_self_referential", True))
    strong_min_triggers = int(params.get("strong_min_triggers", 2))

    severity: Severity = "weak"
    if ("self_referential" in triggers and strong_on_self_ref) or (len(triggers) >= strong_min_triggers):
        severity = "strong"

    details: dict[str, Any] = {
        "word_count": wc,
        "min_word_count": min_wc,
        "triggers": triggers,
        "require_definition_cue": require_def_cue,
        "has_definition_cue": has_def_cue,
    }
    if self_ref_term:
        details["self_ref_term"] = self_ref_term

    return RuleResult(
        flags=[
            Flag(
                id="correctness.definition_sanity",
                severity=severity,
                details=details,
                evidence=[ans_raw[:200]],
            )
        ]
    )