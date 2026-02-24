from __future__ import annotations

import re
from typing import Any, Literal

from llm_judge.correctness import judge_correctness_proxy
from llm_judge.rubric_store import get_rubric
from llm_judge.rules.engine import RuleEngine
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import PredictRequest, PredictResponse

# Keep this small + high-signal (don’t overdo it)
_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "please",
    "should",
    "so",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "try",
    "up",
    "we",
    "what",
    "when",
    "where",
    "why",
    "will",
    "with",
    "you",
    "your",
    "thanks",
    "thank",
}

# Normalize common intent synonyms (improves overlap without an LLM)
_SYNONYMS: dict[str, str] = {
    "restart": "reset",
    "reboot": "reset",
    "powercycle": "reset",
    "power-cycle": "reset",
    "wifi": "wireless",
    "wi-fi": "wireless",
}


def _msg_content(m: object) -> str:
    """Safely extract message content from model or dict message."""
    content = getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(m, dict):
        c = m.get("content")
        if isinstance(c, str):
            return c
    return ""


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", text.lower())
    out: set[str] = set()
    for t in raw:
        t = _SYNONYMS.get(t, t)
        if t and t not in _STOPWORDS:
            out.add(t)
    return out


def _heuristic_relevance(user_text: str, candidate: str) -> int:
    u = _tokenize(user_text)
    a = _tokenize(candidate)

    if not u or not a:
        return 1

    overlap = len(u & a)

    # With stopwords removed, even overlap==1 can be meaningful (e.g., "router")
    if overlap == 0:
        return 1
    if overlap == 1:
        return 3
    if overlap == 2:
        return 4
    return 5


def _heuristic_tone(candidate: str) -> int:
    rude = {"stupid", "idiot", "shut", "dumb", "moron", "trash"}
    polite = {"please", "thanks", "thank", "kindly"}

    toks = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    if toks & rude:
        return 1
    if toks & polite:
        return 5
    return 4


def _apply_rubric_rules(request: PredictRequest, rubric: Any) -> list[str]:
    """
    Run all rules associated with the rubric against the request.
    Returns a list of flag strings (e.g. 'correctness.definition_sanity:strong').

    Patchable by tests via monkeypatch.
    """
    ctx = RuleContext(request=request, rubric=rubric)
    rules_config = getattr(rubric, "rules", None)

    if rules_config:
        # Inline rules list on the rubric object
        engine = RuleEngine(rules=rules_config)
        engine.run(ctx)
    else:
        # Default rule set when rubric has no explicit rules
        engine = RuleEngine(
            rules=[
                {"id": "correctness.correctness_basic"},
                {"id": "correctness.definition_sanity"},
                {"id": "quality.nonsense_basic"},
            ]
        )
        engine.run(ctx)

    return list(getattr(ctx, "flags", []))


def _has_strong_flag(flags: list[str], prefix: str) -> bool:
    """
    True if any flag matches '<prefix>:strong' (or stronger variants later).
    prefix is the base id like 'quality.nonsense_basic' or 'correctness.definition_sanity'.
    """
    for f in flags:
        base, _, sev = f.partition(":")
        if base == prefix and sev == "strong":
            return True
    return False


def _any_strong_under_namespace(flags: list[str], namespace: str) -> bool:
    """
    True if any flag is '<namespace>.*:strong' e.g. correctness.*:strong.
    """
    for f in flags:
        base, _, sev = f.partition(":")
        if base.startswith(namespace) and sev == "strong":
            return True
    return False


def score_candidate(request: PredictRequest) -> PredictResponse:
    """
    Deterministic scoring: relevance, clarity, correctness (proxy), tone.
    Must NEVER throw for unknown rubrics; return a clean fail response.
    """
    try:
        rubric = get_rubric(request.rubric_id)
    except ValueError:
        return PredictResponse(
            decision="fail",
            overall_score=1.0,
            scores={"relevance": 1, "clarity": 1, "correctness": 1, "tone": 1},
            confidence=0.2,
            flags=["unknown_rubric"],
            explanations={
                "relevance": "Rubric not found; scoring could not be performed.",
                "clarity": "Rubric not found; scoring could not be performed.",
                "correctness": "Rubric not found; scoring could not be performed.",
                "tone": "Rubric not found; scoring could not be performed.",
            },
        )

    user_text = _msg_content(request.conversation[-1]) if request.conversation else ""
    candidate = request.candidate_answer or ""

    relevance = _heuristic_relevance(user_text, candidate)
    tone = _heuristic_tone(candidate)

    clarity = 4
    if not candidate.strip():
        clarity = 1
    elif len(candidate.strip()) < 10:
        clarity = 3

    corr = judge_correctness_proxy(request)
    correctness = int(corr.score)

    scores = {
        "relevance": int(relevance),
        "clarity": int(clarity),
        "correctness": int(correctness),
        "tone": int(tone),
    }

    # Pull policy if present
    policy = getattr(rubric, "decision_policy", None)
    pass_if = float(getattr(policy, "pass_if_overall_score_gte", 3.0)) if policy else 3.0
    fail_if_any = int(getattr(policy, "fail_if_any_dimension_lte", 1)) if policy else 1

    overall_score = sum(scores.values()) / 4.0
    decision: Literal["pass", "fail"] = "pass"
    if overall_score < pass_if or any(v <= fail_if_any for v in scores.values()):
        decision = "fail"

    # Heuristic flags
    flags: list[str] = []
    if scores["relevance"] <= 2:
        flags.append("low_relevance")
    if scores["tone"] <= 2:
        flags.append("rude_tone")

    # Rubric rule flags (inline or default fallback)
    rule_flags = _apply_rubric_rules(request, rubric)
    for f in rule_flags:
        if f not in flags:
            flags.append(f)

    # Hard-fail gates for strong signals (tests cover these branches)
    if _any_strong_under_namespace(flags, "correctness."):
        decision = "fail"
    if _has_strong_flag(flags, "quality.nonsense_basic"):
        decision = "fail"

    confidence = float(getattr(corr, "confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))

    explanations = {
        "relevance": (
            "Answer is unrelated or off-topic"
            if scores["relevance"] <= 2
            else "Answer is relevant to the user's question"
        ),
        "clarity": (
            "Answer is unclear or poorly structured"
            if scores["clarity"] <= 2
            else "Answer is clear and well structured"
        ),
        "correctness": getattr(corr, "explanation", "Correctness proxy applied."),
        "tone": (
            "Rude or inappropriate language detected"
            if scores["tone"] <= 2
            else "Tone is polite and appropriate"
        ),
    }

    return PredictResponse(
        decision=decision,
        overall_score=overall_score,
        scores=scores,
        confidence=confidence,
        flags=flags,
        explanations=explanations,
    )
