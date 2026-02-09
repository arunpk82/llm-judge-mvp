from __future__ import annotations

import re
from typing import Literal

from llm_judge.correctness import judge_correctness_proxy
from llm_judge.rubric_store import get_rubric
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


def score_candidate(request: PredictRequest) -> PredictResponse:
    """
    Deterministic scoring: relevance, clarity, correctness (proxy), tone.
    Must NEVER throw for unknown rubrics; return a clean fail response.
    """
    try:
        #rubric = get_rubric(request.rubric_id)
        get_rubric(request.rubric_id)
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

    # Keep thresholds local unless Rubric schema explicitly exposes policy fields
    pass_if = 3.0
    fail_if_any = 1

    overall_score = sum(scores.values()) / 4.0
    decision: Literal["pass", "fail"] = "pass"
    if overall_score < pass_if or any(v <= fail_if_any for v in scores.values()):
        decision = "fail"

    flags: list[str] = []
    if scores["relevance"] <= 2:
        flags.append("low_relevance")
    if scores["tone"] <= 2:
        flags.append("rude_tone")

    confidence = float(corr.confidence)
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
        "correctness": corr.explanation,
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
