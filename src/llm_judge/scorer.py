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


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(v)))


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


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


def _has_structure(text: str) -> bool:
    # Bullets, numbered lists, headings, or multi-paragraph answers tend to read clearer.
    if "\n\n" in text:
        return True
    if re.search(r"(^|\n)\s*[-*]\s+", text):
        return True
    if re.search(r"(^|\n)\s*\d+\.\s+", text):
        return True
    if re.search(r"(^|\n)#+\s+", text):
        return True
    return False


def _heuristic_tone(candidate: str) -> int:
    rude = {
        "stupid",
        "idiot",
        "shut",
        "dumb",
        "moron",
        "trash",
        "worthless",
        "loser",
    }
    polite = {"please", "thanks", "thank", "kindly", "appreciate"}

    toks = set(re.findall(r"[a-z0-9]+", candidate.lower()))
    # Excessive shouting is a strong negative signal.
    if sum(1 for c in candidate if c.isupper()) >= 20 and len(candidate) >= 60:
        return 2
    if toks & rude:
        return 1
    if toks & polite:
        return 5
    return 4


def _needs_freshness_warning(user_text: str, candidate: str) -> bool:
    # If user asks for latest/today/current and candidate provides definitive facts,
    # deterministic judge should be conservative.
    u = user_text.lower()
    if any(k in u for k in ("latest", "most recent", "today", "current", "now", "this week", "this month")):
        a = candidate.lower()
        has_numbers = bool(re.search(r"\b\d{2,4}\b", a))
        has_dates = bool(re.search(r"\b(20\d{2}|19\d{2})\b", a))
        has_uncertainty = any(h in a for h in ("might", "may", "could", "not sure", "uncertain", "as of"))
        if (has_numbers or has_dates) and not has_uncertainty:
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

    # Clarity: penalize very short answers; reward structure.
    cand_strip = candidate.strip()
    clarity = 4
    if not cand_strip:
        clarity = 1
    elif len(cand_strip) < 15:
        clarity = 2
    elif len(cand_strip) < 40:
        clarity = 3
    if _has_structure(candidate):
        clarity = min(5, clarity + 1)

    corr = judge_correctness_proxy(request)
    correctness = int(corr.score)

    # Conservative adjustment for time-sensitive asks without explicit freshness handling.
    if _needs_freshness_warning(user_text, candidate):
        correctness = max(1, correctness - 1)

    # Candidate dimension scorers (extend here as you add deterministic capabilities)
    dim_scores: dict[str, int] = {
        "relevance": int(relevance),
        "clarity": int(clarity),
        "correctness": int(correctness),
        "tone": int(tone),
    }

    # Only score rubric-defined dimensions (future-proofing for new rubrics)
    scores: dict[str, int] = {}
    for dim in rubric.dimensions:
        if dim in dim_scores:
            scores[dim] = _clamp_int(dim_scores[dim], rubric.scale_min, rubric.scale_max)
        else:
            # Unknown dimension requested by rubric; score at minimum and flag.
            scores[dim] = rubric.scale_min

    # Weighted overall score
    total_w = 0.0
    total_s = 0.0
    for dim, s in scores.items():
        w = float(rubric.weights.get(dim, 1.0))
        total_w += w
        total_s += w * float(s)
    overall_score = (total_s / total_w) if total_w > 0 else float(rubric.scale_min)
    decision: Literal["pass", "fail"] = "pass"
    if overall_score < rubric.pass_if_overall_score_gte or any(
        v <= rubric.fail_if_any_dimension_lte for v in scores.values()
    ):
        decision = "fail"

    flags: list[str] = []
    if "relevance" in scores and scores["relevance"] <= 2:
        flags.append("low_relevance")
    if "tone" in scores and scores["tone"] <= 2:
        flags.append("rude_tone")

    if _needs_freshness_warning(user_text, candidate):
        flags.append("time_sensitive_unverified")
    if any(dim not in dim_scores for dim in rubric.dimensions):
        flags.append("rubric_dimension_unscored")

    # Confidence is a composite: correctness proxy is dominant, but we reduce
    # confidence when multiple dimensions are weak.
    weak_cutoff = max(rubric.scale_min, rubric.fail_if_any_dimension_lte)
    weak_dims = sum(1 for v in scores.values() if v <= weak_cutoff)
    conf = float(corr.confidence)
    conf -= 0.10 * weak_dims
    if "time_sensitive_unverified" in flags:
        conf -= 0.10
    confidence = _clamp_float(conf, 0.05, 0.95)

    explanations: dict[str, str] = {}
    if "relevance" in scores:
        explanations["relevance"] = (
            "Answer is unrelated or off-topic"
            if scores["relevance"] <= 2
            else "Answer is relevant to the user's question"
        )
    if "clarity" in scores:
        explanations["clarity"] = (
            "Answer is unclear or poorly structured"
            if scores["clarity"] <= 2
            else "Answer is clear and well structured"
        )
    if "correctness" in scores:
        explanations["correctness"] = corr.explanation
    if "tone" in scores:
        explanations["tone"] = (
            "Rude or inappropriate language detected"
            if scores["tone"] <= 2
            else "Tone is polite and appropriate"
        )

    return PredictResponse(
        decision=decision,
        overall_score=overall_score,
        scores=scores,
        confidence=confidence,
        flags=flags,
        explanations=explanations,
    )
