from __future__ import annotations

import re
from typing import Any, Literal

from llm_judge.correctness import judge_correctness_proxy
from llm_judge.rubric_store import get_rubric
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import PredictRequest, PredictResponse

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

_SYNONYMS: dict[str, str] = {
    "restart": "reset",
    "reboot": "reset",
    "powercycle": "reset",
    "power-cycle": "reset",
    "wifi": "wireless",
    "wi-fi": "wireless",
}

# Hard-fail signals (flag-first, scorer-decides)
# NOTE: These are BASE IDS (no :severity suffix).
_HARD_FAIL_FLAGS: set[str] = {
    # legacy
    "unknown_rubric",
    # correctness
    "correctness.definition_wrong",
    "correctness.unsafe_advice",
    "correctness.known_fact_mismatch",
    "correctness.math_incorrect",
    "correctness.nonsense_detected",
    "correctness.empty_answer",
    # PR6 correctness
    "correctness.definition_sanity",
    # quality (base hard fails)
    "quality.off_topic",
    "quality.category_error",
    "quality.vacuous",
}

# Strong quality flags that should hard-fail when severity is strong.
# (Keep small & high-confidence.)
_STRONG_QUALITY_HARD_FAIL: set[str] = {
    "quality.nonsense_basic",
}

# Strong (but not always hard) clarity impacts (base ids)
_CLARITY_DOWN_FLAGS: set[str] = {
    "quality.repetition_basic",
}


def _msg_content(m: object) -> str:
    content = getattr(m, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(m, dict):
        c = m.get("content")
        if isinstance(c, str):
            return c
    return ""


def _tokenize(text: str) -> set[str]:
    raw = re.findall(r"[a-z0-9]+", (text or "").lower())
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
    toks = set(re.findall(r"[a-z0-9]+", (candidate or "").lower()))
    if toks & rude:
        return 1
    if toks & polite:
        return 5
    return 4


def _flag_to_str(flag: Any) -> str | None:
    """
    Convert a structured Flag (or string) into a stable string flag identifier.
    - If it's already a string, return it.
    - If it has id/severity attributes, return "id:severity".
    """
    if isinstance(flag, str):
        return flag
    fid = getattr(flag, "id", None)
    sev = getattr(flag, "severity", None)
    if isinstance(fid, str) and isinstance(sev, str):
        return f"{fid}:{sev}"
    if isinstance(fid, str):
        return fid
    return None


def _apply_rubric_rules(request: PredictRequest, rubric: object) -> list[str]:
    """
    Apply deterministic rules for this rubric.

    Priority:
      1) If rubric YAML defines `rules: [...]`, use that (inline configuration)
      2) Else, fall back to configs/rules/{rubric_id}_{version}.yaml via rules.engine

    Returns a de-duplicated list of flag strings (id or id:severity).
    """
    try:
        from llm_judge.rules.registry import get_rule
    except Exception:
        return []

    ctx = RuleContext(request=request, rubric=rubric)
    out_flags: list[str] = []

    rule_items = getattr(rubric, "rules", None)
    if isinstance(rule_items, list) and rule_items:
        # 1) Inline rules inside rubric file
        for item in rule_items:
            rid: str | None = None
            params: dict[str, Any] = {}

            if isinstance(item, str):
                rid = item
            elif isinstance(item, dict):
                rid_raw = item.get("id")
                if isinstance(rid_raw, str):
                    rid = rid_raw
                p = item.get("params")
                if isinstance(p, dict):
                    params = p

            if not rid:
                continue

            try:
                rule = get_rule(rid)
            except Exception:
                continue

            try:
                rr = rule.apply(ctx, params) if hasattr(rule, "apply") else rule(ctx, params)
                rr_flags = getattr(rr, "flags", None)
                if isinstance(rr_flags, list):
                    for f in rr_flags:
                        fs = _flag_to_str(f)
                        if fs:
                            out_flags.append(fs)
            except Exception:
                continue
    else:
        # 2) Fallback to rule plan YAML via rules.engine
        try:
            from llm_judge.rules.engine import load_plan_for_rubric, run_rules
        except Exception:
            pass
        else:
            version = getattr(rubric, "version", None)
            if not isinstance(version, str) or not version:
                version = "v1"

            try:
                plan = load_plan_for_rubric(request.rubric_id, version)
                rr = run_rules(ctx, plan)
                for f in rr.flags:
                    fs = _flag_to_str(f)
                    if fs:
                        out_flags.append(fs)
            except Exception:
                pass

    # Include legacy ctx.flags strings if rules appended them
    ctx_flags = getattr(ctx, "flags", None)
    if isinstance(ctx_flags, list):
        for f in ctx_flags:
            if isinstance(f, str):
                out_flags.append(f)

    # Deduplicate while preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for f in out_flags:
        if f not in seen:
            seen.add(f)
            uniq.append(f)
    return uniq


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
    candidate = (request.candidate_answer or "").strip()

    relevance = _heuristic_relevance(user_text, candidate)
    tone = _heuristic_tone(candidate)

    clarity = 4
    if not candidate:
        clarity = 1
    elif len(candidate) < 10:
        clarity = 3

    corr = judge_correctness_proxy(request)
    correctness = int(corr.score)

    scores = {
        "relevance": int(relevance),
        "clarity": int(clarity),
        "correctness": int(correctness),
        "tone": int(tone),
    }

    flags: list[str] = []

    # Legacy signals
    if scores["relevance"] <= 2:
        flags.append("low_relevance")
    if scores["tone"] <= 2:
        flags.append("rude_tone")

    # Deterministic rule signals
    rule_flags = _apply_rubric_rules(request, rubric)
    for f in rule_flags:
        if f not in flags:
            flags.append(f)

    # Base ids (drop :severity for mapping)
    base_flags = {f.split(":", 1)[0] for f in flags}

    # Strong base ids (only those explicitly marked :strong)
    strong_base_flags = {
        f.split(":", 1)[0]
        for f in flags
        if isinstance(f, str) and f.endswith(":strong")
    }

    # Score impacts (flag-first, scorer decides)
    if "quality.off_topic" in base_flags:
        scores["relevance"] = min(scores["relevance"], 1)
    if "quality.category_error" in base_flags:
        scores["correctness"] = min(scores["correctness"], 1)
    if "quality.vacuous" in base_flags:
        scores["clarity"] = min(scores["clarity"], 2)
    if any(f in _CLARITY_DOWN_FLAGS for f in base_flags):
        scores["clarity"] = min(scores["clarity"], 2)

    # Correctness downshifts from deterministic flags
    if "correctness.definition_wrong" in base_flags:
        scores["correctness"] = min(scores["correctness"], 1)
    if "correctness.definition_sanity" in base_flags:
        scores["correctness"] = min(scores["correctness"], 1)

    policy = getattr(rubric, "decision_policy", None)
    pass_if = float(getattr(policy, "pass_if_overall_score_gte", 3.0)) if policy is not None else 3.0
    fail_if_any_lte = int(getattr(policy, "fail_if_any_dimension_lte", 2)) if policy is not None else 2

    overall_score = sum(scores.values()) / 4.0
    decision: Literal["pass", "fail"] = "pass"

    if overall_score < pass_if or any(v <= fail_if_any_lte for v in scores.values()):
        decision = "fail"

    # Base hard-fail flags (regardless of severity)
    if any(f in _HARD_FAIL_FLAGS for f in base_flags):
        decision = "fail"

    # PR6: Strong correctness flags hard-fail
    if any(f.startswith("correctness.") for f in strong_base_flags):
        decision = "fail"

    # PR6: Selected strong quality flags hard-fail
    if any(f in _STRONG_QUALITY_HARD_FAIL for f in strong_base_flags):
        decision = "fail"

    confidence = float(corr.confidence)
    confidence = max(0.0, min(1.0, confidence))

    explanations = {
        "relevance": (
            "Answer is unrelated or off-topic"
            if scores["relevance"] <= 2
            else "Answer is relevant to the user's question"
        ),
        "clarity": (
            "Answer is vague, repetitive, or poorly structured"
            if scores["clarity"] <= 2
            else "Answer is clear and well structured"
        ),
        "correctness": (
            "Correctness/quality rule(s) flagged an issue: "
            + ", ".join([f for f in base_flags if f.startswith(("correctness.", "quality."))])
            if any(f.startswith(("correctness.", "quality.")) for f in base_flags)
            else corr.explanation
        ),
        "tone": (
            "Answer uses rude or inappropriate language"
            if scores["tone"] <= 2
            else "Tone is appropriate and respectful"
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