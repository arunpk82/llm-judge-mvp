from __future__ import annotations

import logging
import re
from typing import Any, Literal

from llm_judge.correctness import judge_correctness_proxy
from llm_judge.rubric_store import get_rubric
from llm_judge.rules.engine import RuleEngine, load_plan_for_rubric
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

# Rubrics where answers are expected to be short numbers — skip token-overlap
# relevance scoring and quality/nonsense hard-fail checks entirely.
_MATH_RUBRICS: frozenset[str] = frozenset({"math_basic"})

# Regex to detect a pure numeric answer (possibly negative/decimal)
_NUMERIC_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?\s*$")

# Keep this small + high-signal (don't overdo it)
_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can", "could",
    "do", "does", "did", "for", "from", "how", "i", "in", "is", "it", "me",
    "my", "of", "on", "or", "please", "should", "so", "that", "the", "their",
    "then", "this", "to", "try", "up", "we", "what", "when", "where", "why",
    "will", "with", "you", "your", "thanks", "thank",
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


def _heuristic_relevance(user_text: str, candidate: str, rubric_id: str = "") -> int:
    # For math rubrics, a numeric answer is always relevant — don't penalize brevity.
    if rubric_id in _MATH_RUBRICS:
        return 5 if _NUMERIC_RE.match(candidate) else 3

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


def _heuristic_tone(candidate: str, user_text: str = "") -> int:
    """
    Score tone 1-5 using phrase patterns and structural signals.

    Args:
        candidate: The AI response text
        user_text: The user's question (needed for structural checks)

    Returns:
        int: Tone score 1-5
    """
    text_lower = candidate.lower().strip()

    if not text_lower:
        return 1  # Empty response is worst tone

    # ================================================================
    # CATEGORY 1: DISMISSIVE
    # Signal: impatience, deflection, shutting down without a path forward
    # ================================================================
    dismissive_phrases = [
        # Impatient callbacks
        "as i mentioned", "as i said", "like i said", "as previously stated",
        "as i just", "i already told you", "i just explained",
        # Dead-end deflections
        "refer to the documentation", "check the website", "google it",
        "contact support", "contact billing", "call this number",
        "email support@", "email us at",
        # Abrupt shutdowns
        "that's just how it is", "nothing else i can do",
        "there is nothing else", "cannot help with that",
        "can't help with that", "not something we can",
    ]

    # ================================================================
    # CATEGORY 2: BLAME-SHIFTING
    # Signal: second-person + obligation/fault language
    # ================================================================
    blame_phrases = [
        # Accusatory conditionals
        "if you had read", "if you had checked", "next time make sure",
        "next time, make sure", "you should have",
        # Direct accusations
        "you didn't", "you failed to", "you must have entered",
        "you forgot to", "you neglected",
        # Corporate wall
        "not our policy", "not our problem", "not my problem",
        "we are not responsible", "not responsible for",
        "outside of our control", "outside our control",
        # Externalizer
        "issue with your", "problem with your", "fault on your end",
        "on your end",
    ]

    # ================================================================
    # CATEGORY 3: PUSHY
    # Signal: unwarranted urgency, pressure, argumentative
    # ================================================================
    pushy_phrases = [
        # Unwarranted imperatives
        "you need to upgrade", "you must switch", "you have to buy",
        # False consensus
        "everyone is using", "all our best customers",
        "most people prefer", "everyone upgrades",
        # Argumentative
        "why wouldn't you", "it makes no sense", "why would you not",
        # Unearned authority
        "trust me", "believe me", "i guarantee",
        # Artificial urgency
        "don't miss out", "act now", "before it's too late",
        "limited time", "hurry",
    ]

    # ================================================================
    # CATEGORY 4: CONDESCENDING
    # Signal: trivializing difficulty, implying incompetence
    # ================================================================
    condescending_phrases = [
        # Competence doubters
        "maybe you're using it wrong", "are you sure you",
        "did you even try", "did you even read",
        # Patronizing
        "as you can clearly see", "it's actually quite simple",
        "it's really not that hard", "this is basic",
        # Masked insults
        "even a beginner", "standard practice dictates",
        "any user would know", "common sense",
    ]

    # Trivializers - check these separately as they're single words
    # that need sentence-start or standalone context
    trivializer_words = ["obviously", "simply", "basically"]

    # ================================================================
    # PHASE 1: Check for strong negative signals (score 1)
    # ================================================================
    strong_negative_categories = [
        ("blame", blame_phrases),
        ("pushy", pushy_phrases),
        ("condescending", condescending_phrases),
    ]

    for category_name, phrases in strong_negative_categories:
        for phrase in phrases:
            if phrase in text_lower:
                return 1

    # Check trivializers at sentence boundaries
    sentences = re.split(r'[.!?]\s+', text_lower)
    for sent in sentences:
        sent_stripped = sent.strip()
        for word in trivializer_words:
            if sent_stripped.startswith(word):
                return 1

    # ================================================================
    # PHASE 2: Check for moderate negative signals (score 2)
    # ================================================================

    # Dismissive phrases → score 2 (not as severe as blame/pushy, but still bad)
    for phrase in dismissive_phrases:
        if phrase in text_lower:
            return 2

    # ================================================================
    # PHASE 3: Structural checks
    # ================================================================

    # Short response to detailed question → dismissive tone
    # EXCEPT: if response contains empathy/polite signals, short is not dismissive
    response_words = len(re.findall(r'\b\w+\b', text_lower))
    question_words = len(re.findall(r'\b\w+\b', user_text.lower())) if user_text else 0

    has_empathy = any(e in text_lower for e in [
        "i understand", "i'm sorry", "i apologize", "let me help",
        "let me look into", "let me check",
    ])

    # Very short response (under 10 words) to a substantial question (over 15 words)
    # BUT NOT if the response shows empathy (short empathetic acknowledgment is OK)
    if response_words < 10 and question_words > 15 and not has_empathy:
        return 2  # Structural dismissiveness

    # ================================================================
    # PHASE 4: Check for context-aware rude words (score 1)
    # These need surrounding context to distinguish rude from non-rude usage
    # ================================================================
    context_rude_patterns = [
        r"\byou'?re\s+trash\b",          # "you're trash" is rude
        r"\bthat'?s?\s+trash\b",          # "that's trash" is rude
        r"\bstupid\b", r"\bidiot\b", r"\bmoron\b", r"\bdumb\b",
        r"\bshut\s+up\b",
    ]

    for pattern in context_rude_patterns:
        if re.search(pattern, text_lower):
            return 1

    # ================================================================
    # PHASE 5: Check for positive signals
    # Polite words alone don't make good tone — they need substance
    # ================================================================
    polite_phrases = [
        "i understand", "i'm sorry", "i apologize",
        "let me help", "happy to help", "glad to help",
        "i can help", "let me look into", "let me check",
        "thank you for", "thanks for your patience",
        "is there anything else", "don't hesitate to",
        "i appreciate", "great question",
    ]

    empathy_markers = [
        "frustrating", "understand your concern",
        "sorry to hear", "i can see how", "that must be",
    ]

    polite_count = sum(1 for p in polite_phrases if p in text_lower)
    empathy_count = sum(1 for e in empathy_markers if e in text_lower)

    # Strong positive: empathy + polite + substantial response
    if empathy_count >= 1 and polite_count >= 1 and response_words > 30:
        return 5

    # Moderate positive: polite with substance
    if polite_count >= 2 and response_words > 20:
        return 5

    # Some positive signal with reasonable length
    if polite_count >= 1 and response_words > 20:
        return 4

    # Empathy even in short response
    if empathy_count >= 1 or polite_count >= 1:
        return 4

    # ================================================================
    # PHASE 6: Default based on response substance
    # A long, well-structured response with no negative signals
    # is at least neutral-positive (4). Absence of negative + substance = OK.
    # A short response with no signals is neutral (3).
    # ================================================================
    if response_words > 25:
        return 4  # Substantive response with no negative signals
    return 3

def _apply_rubric_rules(request: PredictRequest, rubric: Any) -> list[str]:
    """
    Run all rules associated with the rubric against the request.
    Returns a list of flag strings (e.g. 'correctness.definition_sanity:strong').

    Resolution order (EPIC-2.1 — governed rule plan loading):
      1. Config-driven plan: configs/rules/{rubric_id}_{version}.yaml
      2. Inline rules on the rubric object
      3. Hardcoded defaults (backward compatibility — will be removed)

    Patchable by tests via monkeypatch.
    """
    ctx = RuleContext(request=request, rubric=rubric)
    rubric_id = getattr(rubric, "rubric_id", "")
    rubric_version = getattr(rubric, "version", "v1")
    rules_config = getattr(rubric, "rules", None)

    # 1. Try config-driven plan (governed path)
    try:
        plan = load_plan_for_rubric(rubric_id, rubric_version)
        if plan.rules:
            engine = RuleEngine(rules=plan.rules)
            engine.run(ctx)
            logger.debug(
                "rules.plan.loaded",
                extra={"rubric_id": rubric_id, "version": rubric_version,
                       "source": "config", "rule_count": len(plan.rules)},
            )
            return list(getattr(ctx, "flags", []))
    except (FileNotFoundError, ValueError, OSError):
        # Config file doesn't exist for this rubric — fall through
        pass

    # 2. Inline rules on the rubric object
    if rules_config:
        engine = RuleEngine(rules=rules_config)
        engine.run(ctx)
        logger.debug(
            "rules.plan.loaded",
            extra={"rubric_id": rubric_id, "source": "inline",
                   "rule_count": len(rules_config)},
        )
        return list(getattr(ctx, "flags", []))

    # 3. Hardcoded defaults (backward compat — deprecated)
    logger.warning(
        "rules.plan.fallback_to_hardcoded",
        extra={"rubric_id": rubric_id, "version": rubric_version,
               "hint": "Create configs/rules/{rubric_id}_{version}.yaml"},
    )
    default_rules: list[dict[str, Any]] = [
        {"id": "correctness.basic"},
        {"id": "correctness.definition_sanity"},
    ]
    if rubric_id not in _MATH_RUBRICS:
        default_rules.append({"id": "quality.nonsense_basic"})

    engine = RuleEngine(rules=default_rules)
    engine.run(ctx)
    return list(getattr(ctx, "flags", []))


def _has_strong_flag(flags: list[str], prefix: str) -> bool:
    """
    True if any flag matches '<prefix>:strong'.
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

    rubric_id = request.rubric_id
    user_text = _msg_content(request.conversation[-1]) if request.conversation else ""
    candidate = request.candidate_answer or ""

    relevance = _heuristic_relevance(user_text, candidate, rubric_id)
    tone = _heuristic_tone(candidate, user_text)

    clarity = 4
    if not candidate.strip():
        clarity = 1
    elif rubric_id in _MATH_RUBRICS:
        # Short numeric answers are perfectly clear for math rubrics
        clarity = 5 if _NUMERIC_RE.match(candidate) else 3
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

    # Hard-fail gates for strong signals
    if _any_strong_under_namespace(flags, "correctness."):
        decision = "fail"
    # Only apply quality hard-fail for non-math rubrics
    if rubric_id not in _MATH_RUBRICS and _has_strong_flag(flags, "quality.nonsense_basic"):
        decision = "fail"
    if rubric_id not in _MATH_RUBRICS and _has_strong_flag(flags, "quality.repetition_basic"):
        decision = "fail"

    # Rule flags influence dimensional scores (not just decision).
    # When quality rules fire, scores should reflect the detected issue.
    if rubric_id not in _MATH_RUBRICS:
        if _has_strong_flag(flags, "quality.nonsense_basic"):
            scores["clarity"] = min(scores["clarity"], 1)
            scores["correctness"] = min(scores["correctness"], 1)
            scores["relevance"] = min(scores["relevance"], 2)
        if _has_strong_flag(flags, "quality.repetition_basic"):
            scores["clarity"] = min(scores["clarity"], 1)
            scores["relevance"] = min(scores["relevance"], 2)
        if _has_flag(flags, "quality.nonsense_basic"):
            scores["clarity"] = min(scores["clarity"], 2)
        if _has_flag(flags, "quality.repetition_basic"):
            scores["clarity"] = min(scores["clarity"], 2)

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

def _has_flag(flags: list[str], prefix: str) -> bool:
    """True if any flag matches the prefix (any severity)."""
    for f in flags:
        base, _, _ = f.partition(":")
        if base == prefix:
            return True
    return False