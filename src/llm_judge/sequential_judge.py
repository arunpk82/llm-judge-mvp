"""
Sequential Gate Judge — Gate 1 first, Gate 2 only when needed.

Implements the sequential gate model:
  Gate 1 (deterministic): fast, cheap, runs on everything
  Gate 2 (LLM): slow, semantic, runs only on low-confidence cases

Escalation criteria (Gate 1 → Gate 2):
  1. No quality flags fired (rules had nothing to say)
  2. All dimension scores are in the default range (3-4)
  3. Overall score is borderline (near the pass/fail threshold)

When Gate 2 runs, its result replaces Gate 1's result entirely.
When Gate 1 is confident (flags fired, clear signal), Gate 2 is skipped.
"""
from __future__ import annotations

import structlog

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse

logger = structlog.get_logger()

# Scores in this range indicate Gate 1 had no strong signal
_DEFAULT_SCORE_RANGE = {3, 4}

# Overall score within this band is borderline pass/fail
_BORDERLINE_LOW = 2.5
_BORDERLINE_HIGH = 3.8


def _is_low_confidence(response: PredictResponse) -> bool:
    """
    Determine if Gate 1's result lacks confidence.

    Returns True when Gate 1 had no meaningful signal — meaning
    Gate 2 should evaluate the case for semantic understanding.
    """
    # Strong signal: quality or correctness flags fired
    has_quality_flags = any(
        f.startswith("quality.") or f.startswith("correctness.")
        for f in response.flags
    )
    if has_quality_flags:
        return False  # Gate 1 is confident — flags fired

    # Strong signal: any dimension scored very low (1-2) or very high (5)
    scores = list(response.scores.values())
    has_extreme_score = any(s <= 2 or s >= 5 for s in scores)
    if has_extreme_score:
        return False  # Gate 1 found something definitive

    # All scores in default range (3-4) with no flags = no signal
    all_default = all(s in _DEFAULT_SCORE_RANGE for s in scores)

    # Borderline overall score = uncertain decision
    borderline = _BORDERLINE_LOW <= response.overall_score <= _BORDERLINE_HIGH

    # Escalate if both conditions are true:
    # scores are all defaults AND overall is in the borderline zone
    if all_default and borderline:
        logger.info(
            "judge.escalating",
            reason="low_confidence",
            overall=response.overall_score,
            scores=response.scores,
            flags=response.flags,
        )
        return True

    return False


class SequentialJudge(JudgeEngine):
    """
    Combined Gate 1 + Gate 2 pipeline.

    Gate 1 runs on every request. If Gate 1 is confident (flags fired,
    extreme scores, clear pass/fail), its result is returned immediately.
    If Gate 1 lacks confidence, Gate 2 (LLM) evaluates the case.

    If Gate 2 fails (API error, timeout), Gate 1's result is used as fallback.
    """

    def __init__(self, gate1: JudgeEngine, gate2: JudgeEngine) -> None:
        self._gate1 = gate1
        self._gate2 = gate2

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        # Always run Gate 1
        gate1_result = self._gate1.evaluate(request)

        # Check if Gate 1 is confident
        if not _is_low_confidence(gate1_result):
            logger.debug(
                "judge.gate1_confident",
                decision=gate1_result.decision,
                scores=gate1_result.scores,
            )
            return gate1_result

        # Gate 1 lacks confidence — escalate to Gate 2
        try:
            gate2_result = self._gate2.evaluate(request)
            logger.info(
                "judge.gate2_used",
                gate1_decision=gate1_result.decision,
                gate2_decision=gate2_result.decision,
                gate1_scores=gate1_result.scores,
                gate2_scores=gate2_result.scores,
            )
            return gate2_result
        except Exception as e:  # noqa: BLE001
            # Gate 2 failed — fall back to Gate 1's result
            logger.warning(
                "judge.gate2_failed",
                error=str(e),
                fallback_decision=gate1_result.decision,
            )
            return gate1_result
