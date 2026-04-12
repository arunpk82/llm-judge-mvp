from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from llm_judge.llm_correctness import LLMCorrectnessResult, judge_correctness_llm
from llm_judge.schemas import PredictRequest


@dataclass(frozen=True)
class CorrectnessProxyResult:
    score: int
    explanation: str
    confidence: float


def judge_correctness_proxy(request: PredictRequest) -> CorrectnessProxyResult:
    """
    Synchronous, deterministic correctness heuristic (cheap + stable).
    Used by scorer.py (sync) so we don't leak async into core scoring.
    """
    ans = (request.candidate_answer or "").strip().lower()

    if not ans:
        return CorrectnessProxyResult(
            score=1, explanation="Empty answer.", confidence=0.9
        )

    hedges = ("i think", "maybe", "not sure", "i don't know", "cannot be sure")
    if any(h in ans for h in hedges):
        return CorrectnessProxyResult(
            score=2,
            explanation="Answer contains uncertainty markers; correctness is questionable.",
            confidence=0.65,
        )

    return CorrectnessProxyResult(
        score=4,
        explanation="No obvious uncertainty markers; proxy assumes likely correct.",
        confidence=0.7,
    )


def judge_correctness(request: PredictRequest) -> Optional[LLMCorrectnessResult]:
    return judge_correctness_llm(request)
