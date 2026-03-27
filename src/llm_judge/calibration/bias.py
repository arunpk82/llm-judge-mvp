"""
Bias Detection and Mitigation (EPIC 7.2).

Detects and mitigates documented LLM-as-judge biases:
  - Position bias: scores change when answer order is swapped
  - Length bias: longer answers systematically scored higher
  - Self-preference bias: LLM favors its own writing style

Mitigation strategies:
  - Position: evaluate with randomized order, measure inconsistency
  - Length: compute correlation between length and score, flag if > threshold
  - Self-preference: detected via calibration against human scores
"""
from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass
from typing import Any, Literal

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import Message, PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


# =====================================================================
# Position Bias Detection
# =====================================================================

@dataclass
class PositionBiasResult:
    """Result of position bias test for a single case."""
    case_id: str
    original_decision: str
    swapped_decision: str
    consistent: bool
    original_score: float
    swapped_score: float
    score_delta: float


def check_position_bias(
    *,
    judge: JudgeEngine,
    cases: list[dict[str, Any]],
    max_cases: int = 50,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Test position bias by swapping conversation order.

    For each case, evaluates twice: once with original order, once with
    the conversation messages reversed. Measures decision consistency
    and score deviation.
    """
    sample = cases[:max_cases] if len(cases) > max_cases else cases

    results: list[PositionBiasResult] = []

    for case in sample:
        conversation = case.get("conversation", [])
        candidate = str(case.get("candidate_answer", ""))
        rubric_id = str(case.get("rubric_id", "chat_quality"))
        case_id = str(case.get("case_id", "unknown"))

        if len(conversation) < 1:
            continue

        try:
            # Original order
            conv_original = [Message(**m) for m in conversation]
            req_original = PredictRequest(
                conversation=conv_original,
                candidate_answer=candidate,
                rubric_id=rubric_id,
            )
            resp_original = judge.evaluate(req_original)

            # Swapped order — reverse conversation messages
            conv_swapped = list(reversed(conversation))
            conv_swapped_msgs = [Message(**m) for m in conv_swapped]
            req_swapped = PredictRequest(
                conversation=conv_swapped_msgs,
                candidate_answer=candidate,
                rubric_id=rubric_id,
            )
            resp_swapped = judge.evaluate(req_swapped)

            results.append(PositionBiasResult(
                case_id=case_id,
                original_decision=resp_original.decision,
                swapped_decision=resp_swapped.decision,
                consistent=resp_original.decision == resp_swapped.decision,
                original_score=resp_original.overall_score,
                swapped_score=resp_swapped.overall_score,
                score_delta=abs(resp_original.overall_score - resp_swapped.overall_score),
            ))

        except Exception as e:
            logger.warning(
                "bias.position.case_failed",
                extra={"case_id": case_id, "error": str(e)},
            )

    if not results:
        return {"tested": 0, "inconsistency_rate": 0.0, "status": "NO_DATA"}

    inconsistent = sum(1 for r in results if not r.consistent)
    avg_delta = statistics.mean(r.score_delta for r in results)

    return {
        "tested": len(results),
        "consistent": len(results) - inconsistent,
        "inconsistent": inconsistent,
        "inconsistency_rate": round(inconsistent / len(results), 4),
        "avg_score_delta": round(avg_delta, 4),
        "threshold": 0.10,  # < 10% inconsistency is acceptable
        "status": "PASS" if inconsistent / len(results) < 0.10 else "FAIL",
        "details": [
            {
                "case_id": r.case_id,
                "consistent": r.consistent,
                "score_delta": round(r.score_delta, 2),
            }
            for r in results if not r.consistent
        ][:10],
    }


# =====================================================================
# Length Bias Detection
# =====================================================================

def check_length_bias(
    *,
    judgments: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Detect length bias by correlating answer length with scores.

    If the Pearson correlation between candidate_answer length and
    overall_score exceeds the threshold, length bias is flagged.
    """
    lengths: list[float] = []
    scores: list[float] = []

    case_map = {str(c.get("case_id", i)): c for i, c in enumerate(cases)}

    for j in judgments:
        case_id = str(j.get("case_id", ""))
        case = case_map.get(case_id, {})
        answer = str(case.get("candidate_answer", ""))

        score = j.get("judge_scores", {})
        if isinstance(score, dict):
            vals = [v for v in score.values() if isinstance(v, (int, float))]
            if vals:
                lengths.append(float(len(answer)))
                scores.append(statistics.mean(vals))

    if len(lengths) < 5:
        return {"tested": len(lengths), "correlation": 0.0, "status": "INSUFFICIENT_DATA"}

    # Compute Pearson correlation
    n = len(lengths)
    mean_l = statistics.mean(lengths)
    mean_s = statistics.mean(scores)
    std_l = statistics.pstdev(lengths)
    std_s = statistics.pstdev(scores)

    if std_l == 0 or std_s == 0:
        correlation = 0.0
    else:
        cov = sum((ln - mean_l) * (sc - mean_s) for ln, sc in zip(lengths, scores)) / n
        correlation = cov / (std_l * std_s)

    return {
        "tested": n,
        "correlation": round(correlation, 4),
        "threshold": threshold,
        "status": "PASS" if abs(correlation) < threshold else "FAIL",
        "mean_length": round(mean_l, 1),
        "mean_score": round(mean_s, 2),
    }


# =====================================================================
# Debiased Judge Wrapper
# =====================================================================

class DebiasedJudge(JudgeEngine):
    """
    Wraps a judge with position bias mitigation.

    Evaluates each case N times with randomized conversation order
    and returns the majority decision with averaged scores.
    """

    def __init__(
        self,
        judge: JudgeEngine,
        *,
        n_evaluations: int = 3,
        seed: int = 42,
    ) -> None:
        self._judge = judge
        self._n = n_evaluations
        self._rng = random.Random(seed)

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        responses: list[PredictResponse] = []

        for i in range(self._n):
            conv = list(request.conversation)
            if i > 0 and len(conv) > 1:
                self._rng.shuffle(conv)

            try:
                req = PredictRequest(
                    conversation=conv,
                    candidate_answer=request.candidate_answer,
                    rubric_id=request.rubric_id,
                )
                resp = self._judge.evaluate(req)
                responses.append(resp)
            except Exception:
                continue

        if not responses:
            return self._judge.evaluate(request)

        # Majority decision
        pass_count = sum(1 for r in responses if r.decision == "pass")
        decision: Literal["pass", "fail"] = "pass" if pass_count > len(responses) / 2 else "fail"

        # Average scores
        avg_overall = statistics.mean(r.overall_score for r in responses)
        all_dims: set[str] = set()
        for r in responses:
            all_dims.update(r.scores.keys())

        avg_scores: dict[str, int] = {}
        for dim in all_dims:
            vals = [r.scores.get(dim) for r in responses if dim in r.scores]
            valid = [v for v in vals if v is not None]
            avg_scores[dim] = round(statistics.mean(valid)) if valid else 3

        avg_confidence = statistics.mean(r.confidence for r in responses)

        # Track if there was disagreement
        flags = list(responses[0].flags)
        if pass_count not in (0, len(responses)):
            flags.append(f"position_bias_disagreement:{pass_count}/{len(responses)}")

        return PredictResponse(
            decision=decision,
            overall_score=round(avg_overall, 2),
            scores=avg_scores,
            confidence=round(avg_confidence, 2),
            flags=flags,
            explanations=responses[0].explanations,
        )
