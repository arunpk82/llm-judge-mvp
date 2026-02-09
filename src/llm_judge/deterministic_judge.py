from __future__ import annotations

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse
from llm_judge.scorer import score_candidate


class DeterministicJudge(JudgeEngine):
    def evaluate(self, request: PredictRequest) -> PredictResponse:
        return score_candidate(request)
