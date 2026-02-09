from __future__ import annotations

from abc import ABC, abstractmethod

from llm_judge.schemas import PredictRequest, PredictResponse


class JudgeEngine(ABC):
    """Engine contract: synchronous evaluation returning PredictResponse."""

    @abstractmethod
    def evaluate(self, request: PredictRequest) -> PredictResponse:
        """Evaluate a candidate answer and return a PredictResponse."""
        raise NotImplementedError
