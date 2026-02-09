from __future__ import annotations

import os

import structlog

from llm_judge.deterministic_judge import DeterministicJudge
from llm_judge.judge_base import JudgeEngine
from llm_judge.llm_judge import LLMJudge
from llm_judge.schemas import PredictRequest, PredictResponse

logger = structlog.get_logger()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class FallbackJudge(JudgeEngine):
    def __init__(self, primary: JudgeEngine, fallback: JudgeEngine) -> None:
        self._primary = primary
        self._fallback = fallback

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        try:
            return self._primary.evaluate(request)
        except Exception as e:  # noqa: BLE001
            logger.warning("judge.fallback", error=str(e))
            return self._fallback.evaluate(request)


def get_judge_engine() -> JudgeEngine:
    engine_choice = os.getenv("JUDGE_ENGINE", "deterministic").strip().lower()
    timeout_ms = _env_int("JUDGE_TIMEOUT_MS", 2000)

    deterministic = DeterministicJudge()

    if engine_choice == "llm":
        primary = LLMJudge(timeout_ms=timeout_ms)
        return FallbackJudge(primary=primary, fallback=deterministic)

    return deterministic
