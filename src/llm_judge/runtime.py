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
    """
    Factory for judge engines.

    JUDGE_ENGINE env var controls which engine is used:
      deterministic  — Gate 1 only (default)
      gemini         — Gemini API (free tier) + deterministic fallback
      groq           — Groq API (free tier) + deterministic fallback
      openai         — OpenAI API + deterministic fallback
      ollama         — Ollama local + deterministic fallback
      llm            — alias for openai (backward compatible)

    All LLM engines fall back to deterministic on failure.
    """
    engine_choice = os.getenv("JUDGE_ENGINE", "deterministic").strip().lower()
    timeout_ms = _env_int("JUDGE_TIMEOUT_MS", 15000)

    deterministic = DeterministicJudge()

    if engine_choice == "deterministic":
        return deterministic

    # Map legacy "llm" to "openai"
    if engine_choice == "llm":
        engine_choice = "openai"

    primary = LLMJudge(timeout_ms=timeout_ms, engine=engine_choice)
    return FallbackJudge(primary=primary, fallback=deterministic)
