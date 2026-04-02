"""
Judge engine factory — runtime configuration.

EPIC 7.5 CHANGE: Added 'integrated' engine mode that uses IntegratedJudge
with property configuration, versioned prompts, and hallucination checks.

JUDGE_ENGINE env var controls which engine is used:
  deterministic  — Gate 1 only (default)
  integrated     — Gate 2 with full property pipeline (NEW — PCT-1)
  sequential     — Gate 1 first, escalate to integrated Gate 2
  gemini/groq/openai/ollama — raw Gate 2 (legacy, no properties)
"""
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
      integrated     — Gate 2 with full property pipeline (NEW — PCT-1)
      sequential     — Gate 1 first, escalate to Gate 2 when low confidence
                       Requires GATE2_ENGINE to specify which LLM provider
                       Example: JUDGE_ENGINE=sequential GATE2_ENGINE=gemini
      gemini         — Gate 2 Gemini only + deterministic fallback
      groq           — Gate 2 Groq only + deterministic fallback
      openai         — Gate 2 OpenAI only + deterministic fallback
      ollama         — Gate 2 Ollama only + deterministic fallback
      llm            — alias for openai (backward compatible)

    All LLM engines fall back to deterministic on failure.
    """
    engine_choice = os.getenv("JUDGE_ENGINE", "deterministic").strip().lower()
    timeout_ms = _env_int("JUDGE_TIMEOUT_MS", 15000)

    deterministic = DeterministicJudge()

    if engine_choice == "deterministic":
        return deterministic

    # Integrated mode: full property-aware pipeline (PCT-1+)
    if engine_choice == "integrated":
        from llm_judge.integrated_judge import IntegratedJudge

        gate2_engine = os.getenv("GATE2_ENGINE", "gemini").strip().lower()
        primary = IntegratedJudge(engine=gate2_engine, timeout_ms=timeout_ms)
        return FallbackJudge(primary=primary, fallback=deterministic)

    # Sequential mode: Gate 1 + Gate 2 combined
    if engine_choice == "sequential":
        from llm_judge.sequential_judge import SequentialJudge

        gate2_engine = os.getenv("GATE2_ENGINE", "gemini").strip().lower()
        if gate2_engine == "llm":
            gate2_engine = "openai"

        # Use integrated judge for Gate 2 if property config exists
        gate2: JudgeEngine
        try:
            from llm_judge.integrated_judge import IntegratedJudge
            gate2 = IntegratedJudge(
                engine=gate2_engine, timeout_ms=timeout_ms,
            )
        except Exception:
            gate2 = LLMJudge(timeout_ms=timeout_ms, engine=gate2_engine)

        return SequentialJudge(gate1=deterministic, gate2=gate2)

    # Single LLM mode with fallback (legacy — no properties)
    if engine_choice == "llm":
        engine_choice = "openai"

    primary = LLMJudge(timeout_ms=timeout_ms, engine=engine_choice)
    return FallbackJudge(primary=primary, fallback=deterministic)
