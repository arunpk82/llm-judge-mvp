"""
Gate 2 LLM Judge — unified provider with thin adapters.

One judge class, multiple providers. Provider is a config choice, not a code change.
Supports: Gemini (free), OpenAI-compatible (OpenAI/Groq), Ollama (local).

Usage:
    JUDGE_ENGINE=gemini  GEMINI_API_KEY=...          (Google free tier)
    JUDGE_ENGINE=groq    GROQ_API_KEY=...            (Groq free tier)
    JUDGE_ENGINE=openai  LLM_API_KEY=...             (OpenAI)
    JUDGE_ENGINE=ollama  OLLAMA_BASE_URL=...         (local or Colab)

Architecture:
    LLMJudge → ProviderAdapter → httpx → provider API
    Each adapter is ~15 lines: build_request() + parse_response()
    Adding a new provider = one new adapter class, no other changes.
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Final

import httpx
import structlog

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse

logger = structlog.get_logger()

_DEFAULT_TIMEOUT_S: Final[float] = 15.0

_SYSTEM_PROMPT: Final[str] = """\
You are a strict AI response quality evaluator for customer support.
Score each dimension independently on a 1-5 scale:

RELEVANCE: 1=ignores question, 3=partial, 5=fully addresses it.
CLARITY: 1=incoherent, 3=understandable but rough, 5=crystal clear.
CORRECTNESS: 1=fabricated/false, 3=mixed, 5=fully correct.
TONE: 1=rude/dismissive/blaming/pushy, 3=neutral, 5=warm and empathetic.

PASS if avg >= 3.0 AND no dimension <= 2. FAIL otherwise.

Watch for: responses that look helpful but dodge the question, \
polite language masking dismissiveness, fabricated information, \
ignoring context the user already provided.

Return ONLY valid JSON, no markdown fences.\
"""


def _build_eval_prompt(request: PredictRequest) -> str:
    conversation = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in request.conversation
    )
    return (
        f"Customer conversation:\n{conversation}\n\n"
        f"Candidate answer:\n{request.candidate_answer}\n\n"
        'Return JSON: {"decision":"pass/fail","overall_score":float,'
        '"scores":{"relevance":int,"clarity":int,"correctness":int,"tone":int},'
        '"confidence":float,"flags":["..."],'
        '"explanations":{"relevance":"...","clarity":"...","correctness":"...","tone":"..."}}'
    )


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


# Backward-compatible alias (used by existing tests)
_build_prompt = _build_eval_prompt


def _sanitize_scores(parsed: dict) -> PredictResponse:
    """Normalize parsed JSON into a valid PredictResponse."""
    scores = parsed.get("scores", {})
    for dim in ("relevance", "clarity", "correctness", "tone"):
        if dim in scores:
            scores[dim] = max(1, min(5, int(scores[dim])))

    return PredictResponse(
        decision=parsed.get("decision", "fail"),
        overall_score=max(1.0, min(5.0, float(parsed.get("overall_score", 3.0)))),
        scores=scores,
        confidence=max(0.0, min(1.0, float(parsed.get("confidence", 0.7)))),
        flags=parsed.get("flags", []),
        explanations=parsed.get("explanations"),
    )


# ================================================================
# Provider adapters — each ~15 lines
# ================================================================


class ProviderAdapter(ABC):
    """Thin adapter: translate between our format and a provider's API."""

    @abstractmethod
    def build_request(self, prompt: str) -> tuple[str, dict[str, str], dict]:
        """Return (url, headers, payload) for the provider."""

    @abstractmethod
    def parse_response(self, data: dict) -> str:
        """Extract the text content from the provider's response."""


class OpenAIAdapter(ProviderAdapter):
    """Works with OpenAI, Groq, and any OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model

    def build_request(self, prompt: str) -> tuple[str, dict[str, str], dict]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        return self._url, headers, payload

    def parse_response(self, data: dict) -> str:
        return data["choices"][0]["message"]["content"]


class GeminiAdapter(ProviderAdapter):
    """Google Generative Language API (free tier)."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._base = "https://generativelanguage.googleapis.com/v1beta/models"

    def build_request(self, prompt: str) -> tuple[str, dict[str, str], dict]:
        url = f"{self._base}/{self._model}:generateContent?key={self._api_key}"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "responseMimeType": "application/json",
            },
        }
        return url, headers, payload

    def parse_response(self, data: dict) -> str:
        return data["candidates"][0]["content"]["parts"][0]["text"]


class OllamaAdapter(ProviderAdapter):
    """Ollama local server or Colab-hosted."""

    def __init__(self, base_url: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model

    def build_request(self, prompt: str) -> tuple[str, dict[str, str], dict]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "format": "json",
        }
        return self._url, headers, payload

    def parse_response(self, data: dict) -> str:
        return data["message"]["content"]


# ================================================================
# Adapter factory
# ================================================================


def _get_adapter(engine: str) -> ProviderAdapter:
    """Create the right adapter based on engine name."""
    if engine == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiAdapter(api_key=api_key, model=model)

    if engine == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        base_url = "https://api.groq.com/openai/v1"
        model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        return OpenAIAdapter(base_url=base_url, api_key=api_key, model=model)

    if engine == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        return OllamaAdapter(base_url=base_url, model=model)

    # Default: OpenAI-compatible
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is not set")
    base_url = os.getenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
    return OpenAIAdapter(base_url=base_url, api_key=api_key, model=model)


# ================================================================
# LLM Judge — one class for all providers
# ================================================================


class LLMJudge(JudgeEngine):
    """
    Unified LLM judge. Provider is a config choice, not a code change.

    Uses JudgeEngine protocol — drop-in replacement for DeterministicJudge.
    Combined with FallbackJudge in runtime.py for graceful degradation.
    """

    def __init__(
        self,
        timeout_ms: int | None = None,
        engine: str | None = None,
    ) -> None:
        if timeout_ms is not None:
            self._timeout_s = max(timeout_ms, 1) / 1000.0
        else:
            self._timeout_s = float(
                os.getenv("LLM_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S))
            )
        if engine is not None:
            self._engine: str = engine
        else:
            self._engine = os.getenv("JUDGE_ENGINE", "openai")
        self._adapter: ProviderAdapter | None = None

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        if os.getenv("SIMULATE_LLM_TIMEOUT") == "1":
            raise TimeoutError("Simulated LLM timeout")

        if self._adapter is None:
            self._adapter = _get_adapter(self._engine)

        prompt = _build_eval_prompt(request)
        url, headers, payload = self._adapter.build_request(prompt)

        with httpx.Client(timeout=self._timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        try:
            content = self._adapter.parse_response(data)
        except (KeyError, IndexError) as e:
            logger.error(
                "llm_judge.response_shape",
                engine=self._engine,
                error=str(e),
                raw=str(data)[:200],
            )
            raise RuntimeError(f"Unexpected {self._engine} response: {e}") from e

        try:
            parsed = _extract_json(content)
        except json.JSONDecodeError as e:
            logger.error(
                "llm_judge.json_parse",
                engine=self._engine,
                error=str(e),
                content=content[:200],
            )
            raise RuntimeError(f"{self._engine} output not valid JSON: {e}") from e

        return _sanitize_scores(parsed)
