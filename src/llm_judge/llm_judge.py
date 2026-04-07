"""
Gate 2 LLM Judge — unified provider with thin adapters.

EPIC 7.5 CHANGE: LLMJudge now accepts an optional PromptTemplate.
When provided, the versioned prompt replaces the hardcoded system prompt.
The _FALLBACK_SYSTEM_PROMPT is used ONLY when no versioned prompt is
available — and a warning is logged to make the fallback visible.

One judge class, multiple providers. Provider is a config choice, not a code change.
Supports: Gemini (free), OpenAI-compatible (OpenAI/Groq), Ollama (local).
"""
from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

import httpx
import structlog

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse

if TYPE_CHECKING:
    from llm_judge.calibration.prompts import PromptTemplate

logger = structlog.get_logger()

_DEFAULT_TIMEOUT_S: Final[float] = 15.0

# EPIC 7.5: This is now a FALLBACK only — versioned prompts take priority.
# Using this fallback is logged as a WARNING so it's never silent.
_FALLBACK_SYSTEM_PROMPT: Final[str] = """\
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


def _build_eval_prompt(
    request: PredictRequest,
    template: PromptTemplate | None = None,
) -> str:
    """Build the evaluation prompt from request, using versioned template if available."""
    conversation = "\n".join(
        f"{m.role.upper()}: {m.content}" for m in request.conversation
    )

    if template and template.user_template:
        # Use versioned user template
        dimensions_str = ", ".join(template.dimensions) if template.dimensions else ""
        return template.user_template.format(
            conversation=conversation,
            candidate_answer=request.candidate_answer,
            dimensions=dimensions_str,
        )

    # Fallback to basic prompt structure
    return (
        f"Customer conversation:\n{conversation}\n\n"
        f"Candidate answer:\n{request.candidate_answer}\n\n"
        'Return JSON: {"decision":"pass/fail","overall_score":float,'
        '"scores":{"relevance":int,"clarity":int,"correctness":int,"tone":int},'
        '"confidence":float,"flags":["..."],'
        '"explanations":{"relevance":"...","clarity":"...","correctness":"...","tone":"..."}}'
    )


def _get_system_prompt(template: PromptTemplate | None = None) -> str:
    """Get the system prompt — versioned if available, fallback otherwise."""
    if template and template.system_prompt:
        return template.system_prompt
    logger.warning(
        "llm_judge.using_fallback_prompt",
        reason="no versioned prompt provided",
    )
    return _FALLBACK_SYSTEM_PROMPT


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
    def build_request(self, prompt: str, system_prompt: str) -> tuple[str, dict[str, str], dict]:
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

    def build_request(self, prompt: str, system_prompt: str) -> tuple[str, dict[str, str], dict]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
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

    def build_request(self, prompt: str, system_prompt: str) -> tuple[str, dict[str, str], dict]:
        url = f"{self._base}/{self._model}:generateContent?key={self._api_key}"
        headers = {"Content-Type": "application/json"}
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "responseMimeType": "application/json",
            },
        }
        return url, headers, payload

    def parse_response(self, data: dict) -> str:
        return data["candidates"][0]["content"]["parts"][-1]["text"]


class OllamaAdapter(ProviderAdapter):
    """Ollama local server or Colab-hosted."""

    def __init__(self, base_url: str, model: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/chat"
        self._model = model

    def build_request(self, prompt: str, system_prompt: str) -> tuple[str, dict[str, str], dict]:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
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

    EPIC 7.5: Now accepts an optional PromptTemplate for versioned prompts.
    When provided, the versioned prompt replaces the hardcoded fallback.
    """

    def __init__(
        self,
        timeout_ms: int | None = None,
        engine: str | None = None,
        prompt_template: PromptTemplate | None = None,
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
        self._prompt_template = prompt_template

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        if os.getenv("SIMULATE_LLM_TIMEOUT") == "1":
            raise TimeoutError("Simulated LLM timeout")

        if self._adapter is None:
            self._adapter = _get_adapter(self._engine)

        system_prompt = _get_system_prompt(self._prompt_template)
        prompt = _build_eval_prompt(request, self._prompt_template)
        url, headers, payload = self._adapter.build_request(prompt, system_prompt)

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
