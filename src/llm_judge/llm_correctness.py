from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import httpx

from llm_judge.schemas import PredictRequest


class LLMCorrectnessResult:
    def __init__(self, score: int, explanation: str, confidence: float) -> None:
        self.score = score
        self.explanation = explanation
        self.confidence = confidence


def _build_correctness_prompt(request: PredictRequest) -> str:
    user_text = request.conversation[-1].content if request.conversation else ""
    return (
        "You are a strict correctness grader.\n"
        "Return ONLY JSON with keys: score (1-5), explanation (string), confidence (0-1).\n\n"
        f"User: {user_text}\n"
        f"Answer: {request.candidate_answer}\n"
    )


def _extract_openai_content(data: dict[str, Any]) -> Any:
    """
    Supports OpenAI-like response shape:
      {"choices":[{"message":{"content":"{...json...}"}}]}
    Returns the parsed JSON payload inside message.content.
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Missing choices in response")
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        raise ValueError("Missing message in choices[0]")
    content = msg.get("content")
    if not isinstance(content, str):
        raise ValueError("Missing content in message")
    return json.loads(content)


def _coerce_result(data: Any) -> LLMCorrectnessResult:
    """
    Tests expect:
    - ValueError for non-dict inputs
    - ValueError for out-of-range score/confidence
    - Accept either direct payload OR OpenAI-shaped wrapper
    """
    if not isinstance(data, dict):
        raise ValueError("Result must be an object")

    payload: Any
    if "choices" in data:
        payload = _extract_openai_content(data)
    else:
        payload = data

    if not isinstance(payload, dict):
        raise ValueError("Payload must be an object")

    score = payload.get("score")
    explanation = payload.get("explanation")
    confidence = payload.get("confidence")

    if not isinstance(score, int):
        raise ValueError("score must be int")
    if not (1 <= score <= 5):
        raise ValueError("score out of range (1-5)")

    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation must be non-empty string")

    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be number")
    confidence_f = float(confidence)
    if not (0.0 <= confidence_f <= 1.0):
        raise ValueError("confidence out of range (0-1)")

    return LLMCorrectnessResult(score=score, explanation=explanation, confidence=confidence_f)


async def _judge_correctness_llm_async(request: PredictRequest) -> LLMCorrectnessResult:
    endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:8001/v1/chat/completions")
    timeout_s = float(os.getenv("LLM_TIMEOUT_S", "5.0"))

    payload = {
        "model": os.getenv("LLM_MODEL", "gpt-4o-mini"),
        "messages": [
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": _build_correctness_prompt(request)},
        ],
        "temperature": 0.0,
    }

    async_client_factory = cast(Any, httpx.AsyncClient)
    async with async_client_factory(timeout=timeout_s) as client:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return _coerce_result(data)


def judge_correctness_llm(request: PredictRequest) -> LLMCorrectnessResult:
    """
    IMPORTANT: This function must be SYNC because tests call it without `await`,
    even inside an async test function.

    We still use AsyncClient (so test monkeypatching works) by running the async
    implementation inside a dedicated thread.
    """
    def _run() -> LLMCorrectnessResult:
        return asyncio.run(_judge_correctness_llm_async(request))

    with ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(_run).result()
