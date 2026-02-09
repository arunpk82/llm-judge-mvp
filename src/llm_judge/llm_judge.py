import json
import os
from typing import Final

import httpx

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import PredictRequest, PredictResponse

_DEFAULT_BASE_URL: Final[str] = "https://api.openai.com/v1"
_DEFAULT_MODEL: Final[str] = "gpt-4.1-mini"
_DEFAULT_TIMEOUT_S: Final[float] = 2.0


def _build_prompt(request: PredictRequest) -> str:
    conversation = "\n".join([f"{m.role.upper()}: {m.content}" for m in request.conversation])
    return (
        """
        You are an evaluation engine. Score the candidate answer for a chatbot using these dimensions:
        - relevance, clarity, correctness, tone

        Return ONLY valid JSON with these fields:
        - decision: "pass" or "fail"
        - overall_score: float 0..5
        - scores: object with integer scores 1..5 for relevance/clarity/correctness/tone
        - confidence: float 0..1
        - flags: array of strings
        - explanations: object with string explanations for each dimension

        Conversation:
        {conversation}

        Candidate answer:
        {candidate_answer}
        """.strip()
    ).format(conversation=conversation, candidate_answer=request.candidate_answer)


class LLMJudge(JudgeEngine):
    """
    Synchronous judge to match JudgeEngine.evaluate() -> PredictResponse (non-async).
    """

    def __init__(self, timeout_ms: int | None = None) -> None:
        if timeout_ms is not None:
            self._timeout_s = max(timeout_ms, 1) / 1000.0
        else:
            self._timeout_s = float(os.getenv("LLM_TIMEOUT_S", str(_DEFAULT_TIMEOUT_S)))

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        # Test/dev simulation hook
        if os.getenv("SIMULATE_LLM_TIMEOUT") == "1":
            raise TimeoutError("Simulated LLM timeout")

        base_url = os.getenv("LLM_API_BASE_URL", _DEFAULT_BASE_URL).rstrip("/")
        api_key = os.getenv("LLM_API_KEY")
        model = os.getenv("LLM_MODEL", _DEFAULT_MODEL)

        if not api_key:
            raise RuntimeError("LLM_API_KEY is not set")

        prompt = _build_prompt(request)

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a strict JSON-only evaluator."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}/chat/completions"

        # Correct typing/signature for httpx.Client
        with httpx.Client(timeout=self._timeout_s) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"Unexpected LLM response shape: {e}") from e

        try:
            parsed = json.loads(content)
        except Exception as e:
            raise RuntimeError(f"LLM output was not valid JSON: {e}") from e

        return PredictResponse(**parsed)
