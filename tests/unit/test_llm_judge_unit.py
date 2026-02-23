from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import llm_judge.llm_judge as lj
from llm_judge.schemas import Message, PredictRequest


@dataclass
class _FakeLLMResponse:
    # match what LLMJudge expects after parsing (or what it reads directly)
    decision: str = "pass"
    overall_score: float = 4.0
    scores: dict[str, int] | None = None
    flags: list[str] | None = None
    explanations: dict[str, str] | None = None
    confidence: float = 0.8

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = {"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4}
        if self.flags is None:
            self.flags = []
        if self.explanations is None:
            self.explanations = {
                "relevance": "ok",
                "clarity": "ok",
                "correctness": "ok",
                "tone": "ok",
            }


def _openai_compatible_success_payload() -> dict[str, Any]:
    """
    Return an OpenAI-compatible response object:
    {"choices":[{"message":{"content":"<json>"} }]}
    where <json> is the serialized PredictResponse-like dict.
    """
    # NOTE: use JSON string because llm_judge.py typically json.loads(message.content)
    return {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"decision":"pass","overall_score":4.0,'
                        '"scores":{"relevance":4,"clarity":4,"correctness":4,"tone":4},'
                        '"confidence":0.8,"flags":[],"explanations":{'
                        '"relevance":"ok","clarity":"ok","correctness":"ok","tone":"ok"}}'
                    )
                }
            }
        ]
    }


def test_llm_judge_evaluate_happy_path_without_network(monkeypatch) -> None:
    """
    Drives llm_judge.py evaluate() through its httpx.Client path, without network.
    """

    monkeypatch.setenv("LLM_API_KEY", "dummy")

    # If your implementation ever adds internal helper functions, this still supports them.
    def _fake_call(*args: Any, **kwargs: Any) -> Any:
        return _FakeLLMResponse()

    for attr in ("_call_llm", "_invoke_llm", "_judge_with_llm", "_request_llm"):
        if hasattr(lj, attr):
            monkeypatch.setattr(lj, attr, _fake_call)
            break
    else:
        # Stub httpx.Client used as a context manager
        if hasattr(lj, "httpx"):

            class _FakeResponse:
                status_code = 200
                
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> Any:
                    return _openai_compatible_success_payload()

                @property
                def text(self) -> str:
                    return "ok"

            class _FakeClient:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    pass

                def __enter__(self) -> "_FakeClient":
                    return self

                def __exit__(self, exc_type, exc, tb) -> None:
                    return None

                def post(self, *args: Any, **kwargs: Any) -> Any:
                    return _FakeResponse()

            class _FakeHTTPX:
                Client = _FakeClient

            monkeypatch.setattr(lj, "httpx", _FakeHTTPX)

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi there!",
        rubric_id="chat_quality",
    )

    out = lj.LLMJudge().evaluate(req)
    assert out.decision in {"pass", "fail"}
    assert isinstance(out.scores, dict)
    assert "relevance" in out.scores


def test_llm_judge_handles_non_200_response(monkeypatch) -> None:
    """
    Covers error branch where the LLM endpoint returns non-200 or invalid payload.
    """

    monkeypatch.setenv("LLM_API_KEY", "dummy")

    if hasattr(lj, "httpx"):

        class _FakeResponse:
            status_code = 500  # <-- must be non-200 to hit the error path
            
            def raise_for_status(self) -> None:
                raise RuntimeError("HTTP 500")

            def json(self) -> Any:
                raise ValueError("no json")

            @property
            def text(self) -> str:
                return "server error"

        class _FakeClient:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __enter__(self) -> "_FakeClient":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

            def post(self, *args: Any, **kwargs: Any) -> Any:
                return _FakeResponse()

        class _FakeHTTPX:
            Client = _FakeClient

        monkeypatch.setattr(lj, "httpx", _FakeHTTPX)

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi there!",
        rubric_id="chat_quality",
    )

    with pytest.raises(Exception):
        lj.LLMJudge().evaluate(req)