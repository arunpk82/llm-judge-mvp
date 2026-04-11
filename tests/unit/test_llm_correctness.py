import json
from typing import Any

import pytest

from llm_judge.llm_correctness import _coerce_result, judge_correctness_llm
from llm_judge.schemas import Message, PredictRequest


def _req() -> PredictRequest:
    return PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="It's 4.",
        rubric_id="chat_quality",
    )


def test_coerce_result_happy_path() -> None:
    obj: Any = {"score": 4, "explanation": "Correct.", "confidence": 0.9}
    res = _coerce_result(obj)
    assert res.score == 4
    assert res.explanation == "Correct."
    assert res.confidence == 0.9


def test_coerce_result_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        _coerce_result(["not", "a", "dict"])


@pytest.mark.parametrize(
    "obj",
    [
        {"score": 0, "explanation": "x", "confidence": 0.5},  # score too low
        {"score": 6, "explanation": "x", "confidence": 0.5},  # score too high
        {"score": 3, "explanation": "x", "confidence": -0.1},  # confidence too low
        {"score": 3, "explanation": "x", "confidence": 1.1},  # confidence too high
    ],
)
def test_coerce_result_rejects_out_of_range(obj: Any) -> None:
    with pytest.raises(ValueError):
        _coerce_result(obj)


@pytest.mark.anyio
async def test_judge_correctness_llm_parses_openai_shape(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    # monkeypatch.setenv("OPENAI_API_BASE", "https://example.test")  # optional

    class _FakeResp:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            payload = {
                "score": 5,
                "explanation": "Math checks out.",
                "confidence": 0.95,
            }
            return {"choices": [{"message": {"content": json.dumps(payload)}}]}

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "_FakeClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(self, url: str, json: Any) -> _FakeResp:
            return _FakeResp()

    import llm_judge.llm_correctness as mod

    monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeClient)

    res = judge_correctness_llm(_req())
    assert res is not None
    assert res.score == 5
