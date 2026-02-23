from __future__ import annotations

import pytest

from llm_judge.deterministic_judge import DeterministicJudge
from llm_judge.llm_judge import LLMJudge, _build_prompt
from llm_judge.schemas import Message, PredictRequest


def test_deterministic_judge_smoke() -> None:
    req = PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="2+2 is 4.",
        rubric_id="chat_quality",
    )
    resp = DeterministicJudge().evaluate(req)
    assert resp.decision in {"pass", "fail"}
    assert isinstance(resp.scores, dict)
    assert isinstance(resp.flags, list)


def test_llm_judge_build_prompt_contains_conversation_and_candidate() -> None:
    req = PredictRequest(
        conversation=[Message(role="user", content="Hello there")],
        candidate_answer="Hi!",
        rubric_id="chat_quality",
    )
    prompt = _build_prompt(req)
    assert "USER: Hello there" in prompt
    assert "Hi!" in prompt


def test_llm_judge_requires_api_key(monkeypatch) -> None:
    # Ensure no API key is set
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    req = PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="4",
        rubric_id="chat_quality",
    )

    with pytest.raises(RuntimeError):
        LLMJudge().evaluate(req)