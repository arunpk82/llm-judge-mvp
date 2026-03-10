from __future__ import annotations

import pytest

from llm_judge.llm_judge import LLMJudge, _build_prompt
from llm_judge.schemas import Message, PredictRequest


def test_build_prompt_smoke() -> None:
    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi",
        rubric_id="chat_quality",
    )
    prompt = _build_prompt(req)
    assert "USER:" in prompt
    assert "Candidate answer:" in prompt


def test_llm_judge_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    req = PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="4",
        rubric_id="chat_quality",
    )

    with pytest.raises(RuntimeError):
        LLMJudge().evaluate(req)
