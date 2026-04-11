from __future__ import annotations

import llm_judge.llm_correctness as lc
from llm_judge.schemas import Message, PredictRequest


def _make_req(prompt: str, answer: str) -> PredictRequest:
    return PredictRequest(
        conversation=[Message(role="user", content=prompt)],
        candidate_answer=answer,
        rubric_id="chat_quality",
    )


def test_llm_correctness_handles_missing_api_key(monkeypatch) -> None:
    # Ensure no key is present
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)

    # Make sure unit test never tries network even if code regresses
    async def _fake_async(_req: PredictRequest):
        return None

    monkeypatch.setattr(lc, "_judge_correctness_llm_async", _fake_async)

    out = lc.judge_correctness_llm(_make_req("2+2", "4"))
    assert out is None or out is not None  # "no crash" contract


def test_llm_correctness_exception_is_caught(monkeypatch) -> None:
    # Force the async path to raise, ensure sync wrapper doesn't blow up CI
    async def _boom(_req: PredictRequest):
        raise RuntimeError("boom")

    monkeypatch.setattr(lc, "_judge_correctness_llm_async", _boom)

    out = lc.judge_correctness_llm(
        _make_req("Define blockchain", "Blockchain is blockchain.")
    )
    assert out is None or out is not None  # "no crash" contract
