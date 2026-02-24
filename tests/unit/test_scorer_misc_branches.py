from __future__ import annotations

import llm_judge.scorer as scorer
from llm_judge.schemas import Message, PredictRequest


def test_scorer_unknown_rubric_returns_fail(monkeypatch) -> None:
    def _raise(_: str) -> object:
        raise ValueError("missing rubric")

    monkeypatch.setattr(scorer, "get_rubric", _raise)

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi",
        rubric_id="does_not_exist",
    )
    resp = scorer.score_candidate(req)
    assert resp.decision == "fail"
    assert "unknown_rubric" in resp.flags
