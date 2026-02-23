from __future__ import annotations

from llm_judge.schemas import Message, PredictRequest
from llm_judge.scorer import score_candidate


def test_scorer_hard_fails_on_correctness_definition_sanity() -> None:
    req = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="Blockchain is blockchain.",
        rubric_id="chat_quality",
    )
    resp = score_candidate(req)

    base_flags = {x.split(":", 1)[0] for x in resp.flags}
    assert "correctness.definition_sanity" in base_flags
    assert resp.decision == "fail"