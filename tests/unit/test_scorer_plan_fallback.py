from __future__ import annotations

from types import SimpleNamespace

import llm_judge.scorer as scorer
from llm_judge.schemas import Message, PredictRequest


def test_scorer_hard_fails_on_strong_correctness(monkeypatch) -> None:
    fake_policy = SimpleNamespace(pass_if_overall_score_gte=3.0, fail_if_any_dimension_lte=2)
    fake_rubric = SimpleNamespace(
        rubric_id="chat_quality", version="v1", decision_policy=fake_policy, rules=None
    )

    monkeypatch.setattr(scorer, "get_rubric", lambda rubric_id: fake_rubric)

    # Inject a strong correctness flag to hit the strong-correctness hard-fail branch
    monkeypatch.setattr(
        scorer,
        "_apply_rubric_rules",
        lambda request, rubric: ["correctness.definition_sanity:strong"],
    )

    req = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="Blockchain is blockchain.",
        rubric_id="chat_quality",
    )
    resp = scorer.score_candidate(req)
    assert "correctness.definition_sanity:strong" in resp.flags
    assert resp.decision == "fail"
