from __future__ import annotations

from types import SimpleNamespace

import llm_judge.scorer as scorer
from llm_judge.schemas import Message, PredictRequest


def test_scorer_uses_inline_rules_list(monkeypatch) -> None:
    # Fake rubric with inline rules to force scorer's inline branch.
    fake_policy = SimpleNamespace(pass_if_overall_score_gte=3.0, fail_if_any_dimension_lte=2)

    fake_rubric = SimpleNamespace(
        rubric_id="chat_quality",
        version="v1",
        decision_policy=fake_policy,
        # Inline rule list (dict form)
        rules=[
            {"id": "quality.nonsense_basic"},
            {"id": "correctness.definition_sanity"},
        ],
    )

    monkeypatch.setattr(scorer, "get_rubric", lambda rubric_id: fake_rubric)

    req = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="!!! @@ ### $$$ %%%",  # nonsense_basic should flag
        rubric_id="chat_quality",
    )
    resp = scorer.score_candidate(req)

    # Should include deterministic rule flags
    base = {f.split(":", 1)[0] for f in resp.flags}
    assert "quality.nonsense_basic" in base or "correctness.definition_sanity" in base


def test_scorer_hard_fails_on_strong_quality_nonsense(monkeypatch) -> None:
    # Force rubric fallback or inline rules doesn't matter; we will patch _apply_rubric_rules
    fake_policy = SimpleNamespace(pass_if_overall_score_gte=3.0, fail_if_any_dimension_lte=2)
    fake_rubric = SimpleNamespace(
        rubric_id="chat_quality", version="v1", decision_policy=fake_policy, rules=[]
    )

    monkeypatch.setattr(scorer, "get_rubric", lambda rubric_id: fake_rubric)

    # Inject a strong quality flag to hit the strong-quality hard-fail branch
    monkeypatch.setattr(
        scorer, "_apply_rubric_rules", lambda request, rubric: ["quality.nonsense_basic:strong"]
    )

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="ok",
        rubric_id="chat_quality",
    )
    resp = scorer.score_candidate(req)
    assert "quality.nonsense_basic:strong" in resp.flags
    assert resp.decision == "fail"
