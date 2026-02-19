from __future__ import annotations

from llm_judge.deterministic_judge import DeterministicJudge
from llm_judge.schemas import Message, PredictRequest


def _req(user: str, candidate: str, rubric_id: str = "chat_quality") -> PredictRequest:
    return PredictRequest(
        conversation=[Message(role="user", content=user)],
        candidate_answer=candidate,
        rubric_id=rubric_id,
    )


def test_deterministic_judge_uses_rubric_policy_fail_if_any_dimension_lte() -> None:
    # Rubric policy sets fail_if_any_dimension_lte: 2
    # Make tone score == 2 via excessive shouting (see scorer.py)
    user = "How do I reset my router?"
    candidate = (
        "RESET THE ROUTER BY UNPLUGGING IT FOR 30 SECONDS AND PLUGGING IT BACK IN. "
        "THEN WAIT 2 MINUTES FOR IT TO RECONNECT."
    )
    out = DeterministicJudge().evaluate(_req(user, candidate))
    assert out.scores["tone"] == 2
    assert out.decision == "fail"


def test_deterministic_judge_passes_clear_relevant_polite_answer() -> None:
    user = "How do I reset my router?"
    candidate = """You can try this:\n\n1. Unplug the router for 30 seconds.\n2. Plug it back in and wait 2 minutes.\n\nIf that doesn't help, share the model number."""
    out = DeterministicJudge().evaluate(_req(user, candidate))
    assert out.decision == "pass"
    assert out.scores["relevance"] >= 3
    assert out.scores["clarity"] >= 3


def test_unknown_rubric_returns_clean_failure() -> None:
    out = DeterministicJudge().evaluate(_req("Hi", "Hello", rubric_id="does_not_exist"))
    assert out.decision == "fail"
    assert "unknown_rubric" in out.flags
