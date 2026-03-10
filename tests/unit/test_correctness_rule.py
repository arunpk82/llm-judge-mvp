from __future__ import annotations

from llm_judge.rubric_store import get_rubric
from llm_judge.rules.base import RuleContext
from llm_judge.rules.correctness_basic import CorrectnessBasicRule
from llm_judge.schemas import Message, PredictRequest


def test_correctness_basic_rule_emits_flags() -> None:
    rubric = get_rubric("chat_quality")

    req = PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="It is 5.",
        rubric_id="chat_quality",
    )

    ctx = RuleContext(request=req, rubric=rubric)
    CorrectnessBasicRule().apply(ctx)

    assert any(f.startswith("correctness.") for f in ctx.flags)
    assert "correctness.math_incorrect" in ctx.flags
