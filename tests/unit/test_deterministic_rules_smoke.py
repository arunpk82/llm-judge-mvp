from llm_judge.rubric_store import get_rubric
from llm_judge.rules.registry import get_rule
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import Message, PredictRequest


def test_new_rules_execute_and_return_flags() -> None:
    rubric = get_rubric("chat_quality")

    req = PredictRequest(
        conversation=[Message(role="user", content="What is 2+2?")],
        candidate_answer="It is 5. It is 5. It is 5.",
        rubric_id="chat_quality",
    )
    ctx = RuleContext(request=req, rubric=rubric)

    # correctness.basic
    rr = get_rule("correctness.basic").apply(ctx, {})
    assert rr is not None

    # repetition
    rr2 = get_rule("quality.repetition_basic").apply(ctx, {})
    assert rr2 is not None

    # definition sanity (should not crash even if it does not flag)
    rr3 = get_rule("correctness.definition_sanity").apply(ctx, {})
    assert rr3 is not None

    # nonsense (should not crash)
    rr4 = get_rule("quality.nonsense_basic").apply(ctx, {})
    assert rr4 is not None
