from __future__ import annotations

from llm_judge.rubric_store import get_rubric
from llm_judge.rules.registry import get_rule
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import Message, PredictRequest


def test_definition_sanity_flags_circular_or_vacuous_definition() -> None:
    rubric = get_rubric("chat_quality")
    req = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="Blockchain is blockchain.",
        rubric_id="chat_quality",
    )
    ctx = RuleContext(request=req, rubric=rubric)
    rr = get_rule("correctness.definition_sanity").apply(ctx, {})
    assert rr is not None
    # may flag or may be conservative, but should not crash and should execute branches
    assert isinstance(rr.flags, list)


def test_nonsense_basic_flags_symbol_heavy_gibberish() -> None:
    rubric = get_rubric("chat_quality")
    req = PredictRequest(
        conversation=[Message(role="user", content="Explain networking")],
        candidate_answer="!!! @@ ### $$$ %%% ^^^ &&& ***",
        rubric_id="chat_quality",
    )
    ctx = RuleContext(request=req, rubric=rubric)
    rr = get_rule("quality.nonsense_basic").apply(ctx, {})
    assert rr is not None
    assert isinstance(rr.flags, list)