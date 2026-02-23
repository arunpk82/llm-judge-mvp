from __future__ import annotations

from llm_judge.rules.registry import RULE_REGISTRY, get_rule


def test_registry_contains_correctness_rule() -> None:
    assert "correctness.basic" in RULE_REGISTRY


def test_get_rule_instantiates_correctness_rule() -> None:
    rule = get_rule("correctness.basic")
    assert getattr(rule, "rule_id", "") == "correctness.basic"
