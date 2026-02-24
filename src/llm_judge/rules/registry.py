from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from llm_judge.rules.types import RuleContext, RuleResult

RuleFn = Callable[[RuleContext, dict[str, Any]], RuleResult]


@runtime_checkable
class Rule(Protocol):
    def apply(self, ctx: RuleContext, params: dict[str, Any]) -> RuleResult: ...
    def __call__(self, ctx: RuleContext, params: dict[str, Any]) -> RuleResult: ...


@dataclass(frozen=True)
class FunctionRule:
    rule_id: str
    fn: RuleFn

    def apply(self, ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
        return self.fn(ctx, params)

    def __call__(self, ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
        return self.fn(ctx, params)


RULE_REGISTRY: dict[str, Rule] = {}


def register(rule_id: str) -> Callable[[RuleFn], RuleFn]:
    def _decorator(fn: RuleFn) -> RuleFn:
        RULE_REGISTRY[rule_id] = FunctionRule(rule_id=rule_id, fn=fn)
        return fn

    return _decorator


def get_rule(rule_id: str) -> Rule:
    """Return a registered rule object with .apply() and callable behavior."""
    try:
        return RULE_REGISTRY[rule_id]
    except KeyError as e:
        raise KeyError(f"Unknown rule_id: {rule_id}") from e


# Ensure rule registration happens as soon as registry is imported.
# This keeps tests and runtime deterministic without requiring manual bootstrap calls.
try:
    import llm_judge.rules.bootstrap  # noqa: F401
except Exception:
    # Never crash import if a rule module has issues; tests will surface it.
    pass
