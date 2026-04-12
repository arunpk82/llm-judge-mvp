from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias, Union, runtime_checkable

from llm_judge.rules.types import RuleContext, RuleResult

# Function-style rule: (ctx, params) -> RuleResult
RuleFn: TypeAlias = Callable[[RuleContext, dict[str, Any]], RuleResult]


@runtime_checkable
class RuleObj(Protocol):
    """
    Instance protocol for rules.

    We keep this intentionally flexible:
    - production rules may accept (ctx, params)
    - test dummy rules may accept (ctx) only
    """

    def apply(self, ctx: RuleContext, *args: Any, **kwargs: Any) -> RuleResult: ...


@runtime_checkable
class RuleClass(Protocol):
    """Class protocol for rules that can be instantiated with no args."""

    def __call__(self) -> RuleObj: ...


# type[Any] instead of type[RuleObj] so mypy accepts arbitrary rule classes
# (including test dummies) at call sites. Runtime safety is enforced by the
# hasattr(obj, "apply") check inside _decorator.
RuleImpl: TypeAlias = Union[RuleFn, type[Any]]


@dataclass(frozen=True)
class FunctionRule:
    rule_id: str
    fn: RuleFn

    def apply(
        self, ctx: RuleContext, params: dict[str, Any] | None = None
    ) -> RuleResult:
        return self.fn(ctx, params or {})


RULE_REGISTRY: dict[str, RuleObj] = {}


# Callable[[Any], Any] so @register(...) works on any class without mypy complaining
# that the class doesn't structurally satisfy type[RuleObj].
def register(rule_id: str) -> Callable[[Any], Any]:
    """
    Decorator-based registration.

    Supports:
    - @register("x.y") on a function(ctx, params)->RuleResult
    - @register("x.y") on a class with apply(ctx) OR apply(ctx, params)
    """

    def _decorator(impl: RuleImpl) -> RuleImpl:
        if isinstance(impl, type):
            obj = impl()  # instantiate class rule
            # runtime safety: must have apply()
            if not hasattr(obj, "apply"):
                raise TypeError(
                    f"Rule class '{impl.__name__}' must define apply(ctx, ...)"
                )
            RULE_REGISTRY[rule_id] = obj
        else:
            RULE_REGISTRY[rule_id] = FunctionRule(rule_id=rule_id, fn=impl)
        return impl

    return _decorator


def get_rule(rule_id: str) -> RuleObj:
    try:
        return RULE_REGISTRY[rule_id]
    except KeyError as e:
        raise KeyError(f"Unknown rule_id: {rule_id}") from e


# Best-effort bootstrap import (never crash import on rule module issues)
try:
    import llm_judge.rules.bootstrap  # noqa: F401
except Exception:
    pass
