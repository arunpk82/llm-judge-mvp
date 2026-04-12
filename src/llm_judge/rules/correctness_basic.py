from __future__ import annotations

from typing import Any, Literal

from llm_judge.rules.correctness.detectors.math_basic import detect_math_incorrect
from llm_judge.rules.registry import register
from llm_judge.rules.types import Flag, RuleContext, RuleResult

Severity = Literal["weak", "strong"]


class CorrectnessBasicRule:
    """
    Backwards-compatible rule object expected by existing tests.

    Behavior:
    - emits string flags into ctx.flags (legacy pattern)
    - also returns RuleResult with structured flags (new pattern)
    """

    rule_id = "correctness.basic"

    def apply(
        self, ctx: RuleContext, params: dict[str, Any] | None = None
    ) -> RuleResult:
        params = params or {}
        user_text = (ctx.user_text or "").strip()
        candidate = (ctx.candidate or "").strip()

        emitted: list[Flag] = []

        if not candidate:
            ctx.flags.append("correctness.empty_answer")
            emitted.append(
                Flag(
                    id="correctness.empty_answer",
                    severity="strong",
                    details={"reason": "empty_answer"},
                    evidence=[],
                )
            )
            return RuleResult(flags=emitted)

        # Delegate to the math detector (handles +, -, *, /, %, ** and all phrasings)
        sig = detect_math_incorrect(user_text, candidate)
        if sig is not None:
            for flag_id in sig.flags:
                ctx.flags.append(flag_id)
                emitted.append(
                    Flag(
                        id=flag_id,
                        severity="strong",
                        details={},
                        evidence=[candidate[:160]],
                    )
                )

        return RuleResult(flags=emitted)


@register("correctness.basic")
def correctness_basic(ctx: RuleContext, params: dict[str, Any]) -> RuleResult:
    return CorrectnessBasicRule().apply(ctx, params)
