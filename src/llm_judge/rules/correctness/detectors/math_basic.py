from __future__ import annotations

import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class CorrectnessSignal:
    flags: list[str]


_ALLOWED_BINOPS: dict[type[ast.operator], Callable[..., float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], Callable[..., float]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _safe_eval_expr(expr: str) -> Optional[float]:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_BINOPS:
            left = _eval(n.left)
            right = _eval(n.right)
            fn = _ALLOWED_BINOPS[type(n.op)]
            return float(fn(left, right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_UNARYOPS:
            operand = _eval(n.operand)
            fn = _ALLOWED_UNARYOPS[type(n.op)]
            return float(fn(operand))
        raise ValueError("Unsafe expression node")

    try:
        return _eval(node)
    except Exception:
        return None


# Matches expressions like: -4 - -11, 29 * 18, 2+2, 10 / -4, etc.
# Operands may have an optional leading sign; operator is a single char.
_MATH_EXPR_RE = re.compile(
    r"(?P<expr>"
    r"[-+]?\s*\d+(?:\.\d+)?"                          # first operand
    r"(?:"
    r"\s*[\+\-\*\/%]\s*"                               # binary operator
    r"[-+]?\s*\d+(?:\.\d+)?"                           # subsequent operand
    r")+"
    r")"
)

# Matches a number after semantic anchors, or the last number in text
_ANSWER_NUMBER_RE = re.compile(
    r"(?:=\s*|(?:is|equals|are)\s+)([-+]?\d+(?:\.\d+)?)"
    r"|"
    r"([-+]?\d+(?:\.\d+)?)\s*[.!]?\s*$"
)
_NUMBER_IN_TEXT_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _extract_math_expr(user_text: str) -> Optional[str]:
    m = _MATH_EXPR_RE.search(user_text)
    if not m:
        return None
    # Remove internal spaces so AST parses cleanly: "-4 - -11" -> "-4--11"
    expr = re.sub(r"\s+", "", m.group("expr"))
    return expr


def _extract_answer_number(text: str) -> Optional[float]:
    m = _ANSWER_NUMBER_RE.search(text)
    if m:
        val = m.group(1) or m.group(2)
        try:
            return float(val)
        except Exception:
            pass

    # Fallback: last number in text
    all_nums = _NUMBER_IN_TEXT_RE.findall(text)
    if not all_nums:
        return None
    try:
        return float(all_nums[-1])
    except Exception:
        return None


def detect_math_incorrect(prompt: str, answer: str) -> Optional[CorrectnessSignal]:
    expr = _extract_math_expr(prompt)
    if not expr:
        return None

    expected = _safe_eval_expr(expr)
    if expected is None:
        return None

    got = _extract_answer_number(answer)
    if got is None:
        return None

    tol = 1e-9
    if abs(got - expected) <= tol:
        return None

    return CorrectnessSignal(flags=["correctness.math_incorrect"])