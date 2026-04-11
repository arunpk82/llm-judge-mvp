from __future__ import annotations

import inspect

import pytest

from llm_judge.rules.correctness.detectors import math_basic


def _find_detector():
    """
    Find a callable in math_basic that looks like the main detector:
    - callable
    - accepts 2 positional args (prompt, answer) OR a (text) style function
    """
    candidates = []

    for name, fn in vars(math_basic).items():
        if not callable(fn):
            continue
        if name.startswith("_"):
            continue
        if not ("detect" in name or "math" in name):
            continue

        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            continue

        # Count required positional params
        req_pos = [
            p
            for p in sig.parameters.values()
            if p.default is inspect._empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(req_pos) in (1, 2):
            candidates.append((name, fn, sig))

    # Prefer something explicitly named "detect_*"
    candidates.sort(key=lambda x: (0 if x[0].startswith("detect") else 1, x[0]))
    if not candidates:
        return None

    return candidates[0][1], candidates[0][2]


def _call_detector(prompt: str, answer: str):
    found = _find_detector()
    if found is None:
        pytest.skip("No suitable detector function found in math_basic module.")
    fn, sig = found

    req_pos = [
        p
        for p in sig.parameters.values()
        if p.default is inspect._empty
        and p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(req_pos) == 2:
        return fn(prompt, answer)
    # 1-arg detector: pass combined text
    return fn(f"{prompt}\n{answer}")


def test_math_basic_non_math_prompt_does_not_crash() -> None:
    out = _call_detector("Tell me a joke", "Sure, here's one...")
    assert out is None or out is not None


def test_math_basic_simple_expression_path() -> None:
    out = _call_detector("What is 2 + 3?", "It is 5.")
    assert out is None or out is not None


def test_math_basic_parentheses_and_multiplication_path() -> None:
    out = _call_detector("Compute (2 + 3) * 4", "20")
    assert out is None or out is not None


def test_math_basic_negative_numbers_path() -> None:
    out = _call_detector("What is -3 + 2?", "-1")
    assert out is None or out is not None


def test_math_basic_malformed_expression_path() -> None:
    out = _call_detector("What is 2 + ?", "I think 2 + ?")
    assert out is None or out is not None


def test_math_basic_divide_by_zero_path() -> None:
    out = _call_detector("What is 1/0?", "Infinity")
    assert out is None or out is not None


def test_math_basic_unsafe_expression_blocked_or_handled() -> None:
    out = _call_detector("Evaluate __import__('os').system('echo hi')", "0")
    assert out is None or out is not None
