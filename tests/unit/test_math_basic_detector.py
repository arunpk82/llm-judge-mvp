from __future__ import annotations

from llm_judge.rules.correctness.detectors.math_basic import detect_math_incorrect


def test_math_basic_detects_incorrect_addition() -> None:
    sig = detect_math_incorrect("What is 2+2?", "It is 5.")
    assert sig is not None
    assert "correctness.math_incorrect" in sig.flags


def test_math_basic_allows_correct_addition() -> None:
    sig = detect_math_incorrect("What is 2+2?", "2+2 is 4.")
    assert sig is None


def test_math_basic_ignores_non_math_questions() -> None:
    sig = detect_math_incorrect(
        "Define blockchain", "Blockchain is a distributed ledger."
    )
    assert sig is None


def test_math_basic_detects_incorrect_multiplication() -> None:
    sig = detect_math_incorrect("What is 29 * 18?", "It is 500.")
    assert sig is not None
    assert "correctness.math_incorrect" in sig.flags
