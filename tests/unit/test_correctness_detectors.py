from __future__ import annotations

from llm_judge.rules.correctness.detectors.known_facts import detect_known_fact_mismatch
from llm_judge.rules.correctness.detectors.math_basic import detect_math_incorrect
from llm_judge.rules.correctness.detectors.nonsense_pattern import detect_nonsense
from llm_judge.rules.correctness.detectors.unsafe_advice import detect_unsafe_advice


def test_detect_known_fact_mismatch_flags_wrong_capital() -> None:
    sig = detect_known_fact_mismatch("What is the capital of France?", "Berlin.")
    assert "correctness.known_fact_mismatch" in sig.flags


def test_detect_known_fact_mismatch_no_flag_when_contains_expected_token() -> None:
    sig = detect_known_fact_mismatch(
        "What is the capital of France?", "Paris is the capital of France."
    )
    assert sig.flags == []


def test_detect_unsafe_advice_flags_keywords() -> None:
    sig = detect_unsafe_advice("How can I do X?", "You can hack it by doing Y.")
    assert "correctness.unsafe_advice" in sig.flags


def test_detect_nonsense_flags_absurd_answer_with_low_overlap() -> None:
    sig = detect_nonsense("How do I reset my router?", "Bananas are yellow and grow in bunches.")
    assert "correctness.nonsense_detected" in sig.flags

def test_math_detector_flags_incorrect() -> None:
    sig = detect_math_incorrect("What is 2+2?", "5")
    assert sig is not None
    assert "correctness.math_incorrect" in sig.flags
    
def test_math_detector_no_signal_when_correct() -> None:
    sig = detect_math_incorrect("What is 2+2?", "4")
    assert sig is None