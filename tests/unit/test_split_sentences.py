"""Tests for spaCy-backed sentence splitting (ADR-0014).

Specifically exercises the abbreviation cases that the legacy regex
splitter mishandled (RAGTruth case 6 false positive).
"""

from __future__ import annotations

from llm_judge.calibration.hallucination import _split_sentences


class TestAbbreviationHandling:
    """Cases the legacy regex splitter got wrong."""

    def test_initials_in_proper_name_stay_one_sentence(self) -> None:
        # The regression case from RAGTruth case 6.
        text = "J. Paul Getty was the founder. He died in 1976."
        sents = _split_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "J. Paul Getty was the founder."
        assert sents[1] == "He died in 1976."

    def test_title_abbreviation_stays_one_sentence(self) -> None:
        text = "Dr. Smith examined the patient. The patient recovered fully."
        sents = _split_sentences(text)
        assert len(sents) == 2
        assert sents[0] == "Dr. Smith examined the patient."

    def test_us_state_abbreviation(self) -> None:
        text = "Washington, D. C. is the capital. It has many monuments."
        sents = _split_sentences(text)
        assert len(sents) == 2
        assert "D. C." in sents[0]


class TestBasicSplitting:
    """Cases that should work with both spaCy and the regex fallback."""

    def test_two_simple_sentences(self) -> None:
        text = "The cat sat on the mat. The dog ran fast around the yard."
        sents = _split_sentences(text)
        assert len(sents) == 2

    def test_empty_input_returns_empty(self) -> None:
        assert _split_sentences("") == []
        assert _split_sentences("   ") == []

    def test_short_fragments_filtered(self) -> None:
        # Per existing behavior: fragments under 11 chars are dropped.
        text = "Hi. The cat sat on the mat is what happened today."
        sents = _split_sentences(text)
        # "Hi." is 3 chars; should be filtered.
        assert all(len(s) > 10 for s in sents)

    def test_question_and_exclamation_split(self) -> None:
        text = "Is this working? Yes it is working correctly today."
        sents = _split_sentences(text)
        assert len(sents) == 2
