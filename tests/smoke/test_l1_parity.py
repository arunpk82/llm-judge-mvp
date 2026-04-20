"""Smoke test: L1 substring-match clearance parity with Exp 29B.

Source of truth for the per-sentence expectations below:

- Exp 29B (experiments/exp29b_l1_rules_tightened.py): the L1 rule
  tightening run whose contract is "total cleared ~19-21 on RAGTruth-50,
  vs_hallucinated_gt == 0". This test encodes the clearance branches
  Exp 29B validates, on a hand-built fixture that runs in <1s.
- 2026-04-20 measurement run (commit 783b13e on master): full
  RAGTruth-50 calibrate-l1 cleared 19/303 sentences, 0 false-positive
  clears. results/ragtruth50_results.json sentence_level_metrics.by_layer.L1
  under HALLUCINATION_LAYERS=l1.
- src/llm_judge/calibration/hallucination.py :: _l1_substring_match
  (three clearance branches: substring / SequenceMatcher>0.85 / Jaccard>0.80).

Why: the full RAGTruth-50 calibration run takes ~17 minutes. This
smoke locks each branch of L1's clearance logic on a tiny fixture so
any regression in _l1_substring_match fails every PR in <1s.

If a case fails: first rerun `make calibrate-l1` on master and
compare sentence_level_metrics.by_layer.L1. If the aggregate clearance
count has genuinely shifted, update the expectations below AND the
source-of-truth note. If the aggregate is unchanged, the L1
implementation has drifted and must be fixed rather than the test.
"""

from __future__ import annotations

import pytest

from llm_judge.calibration.hallucination import _l1_substring_match

SOURCE_FULL = (
    "J. Paul Getty was born on December 15, 1892. "
    "He founded the Getty Oil Company in 1942. "
    "The company was headquartered in Los Angeles, California. "
    "Getty became a billionaire in 1957, the first American to be "
    "worth over a billion dollars."
)

# Pre-split by hand. L1 receives source_sentences as an already-split
# list from its caller in production, so calling _l1_substring_match
# directly with a hand-built list exercises L1's clearance logic
# without paying the ~10s spaCy cold-load cost. Splitter behavior is
# covered separately by tests that target _split_sentences.
SOURCE_SENTENCES = [
    "J. Paul Getty was born on December 15, 1892.",
    "He founded the Getty Oil Company in 1942.",
    "The company was headquartered in Los Angeles, California.",
    (
        "Getty became a billionaire in 1957, the first American "
        "to be worth over a billion dollars."
    ),
]


# (sentence_id, sentence_text, expected_clear, rationale)
L1_CASES: list[tuple[str, str, bool, str]] = [
    (
        "exact_substring_1",
        "J. Paul Getty was born on December 15, 1892.",
        True,
        "exact substring of source — substring branch fires",
    ),
    (
        "exact_substring_2",
        "He founded the Getty Oil Company in 1942.",
        True,
        "exact substring of source — substring branch fires",
    ),
    (
        "paraphrase_not_cleared",
        "Getty Oil Company was established during 1942 by the founder.",
        False,
        "paraphrase; Jaccard ~0.39 < 0.80 and SequenceMatcher ~0.43 < 0.85",
    ),
    (
        "numeric_mismatch",
        "Getty became a billionaire in 1970.",
        False,
        (
            "1957 -> 1970; Jaccard ~0.31 vs the long source sentence, "
            "SequenceMatcher ~0.52 — both under threshold"
        ),
    ),
    (
        "entity_mismatch",
        "He founded the Apple Corporation in 1942.",
        False,
        (
            "Getty Oil Company -> Apple Corporation; Jaccard 0.5, "
            "SequenceMatcher ~0.57 — both under threshold"
        ),
    ),
    (
        "abbreviation_edge_case",
        "J. Paul Getty",
        True,
        (
            "short abbreviation fragment containing periods. NOT a test "
            "of the regex-vs-spaCy splitter (this test bypasses "
            "_split_sentences by hand-building SOURCE_SENTENCES); it "
            "locks in _l1_substring_match's CURRENT behavior on "
            "short substring inputs. 'J. Paul Getty' is a substring "
            "of the normalized source, so the substring branch fires "
            "and returns True. A future change that adds a length "
            "filter inside _l1_substring_match, or changes how "
            "normalization handles the period after 'J', would flip "
            "this to False and fail this case loudly. Splitter "
            "behavior (ADR-0014, hallucination.py:47) is covered by "
            "tests that target _split_sentences directly"
        ),
    ),
]


@pytest.mark.parametrize(
    "sentence_id,sentence,expected_clear,rationale",
    L1_CASES,
    ids=[c[0] for c in L1_CASES],
)
def test_l1_clearance_parity(
    sentence_id: str,
    sentence: str,
    expected_clear: bool,
    rationale: str,
) -> None:
    got = _l1_substring_match(sentence, SOURCE_SENTENCES, SOURCE_FULL)
    assert got is expected_clear, (
        f"[{sentence_id}] L1 clearance drift: got={got}, "
        f"expected={expected_clear}. Rationale: {rationale}"
    )
