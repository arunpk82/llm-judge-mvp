from __future__ import annotations

import re

from ..base import CorrectnessSignal

_KNOWN_FACTS = [
    (re.compile(r"\bcapital of france\b", re.IGNORECASE), ["paris"]),
    (re.compile(r"\bcapital of japan\b", re.IGNORECASE), ["tokyo"]),
    (re.compile(r"\bwhat does http stand for\b", re.IGNORECASE), ["hypertext transfer protocol"]),
]


def detect_known_fact_mismatch(question: str, answer: str) -> CorrectnessSignal:
    q = question.strip()
    a = answer.strip().lower()

    for pattern, must_contain_any in _KNOWN_FACTS:
        if not pattern.search(q):
            continue

        ok = any(token in a for token in must_contain_any)
        if not ok:
            return CorrectnessSignal(flags=["correctness.known_fact_mismatch"])

        return CorrectnessSignal(flags=[])

    return CorrectnessSignal(flags=[])
