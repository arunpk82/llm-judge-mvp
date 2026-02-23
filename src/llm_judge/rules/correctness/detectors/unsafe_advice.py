from __future__ import annotations

import re

from ..base import CorrectnessSignal

_UNSAFE_PATTERNS = [
    re.compile(r"\bmicrowave\b", re.IGNORECASE),
    re.compile(r"\bhack\b", re.IGNORECASE),
    re.compile(r"\bexploit\b", re.IGNORECASE),
]


def detect_unsafe_advice(question: str, answer: str) -> CorrectnessSignal:
    _ = question
    a = answer.strip()

    if any(p.search(a) for p in _UNSAFE_PATTERNS):
        return CorrectnessSignal(flags=["correctness.unsafe_advice"])

    return CorrectnessSignal(flags=[])
