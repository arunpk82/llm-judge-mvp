from __future__ import annotations

import re

from ..base import CorrectnessSignal

_ABSURD_MARKERS = [
    re.compile(r"\bhappy penguin(s)?\b", re.IGNORECASE),
    re.compile(r"\btype of cheese\b", re.IGNORECASE),
    re.compile(r"\bbananas?\b", re.IGNORECASE),
]

_WORD = re.compile(r"[a-zA-Z]{3,}")


def _keyword_set(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def detect_nonsense(question: str, answer: str) -> CorrectnessSignal:
    q = question.strip()
    a = answer.strip()

    if not a:
        return CorrectnessSignal(flags=["correctness.empty_answer"])

    if not any(p.search(a) for p in _ABSURD_MARKERS):
        return CorrectnessSignal(flags=[])

    qk = _keyword_set(q)
    ak = _keyword_set(a)
    if not qk:
        return CorrectnessSignal(flags=[])

    overlap = len(qk.intersection(ak)) / max(1, len(qk))
    if overlap < 0.15:
        return CorrectnessSignal(flags=["correctness.nonsense_detected"])

    return CorrectnessSignal(flags=[])
