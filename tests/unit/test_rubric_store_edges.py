from __future__ import annotations

import pytest

from llm_judge.rubric_store import get_rubric


def test_get_rubric_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_rubric("this_rubric_does_not_exist")
