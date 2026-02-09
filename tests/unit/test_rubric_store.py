import pytest

from llm_judge.rubric_store import get_rubric


def test_get_rubric_resolves_latest_version() -> None:
    r = get_rubric("chat_quality")
    assert r.rubric_id == "chat_quality"
    assert r.version == "v1"
    assert set(r.dimensions) == {"relevance", "clarity", "correctness", "tone"}


def test_get_rubric_explicit_version() -> None:
    r = get_rubric("chat_quality@v1")
    assert r.rubric_id == "chat_quality"
    assert r.version == "v1"


def test_get_rubric_unknown_raises() -> None:
    with pytest.raises(ValueError):
        get_rubric("does_not_exist")



