import pytest

from llm_judge.control_plane.types import RubricNotInRegistryError
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
    """Backward-compat sentinel: existing ``except ValueError`` callers
    continue to catch unknown-rubric errors via the subclass
    relationship (CP-F14)."""
    with pytest.raises(ValueError):
        get_rubric("does_not_exist")


def test_get_rubric_unknown_raises_specific_type() -> None:
    """CP-F14 type precision: the unknown-rubric raise propagated
    through ``get_rubric`` is specifically RubricNotInRegistryError."""
    with pytest.raises(RubricNotInRegistryError):
        get_rubric("does_not_exist")
