"""Unit tests for the CP-F14 closure: RubricNotInRegistryError.

The new exception lives in :mod:`llm_judge.control_plane.types` and is
raised by :func:`llm_judge.rubric_store._resolve_version` when a
rubric_id is absent from ``rubrics/registry.yaml``'s ``latest:`` map
(the "unknown rubric_id" case at rubric_store.py:102-104).

The sibling raise at rubric_store.py:106-109 ("invalid latest version
for rubric '<id>'") is intentionally NOT migrated in L1-Pkt-2 — that
case describes registry malformation for a known rubric, which is
:class:`RubricSchemaError`'s territory rather than
:class:`RubricNotInRegistryError`'s. Tests below assert this scope
discipline so a future packet that revisits the secondary raise has a
clear precedent.
"""

from __future__ import annotations

import pytest

from llm_judge.control_plane.types import RubricNotInRegistryError
from llm_judge.rubric_store import _resolve_version


def test_rubric_not_in_registry_error_subclasses_value_error() -> None:
    """Subclass relationship preserves backward compatibility for
    callers that do ``except ValueError`` around rubric resolution."""
    assert issubclass(RubricNotInRegistryError, ValueError)


def test_resolve_version_unknown_rubric_id_raises_specific_subclass() -> None:
    """Unknown rubric_id raises RubricNotInRegistryError specifically;
    callers that want type precision can catch the new class without
    inspecting the message string."""
    with pytest.raises(RubricNotInRegistryError) as excinfo:
        _resolve_version("definitely_not_a_real_rubric_id_for_l1_pkt_2")
    assert "not registered" in str(excinfo.value)
    assert "definitely_not_a_real_rubric_id_for_l1_pkt_2" in str(excinfo.value)


def test_resolve_version_unknown_rubric_id_still_caught_by_value_error() -> None:
    """Existing ``except ValueError`` callers continue to catch the new
    exception unchanged. This is the subclass-relationship guarantee
    that CP-F14 closure depends on for backward compatibility."""
    with pytest.raises(ValueError):
        _resolve_version("definitely_not_a_real_rubric_id_for_l1_pkt_2")


def test_rubric_not_in_registry_error_in_module_all() -> None:
    """The exception is exported via :data:`__all__` so external callers
    can rely on a stable import path
    ``from llm_judge.control_plane.types import RubricNotInRegistryError``."""
    from llm_judge.control_plane import types

    assert "RubricNotInRegistryError" in types.__all__


def test_rubric_not_in_registry_error_has_docstring() -> None:
    """The exception class must carry an actionable docstring describing
    the raise condition and the subclass relationship rationale."""
    assert RubricNotInRegistryError.__doc__ is not None
    doc = RubricNotInRegistryError.__doc__
    assert "registry" in doc.lower()
    assert "ValueError" in doc
