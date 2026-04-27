"""Tests for the rubric_version resolution helper added in CP-1c-b.1.

The wrapper layer translates the ``"latest"`` sentinel into a concrete
version against the rubric registry; explicit versions pass through
unchanged. Governance preflight is CP-1c-b.2 scope.
"""
from __future__ import annotations

import pytest

from llm_judge.control_plane.wrappers import _resolve_effective_version


def test_explicit_version_passes_through() -> None:
    """An explicit version like 'v1' is returned unchanged."""
    assert _resolve_effective_version("chat_quality", "v1") == "v1"


def test_latest_resolves_via_registry_chat_quality() -> None:
    """'latest' resolves to the registry's chat_quality pin."""
    assert _resolve_effective_version("chat_quality", "latest") == "v1"


def test_latest_resolves_via_registry_math_basic() -> None:
    """'latest' resolves correctly for a non-default rubric."""
    assert _resolve_effective_version("math_basic", "latest") == "v1"


def test_latest_unregistered_rubric_raises() -> None:
    """Unregistered rubric_id with 'latest' surfaces a ValueError from
    the registry — explicit governance preflight is CP-1c-b.2 scope,
    but the registry lookup itself still rejects unknown ids."""
    with pytest.raises(ValueError):
        _resolve_effective_version("nonexistent_rubric", "latest")
