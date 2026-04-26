"""Tests for the adapter registry.

The seven canonical connectors must be pre-registered at module
import time. ``get`` raises :class:`BenchmarkNotFoundError` with a
helpful "available" list for unknown names. ``list_benchmarks``
returns the names alphabetically sorted.
"""

from __future__ import annotations

import pytest

from llm_judge.benchmarks import BenchmarkAdapter
from llm_judge.benchmarks.registry import (
    BenchmarkNotFoundError,
    get,
    list_benchmarks,
    register,
)

CANONICAL_NAMES = (
    "faithdial",
    "fever",
    "halueval",
    "ifeval",
    "jigsaw",
    "ragtruth_50",
    "toxigen",
)


def test_seven_canonical_connectors_pre_registered() -> None:
    names = list_benchmarks()
    for canonical in CANONICAL_NAMES:
        assert canonical in names, (
            f"{canonical!r} not pre-registered. Got: {names}"
        )


def test_list_benchmarks_returns_sorted_names() -> None:
    names = list_benchmarks()
    assert names == sorted(names), (
        f"list_benchmarks should be sorted alphabetically. Got: {names}"
    )


def test_get_returns_subclass_of_benchmark_adapter() -> None:
    for canonical in CANONICAL_NAMES:
        cls = get(canonical)
        assert issubclass(cls, BenchmarkAdapter), (
            f"{canonical!r} → {cls!r} is not a BenchmarkAdapter subclass"
        )


def test_get_unknown_name_raises_with_available_list() -> None:
    with pytest.raises(BenchmarkNotFoundError) as exc_info:
        get("nonexistent_benchmark_xyz")
    msg = str(exc_info.value)
    assert "nonexistent_benchmark_xyz" in msg
    assert "Available:" in msg
    # At least one canonical name should appear in the available list.
    assert "ragtruth_50" in msg


def test_register_and_get_round_trip() -> None:
    """Registering a custom name resolves through get."""

    class _Probe(BenchmarkAdapter):
        def metadata(self):  # type: ignore[override]
            raise NotImplementedError

        def load_cases(self, *, split="test", max_cases=None):  # type: ignore[override]
            raise NotImplementedError

    register("registry_test_probe", _Probe)
    try:
        assert get("registry_test_probe") is _Probe
        assert "registry_test_probe" in list_benchmarks()
    finally:
        # Clean up — registries are process-global; tests must restore state.
        from llm_judge.benchmarks.registry import _REGISTRY

        _REGISTRY.pop("registry_test_probe", None)


def test_register_last_write_wins() -> None:
    """Re-registering a name replaces the previous class."""

    class _A(BenchmarkAdapter):
        def metadata(self):  # type: ignore[override]
            raise NotImplementedError

        def load_cases(self, *, split="test", max_cases=None):  # type: ignore[override]
            raise NotImplementedError

    class _B(BenchmarkAdapter):
        def metadata(self):  # type: ignore[override]
            raise NotImplementedError

        def load_cases(self, *, split="test", max_cases=None):  # type: ignore[override]
            raise NotImplementedError

    register("registry_test_overwrite", _A)
    register("registry_test_overwrite", _B)
    try:
        assert get("registry_test_overwrite") is _B
    finally:
        from llm_judge.benchmarks.registry import _REGISTRY

        _REGISTRY.pop("registry_test_overwrite", None)
