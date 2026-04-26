"""Tests for the adapter registry.

The seven canonical connectors must be pre-registered at module
import time. ``build`` invokes the registered factory and returns
a configured adapter instance. ``BenchmarkNotFoundError`` is raised
with a helpful "available" list for unknown names. ``list_benchmarks``
returns the names alphabetically sorted.
"""

from __future__ import annotations

import pytest

from llm_judge.benchmarks import BenchmarkAdapter
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
from llm_judge.benchmarks.registry import (
    BenchmarkNotFoundError,
    build,
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


def test_build_returns_instance_of_benchmark_adapter() -> None:
    for canonical in CANONICAL_NAMES:
        instance = build(canonical)
        assert isinstance(instance, BenchmarkAdapter), (
            f"{canonical!r} → {instance!r} is not a BenchmarkAdapter instance"
        )


def test_build_unknown_name_raises_benchmark_not_found_error() -> None:
    with pytest.raises(BenchmarkNotFoundError) as exc_info:
        build("nonexistent_benchmark_xyz")
    msg = str(exc_info.value)
    assert "nonexistent_benchmark_xyz" in msg
    assert "Available:" in msg
    # At least one canonical name should appear in the available list.
    assert "ragtruth_50" in msg


def test_register_accepts_factory() -> None:
    """Registering a zero-arg factory resolves through build."""

    class _Probe(BenchmarkAdapter):
        def metadata(self):  # type: ignore[override]
            raise NotImplementedError

        def load_cases(self, *, split="test", max_cases=None):  # type: ignore[override]
            raise NotImplementedError

    sentinel = _Probe()
    register("registry_test_probe", lambda: sentinel)
    try:
        assert build("registry_test_probe") is sentinel
        assert "registry_test_probe" in list_benchmarks()
    finally:
        # Clean up — registries are process-global; tests must restore state.
        from llm_judge.benchmarks.registry import _REGISTRY

        _REGISTRY.pop("registry_test_probe", None)


def test_register_last_write_wins() -> None:
    """Re-registering a name replaces the previous factory."""

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
        assert isinstance(build("registry_test_overwrite"), _B)
    finally:
        from llm_judge.benchmarks.registry import _REGISTRY

        _REGISTRY.pop("registry_test_overwrite", None)


def test_ragtruth_50_factory_applies_filter() -> None:
    """build('ragtruth_50') returns an adapter pre-restricted to the 50-case slice.

    The factory must call ``set_benchmark_filter`` with the canonical
    benchmark JSON, leaving the adapter with a populated 50-id filter
    set so that ``load_cases`` yields exactly the canonical slice
    rather than the full ~1170-case test dump.
    """
    adapter = build("ragtruth_50")
    assert isinstance(adapter, RAGTruthAdapter)
    # _benchmark_ids is the internal filter populated by set_benchmark_filter.
    # If the factory failed to apply the filter it would still be None.
    assert adapter._benchmark_ids is not None, (
        "ragtruth_50 factory did not apply set_benchmark_filter"
    )
    assert len(adapter._benchmark_ids) == 50, (
        f"expected 50 canonical response IDs; got {len(adapter._benchmark_ids)}"
    )
