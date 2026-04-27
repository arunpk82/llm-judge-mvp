"""Integration tests for F1 registry factory closures.

These tests exercise the full registry path from name → instance,
verifying the F1 closure: registry names produce correctly-configured
adapters without caller-side workarounds.
"""

from __future__ import annotations

import pytest

from llm_judge.benchmarks import BenchmarkAdapter, registry


def test_ragtruth_50_through_registry_produces_50_cases() -> None:
    """The F1 closure: build('ragtruth_50') produces the canonical
    50-case slice, not the 1170-case full corpus.

    This is the load-bearing regression test for the F1 finding.
    """
    adapter = registry.build("ragtruth_50")
    cases = list(adapter.load_cases())
    assert len(cases) == 50, (
        f"Expected 50 cases for canonical ragtruth_50 slice, got {len(cases)}"
    )


def test_all_seven_names_buildable() -> None:
    """Every registered name produces a working BenchmarkAdapter
    instance through the registry. This catches accidental
    registry-name additions or drops; failure here means the
    registry's list_benchmarks() and the actual factory map have
    diverged.
    """
    for name in registry.list_benchmarks():
        adapter = registry.build(name)
        assert isinstance(adapter, BenchmarkAdapter), (
            f"build({name!r}) returned {type(adapter).__name__}, "
            f"not a BenchmarkAdapter subclass"
        )


def test_build_unknown_raises_benchmark_not_found() -> None:
    """build() with an unknown name raises BenchmarkNotFoundError,
    matching the contract from the original get() API.
    """
    with pytest.raises(registry.BenchmarkNotFoundError):
        registry.build("does_not_exist_12345")
