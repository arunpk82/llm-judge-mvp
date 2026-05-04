"""Tests for benchmark dataset registration (CP-F1 closure).

Mirror of the existing :class:`DatasetRegistry` test pattern (also
in ``tests/unit/``). Covers the four scenarios catalogued in
L1-Pkt-A v2.2 §2:

  * Scenario 2 — first-time registration writes the sidecar
  * Scenario 3 — idempotent re-registration returns the same reference
  * Scenario 4 — content collision raises
  * BenchmarkFileNotFoundError on missing path

Plus a malformed-JSON case (defensive) and a sidecar id/version
divergence case (same integrity class as a content-hash mismatch).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from llm_judge.control_plane.types import (
    BenchmarkContentCollisionError,
    BenchmarkFileNotFoundError,
    BenchmarkReference,
)
from llm_judge.datasets.benchmark_registry import (
    register_benchmark,
    sidecar_path_for,
)


def _write_benchmark_json(
    tmp_path: Path,
    *,
    benchmark_id: str = "demo_5",
    version: str = "1.0",
    extra: dict | None = None,
    filename: str | None = None,
) -> Path:
    payload = {
        "benchmark_id": benchmark_id,
        "version": version,
        "source_file": "datasets/benchmarks/demo/responses.jsonl",
        "response_ids": ["0", "1", "2", "3", "4"],
    }
    if extra:
        payload.update(extra)
    name = filename or f"{benchmark_id}_benchmark.json"
    path = tmp_path / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------
# Scenario 2 — first-time registration
# ---------------------------------------------------------------------


def test_register_benchmark_returns_reference(tmp_path: Path) -> None:
    json_path = _write_benchmark_json(tmp_path)
    ref = register_benchmark(json_path)
    assert isinstance(ref, BenchmarkReference)
    assert ref.benchmark_id == "demo_5"
    assert ref.benchmark_version == "1.0"
    assert ref.benchmark_content_hash.startswith("sha256:")
    assert isinstance(ref.benchmark_registration_timestamp, datetime)


def test_register_benchmark_writes_sidecar(tmp_path: Path) -> None:
    json_path = _write_benchmark_json(tmp_path)
    register_benchmark(json_path)
    sidecar = sidecar_path_for(json_path)
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["benchmark_id"] == "demo_5"
    assert payload["benchmark_version"] == "1.0"
    assert payload["benchmark_content_hash"].startswith("sha256:")
    assert payload["source_file"] == json_path.name
    # Round-trips via fromisoformat without raising.
    datetime.fromisoformat(payload["benchmark_registration_timestamp"])


def test_sidecar_path_naming_convention(tmp_path: Path) -> None:
    json_path = tmp_path / "ragtruth_50_benchmark.json"
    sidecar = sidecar_path_for(json_path)
    assert sidecar.name == "ragtruth_50_registration.json"
    assert sidecar.parent == json_path.parent


def test_sidecar_path_falls_back_when_not_named_benchmark(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "halueval_qa.json"
    sidecar = sidecar_path_for(json_path)
    assert sidecar.name == "halueval_qa_registration.json"


# ---------------------------------------------------------------------
# Scenario 3 — idempotent re-registration
# ---------------------------------------------------------------------


def test_re_registration_is_idempotent(tmp_path: Path) -> None:
    json_path = _write_benchmark_json(tmp_path)
    first = register_benchmark(json_path)
    sidecar_mtime = sidecar_path_for(json_path).stat().st_mtime_ns
    second = register_benchmark(json_path)
    assert second.benchmark_content_hash == first.benchmark_content_hash
    assert (
        second.benchmark_registration_timestamp
        == first.benchmark_registration_timestamp
    )
    # Sidecar untouched on idempotent re-registration.
    assert sidecar_path_for(json_path).stat().st_mtime_ns == sidecar_mtime


# ---------------------------------------------------------------------
# Scenario 4 — content collision
# ---------------------------------------------------------------------


def test_content_collision_raises(tmp_path: Path) -> None:
    json_path = _write_benchmark_json(tmp_path)
    register_benchmark(json_path)
    # Mutate the benchmark JSON without re-registering.
    json_path.write_text(
        json.dumps(
            {
                "benchmark_id": "demo_5",
                "version": "1.0",
                "source_file": "datasets/benchmarks/demo/responses.jsonl",
                "response_ids": ["0", "1", "2"],  # changed
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with pytest.raises(BenchmarkContentCollisionError) as ei:
        register_benchmark(json_path)
    msg = str(ei.value)
    assert "demo_5" in msg
    assert "content_hash" in msg or "currently hashes" in msg


def test_id_or_version_divergence_in_sidecar_raises(
    tmp_path: Path,
) -> None:
    """The sidecar's recorded id/version must agree with the JSON
    definition. Tampering with the sidecar to point at a different
    id is the same integrity class as a hash collision."""
    json_path = _write_benchmark_json(tmp_path)
    register_benchmark(json_path)
    sidecar = sidecar_path_for(json_path)
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["benchmark_id"] = "different_id"
    sidecar.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(BenchmarkContentCollisionError):
        register_benchmark(json_path)


# ---------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------


def test_missing_file_raises_benchmark_file_not_found_error(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(BenchmarkFileNotFoundError) as ei:
        register_benchmark(missing)
    assert str(missing) in str(ei.value)


def test_missing_required_field_raises_value_error(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "bad_benchmark.json"
    bad.write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValueError) as ei:
        register_benchmark(bad)
    assert "benchmark_id" in str(ei.value)


def test_non_object_json_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad_benchmark.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError):
        register_benchmark(bad)


# ---------------------------------------------------------------------
# Subclass relationship — backward compat with except ValueError
# ---------------------------------------------------------------------


def test_benchmark_file_not_found_is_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        register_benchmark(tmp_path / "missing.json")


def test_benchmark_content_collision_is_value_error(
    tmp_path: Path,
) -> None:
    json_path = _write_benchmark_json(tmp_path)
    register_benchmark(json_path)
    json_path.write_text(
        json.dumps(
            {
                "benchmark_id": "demo_5",
                "version": "1.0",
                "response_ids": ["different"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        register_benchmark(json_path)
