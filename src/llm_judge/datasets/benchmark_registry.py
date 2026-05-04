"""Benchmark dataset registration (CP-F1 closure, CAP-1 expansion).

Benchmarks (RAGTruth-50, HaluEval, ...) are persistent versioned
inputs that the ``tools/run_batch_evaluation.py`` adapter consumes.
Before this module, an adapter could load any benchmark JSON without
the platform recording its identity or integrity. CP-F1 makes
benchmark files first-class governed CAP-1 inputs:

  * The benchmark JSON definition file (e.g.
    ``datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json``) is
    the unit of registration. Content-hashing the JSON gives
    integrity tracking on the unit-of-meaning; the underlying
    response.jsonl / source_info.jsonl files are referenced by the
    JSON and inherit integrity transitively.
  * Registration is idempotent. The first call writes a sidecar
    ``<benchmark_id>_registration.json`` next to the benchmark JSON
    recording the SHA-256 + UTC timestamp; subsequent calls verify
    the file content still matches and return the recorded
    :class:`BenchmarkReference`.
  * Failures fail closed at the adapter entry — no partial-batch
    execution. ``BenchmarkFileNotFoundError`` and
    ``BenchmarkContentCollisionError`` both subclass
    :class:`ValueError`.

Module location aligned with existing per-dataset convention
(Pre-flight 4 recon): lives in ``src/llm_judge/datasets/`` next to
:class:`DatasetRegistry`. No new ``data/benchmark-registry/``
directory; sidecar files live next to the benchmark JSON they
register.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.control_plane.types import (
    BenchmarkContentCollisionError,
    BenchmarkFileNotFoundError,
    BenchmarkReference,
)

__all__ = ["register_benchmark", "sidecar_path_for"]


def _sha256_file(path: Path) -> str:
    """SHA-256 hex digest of ``path``'s bytes, prefixed ``sha256:``.

    Matches the convention used by
    :func:`llm_judge.datasets.registry._sha256_file` and CAP-1
    hashing so that downstream comparisons stay shape-stable."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sidecar_path_for(benchmark_json_path: Path) -> Path:
    """Return the sidecar registration path next to a benchmark JSON.

    Convention: ``<benchmark_id>_benchmark.json`` ↔
    ``<benchmark_id>_registration.json``. We derive the prefix from
    the benchmark JSON's stem rather than from the JSON's
    ``benchmark_id`` field so the sidecar is locatable from the file
    name alone (no need to read+parse the JSON to find its
    sidecar)."""
    stem = benchmark_json_path.stem
    if stem.endswith("_benchmark"):
        prefix = stem[: -len("_benchmark")]
    else:
        prefix = stem
    return benchmark_json_path.with_name(f"{prefix}_registration.json")


def _load_benchmark_definition(path: Path) -> dict[str, Any]:
    """Read+parse the benchmark JSON. Validates the two fields that
    :class:`BenchmarkReference` consumes (``benchmark_id``,
    ``version``); leaves the rest unvalidated — adapter-format
    concerns are not this function's territory."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"benchmark JSON at {path} is not a JSON object "
            f"(got {type(raw).__name__})"
        )
    for key in ("benchmark_id", "version"):
        val = raw.get(key)
        if not isinstance(val, str) or not val:
            raise ValueError(
                f"benchmark JSON at {path} missing required string "
                f"field {key!r}"
            )
    return raw


def register_benchmark(json_path: str | Path) -> BenchmarkReference:
    """Register a benchmark JSON definition file. Idempotent.

    First call: computes SHA-256 over the JSON file bytes, writes a
    sidecar ``<benchmark_id>_registration.json`` recording the hash
    and a UTC timestamp, returns the resulting
    :class:`BenchmarkReference`.

    Subsequent calls: re-computes the SHA-256, compares to the
    recorded value, and returns the recorded
    :class:`BenchmarkReference` if they agree. Mismatch raises
    :class:`BenchmarkContentCollisionError` so a benchmark whose
    content drifted under a stable ``benchmark_id`` cannot silently
    flow into the platform.

    Raises:
      :class:`BenchmarkFileNotFoundError` if ``json_path`` is missing.
      :class:`BenchmarkContentCollisionError` if a sidecar exists
        but disagrees with the file's current hash.
      :class:`ValueError` if the JSON is malformed or missing the
        required ``benchmark_id`` / ``version`` fields."""
    path = Path(json_path)
    if not path.is_file():
        raise BenchmarkFileNotFoundError(
            f"benchmark JSON definition not found: {path}"
        )

    definition = _load_benchmark_definition(path)
    benchmark_id = definition["benchmark_id"]
    version = definition["version"]
    content_hash = _sha256_file(path)
    sidecar = sidecar_path_for(path)

    if sidecar.is_file():
        recorded = json.loads(sidecar.read_text(encoding="utf-8"))
        recorded_hash = recorded.get("benchmark_content_hash")
        if recorded_hash != content_hash:
            raise BenchmarkContentCollisionError(
                f"benchmark {benchmark_id!r} sidecar at {sidecar} "
                f"records content_hash={recorded_hash!r} but "
                f"{path.name} currently hashes to {content_hash!r}; "
                f"refusing to register a drifted benchmark under a "
                f"stable id."
            )
        recorded_ts = datetime.fromisoformat(
            recorded["benchmark_registration_timestamp"]
        )
        # The sidecar's recorded benchmark_id/version must also agree
        # with the JSON definition. A divergence means someone
        # rewrote the JSON definition without re-registering — same
        # class of integrity failure as a content-hash mismatch.
        if (
            recorded.get("benchmark_id") != benchmark_id
            or recorded.get("benchmark_version") != version
        ):
            raise BenchmarkContentCollisionError(
                f"benchmark sidecar at {sidecar} records "
                f"id/version=({recorded.get('benchmark_id')!r},"
                f"{recorded.get('benchmark_version')!r}) but "
                f"{path.name} currently declares "
                f"({benchmark_id!r},{version!r})."
            )
        return BenchmarkReference(
            benchmark_id=benchmark_id,
            benchmark_version=version,
            benchmark_content_hash=content_hash,
            benchmark_registration_timestamp=recorded_ts,
        )

    timestamp = datetime.now(timezone.utc)
    sidecar.write_text(
        json.dumps(
            {
                "benchmark_id": benchmark_id,
                "benchmark_version": version,
                "benchmark_content_hash": content_hash,
                "benchmark_registration_timestamp": timestamp.isoformat(),
                "source_file": path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return BenchmarkReference(
        benchmark_id=benchmark_id,
        benchmark_version=version,
        benchmark_content_hash=content_hash,
        benchmark_registration_timestamp=timestamp,
    )
