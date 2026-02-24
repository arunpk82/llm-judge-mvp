from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class DatasetSpec:
    """Spec for a deterministic dataset."""

    dataset_id: str
    path: str
    schema_version: str
    ci_sample_size: int
    nightly_full_run: bool


def load_manifest(path: str | Path) -> list[DatasetSpec]:
    p = Path(path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    items = raw.get("datasets")
    if not isinstance(items, list):
        raise ValueError("manifest missing 'datasets' list")

    out: list[DatasetSpec] = []
    for ds in items:
        if not isinstance(ds, dict):
            continue

        dataset_id = str(ds.get("id", "")).strip()
        ds_path = str(ds.get("path", "")).strip()
        schema_version = str(ds.get("schema_version", "v1")).strip()
        ci_sample_size = int(ds.get("ci_sample_size", 300))
        nightly_full_run = bool(ds.get("nightly_full_run", True))

        if not dataset_id or not ds_path:
            raise ValueError("dataset spec requires non-empty 'id' and 'path'")

        out.append(
            DatasetSpec(
                dataset_id=dataset_id,
                path=ds_path,
                schema_version=schema_version,
                ci_sample_size=ci_sample_size,
                nightly_full_run=nightly_full_run,
            )
        )

    return out


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def stable_sample(
    rows: Iterable[dict[str, Any]],
    *,
    sample_size: int,
    key_field: str = "case_id",
) -> list[dict[str, Any]]:
    """Deterministically select N rows using a stable hash of key_field."""
    if sample_size <= 0:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        key = str(r.get(key_field, ""))
        if not key:
            key = json.dumps(r, sort_keys=True)
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()
        scored.append((int(h[:12], 16), r))

    scored.sort(key=lambda t: t[0])
    return [r for _, r in scored[:sample_size]]
