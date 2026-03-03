from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make the module’s public surface explicit.
# This also helps mypy understand that these attributes are intended exports.
__all__ = [
    "DatasetValidationError",
    "DatasetManifest",
    "load_manifest",
    "validate_dataset_rows",
    "load_and_validate_dataset",
]


class DatasetValidationError(ValueError):
    """Raised when the dataset violates the deterministic dataset contract."""


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    version: str
    path: str
    row_count: int
    required_fields: list[str]

    @staticmethod
    def from_path(path: Path) -> "DatasetManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        schema = data.get("schema", {})
        required_fields = list(schema.get("required_fields", []))
        return DatasetManifest(
            dataset_id=str(data["dataset_id"]),
            version=str(data["version"]),
            path=str(data["path"]),
            row_count=int(data["row_count"]),
            required_fields=required_fields,
        )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_manifest(dataset_path: Path) -> DatasetManifest:
    """Load the sibling _index.json manifest for a dataset file."""
    manifest_path = dataset_path.parent / "_index.json"
    if not manifest_path.exists():
        raise DatasetValidationError(f"Missing dataset manifest: {manifest_path}")
    return DatasetManifest.from_path(manifest_path)


def validate_dataset_rows(*, rows: list[dict[str, Any]], manifest: DatasetManifest) -> None:
    """
    Deterministic dataset contract enforcement:
      - manifest.row_count matches actual row count
      - required fields exist
      - case_id exists, is a non-empty string, and is unique
    """
    if manifest.row_count != len(rows):
        raise DatasetValidationError(
            f"Manifest row_count={manifest.row_count} does not match dataset rows={len(rows)}"
        )

    # Always enforce case_id even if manifest omits it.
    required_fields = list(manifest.required_fields)
    if "case_id" not in required_fields:
        required_fields = ["case_id", *required_fields]

    seen: set[str] = set()

    for i, row in enumerate(rows):
        missing = [k for k in required_fields if k not in row]
        if missing:
            raise DatasetValidationError(f"Row {i} missing required fields: {missing}")

        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            raise DatasetValidationError(
                f"Row {i} has invalid case_id (must be non-empty string): {cid!r}"
            )

        if cid in seen:
            raise DatasetValidationError(f"Duplicate case_id detected: {cid}")
        seen.add(cid)


def load_and_validate_dataset(dataset_path: Path) -> tuple[DatasetManifest, list[dict[str, Any]]]:
    """
    Load a dataset and enforce deterministic dataset contract.

    Returns:
      (manifest, rows)
    """
    manifest = load_manifest(dataset_path)

    expected_path = (dataset_path.parent / manifest.path).resolve()
    if expected_path != dataset_path.resolve():
        raise DatasetValidationError(
            f"Dataset path mismatch. Manifest expects {manifest.path}, got {dataset_path.name}"
        )

    rows = _load_jsonl(dataset_path)
    validate_dataset_rows(rows=rows, manifest=manifest)
    return manifest, rows