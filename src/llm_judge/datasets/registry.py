from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import DatasetMetadata

logger = logging.getLogger(__name__)


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file. Matches eval/run.py convention."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@dataclass(frozen=True)
class ResolvedDataset:
    metadata: DatasetMetadata
    dataset_dir: Path

    @property
    def data_path(self) -> Path:
        return self.dataset_dir / self.metadata.data_file


class DatasetRegistry:
    """Resolves datasets by governed metadata.

    Contract:
      datasets/<dataset_id>/dataset.yaml

    Environment override:
      LLM_JUDGE_DATASETS_DIR: defaults to ./datasets
    """

    def __init__(self, root_dir: str | Path | None = None) -> None:
        if root_dir is None:
            root_dir = Path.cwd() / "datasets"
        self.root_dir = Path(root_dir)

    @staticmethod
    def default() -> "DatasetRegistry":
        import os

        root = os.getenv("LLM_JUDGE_DATASETS_DIR")
        return DatasetRegistry(root_dir=root) if root else DatasetRegistry()

    def resolve(self, *, dataset_id: str, version: str) -> ResolvedDataset:
        ds_dir = self.root_dir / dataset_id
        meta_path = ds_dir / "dataset.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing dataset metadata: {meta_path}")

        raw = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        metadata = DatasetMetadata.model_validate(raw)

        if metadata.dataset_id != dataset_id:
            raise ValueError(
                f"dataset.yaml dataset_id mismatch: expected={dataset_id} got={metadata.dataset_id}"
            )
        if metadata.version != version:
            raise ValueError(
                f"dataset.yaml version mismatch: expected={version} got={metadata.version}"
            )

        data_path = ds_dir / metadata.data_file
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset data_file not found: {data_path}")

        # P02 Trust Architecture: verify file integrity on every read.
        # Hash computed at registration must match file on disk — detects
        # silent file replacement between runs.
        if metadata.content_hash:
            actual_hash = _sha256_file(data_path)
            if actual_hash != metadata.content_hash:
                raise ValueError(
                    f"Dataset integrity check failed for {dataset_id}@{version}: "
                    f"expected={metadata.content_hash} actual={actual_hash}. "
                    f"The data file may have been modified since registration."
                )

        # EPIC-1.1: Validate dataset integrity before returning.
        # This makes validation non-bypassable — any code loading a dataset
        # through the registry automatically validates it.
        #
        # We run integrity checks (empty, duplicates, case_id completeness)
        # but NOT strict per-row schema validation — row format varies by
        # rubric/dataset type, and the eval runner handles missing optional
        # fields gracefully with .get().
        try:
            from llm_judge.dataset_validator import (
                _check_integrity,
                _check_security,
                _parse_and_validate_cases_jsonl,
            )

            if data_path.suffix in (".yaml", ".yml"):
                from llm_judge.dataset_validator import _parse_and_validate_cases_yaml
                raw_cases, parse_errors = _parse_and_validate_cases_yaml(data_path)
            else:
                raw_cases, parse_errors = _parse_and_validate_cases_jsonl(data_path)

            # Parse errors (malformed JSON/YAML) are hard failures
            hard_errors = [e for e in parse_errors if e.code in ("MALFORMED_JSON", "MALFORMED_YAML")]
            if hard_errors:
                error_lines = "; ".join(
                    f"[{e.code}] {e.message}" for e in hard_errors[:5]
                )
                raise ValueError(
                    f"Dataset parse failed for {dataset_id}@{version}: {error_lines}"
                )

            # Integrity checks: empty, duplicate IDs, case_id completeness
            integrity_errors = _check_integrity(raw_cases, data_path)
            if integrity_errors:
                error_lines = "; ".join(
                    f"[{e.code}] {e.message}" for e in integrity_errors[:5]
                )
                raise ValueError(
                    f"Dataset integrity check failed for {dataset_id}@{version}: {error_lines}"
                    + (f" (+{len(integrity_errors) - 5} more)" if len(integrity_errors) > 5 else "")
                )

            # TASK-1.1.3: Security scanning (warnings, not hard failures)
            security_warnings = _check_security(raw_cases, data_path)
            if security_warnings:
                for sw in security_warnings[:5]:
                    logger.warning(
                        "dataset.security_warning",
                        extra={"dataset_id": dataset_id, "code": sw.code, "message": sw.message},
                    )
        except ImportError:
            # Graceful degradation: if validator module is unavailable,
            # log warning but don't block — allows minimal environments.
            logger.warning(
                "dataset_validator not available — skipping content validation",
                extra={"dataset_id": dataset_id, "version": version},
            )

        return ResolvedDataset(metadata=metadata, dataset_dir=ds_dir)
