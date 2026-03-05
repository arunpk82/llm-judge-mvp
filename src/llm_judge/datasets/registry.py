from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .models import DatasetMetadata


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

        return ResolvedDataset(metadata=metadata, dataset_dir=ds_dir)
