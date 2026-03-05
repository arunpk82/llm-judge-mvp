from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.datasets import DatasetRegistry


def test_dataset_registry_resolves_golden_v1() -> None:
    reg = DatasetRegistry(root_dir=Path("datasets"))
    resolved = reg.resolve(dataset_id="golden", version="v1")
    assert resolved.metadata.dataset_id == "golden"
    assert resolved.metadata.version == "v1"
    assert resolved.data_path.exists()


def test_dataset_registry_missing_metadata(tmp_path: Path) -> None:
    reg = DatasetRegistry(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.resolve(dataset_id="golden", version="v1")


def test_dataset_registry_version_mismatch(tmp_path: Path) -> None:
    ds_dir = tmp_path / "golden"
    ds_dir.mkdir(parents=True)
    (ds_dir / "dataset.yaml").write_text(
        "dataset_id: golden\nversion: v999\nschema_version: 1\ndata_file: v1.jsonl\n",
        encoding="utf-8",
    )
    (ds_dir / "v1.jsonl").write_text("{}\n", encoding="utf-8")

    reg = DatasetRegistry(root_dir=tmp_path)
    with pytest.raises(ValueError):
        reg.resolve(dataset_id="golden", version="v1")
