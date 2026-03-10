from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.eval.dataset import DatasetValidationError, load_and_validate_dataset
from llm_judge.eval.sampling import SamplingError, deterministic_sample_rows


def test_dataset_requires_manifest_and_case_id(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "ds"
    dataset_dir.mkdir(parents=True)
    dataset_path = dataset_dir / "v1.jsonl"
    dataset_path.write_text('{"conversation":[],"candidate_answer":"x"}\n', encoding="utf-8")

    with pytest.raises(DatasetValidationError):
        load_and_validate_dataset(dataset_path)


def test_sampling_is_deterministic_with_seed() -> None:
    rows = [
        {"case_id": "c3", "x": 1},
        {"case_id": "c1", "x": 2},
        {"case_id": "c2", "x": 3},
        {"case_id": "c4", "x": 4},
    ]

    a = deterministic_sample_rows(rows=rows, seed=1234, sample_size=2)
    b = deterministic_sample_rows(rows=rows, seed=1234, sample_size=2)
    assert [r["case_id"] for r in a] == [r["case_id"] for r in b]


def test_sampling_rejects_missing_case_id() -> None:
    rows = [{"x": 1}, {"x": 2}]
    with pytest.raises(SamplingError):
        deterministic_sample_rows(rows=rows, seed=1, sample_size=1)