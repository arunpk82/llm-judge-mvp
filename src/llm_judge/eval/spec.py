from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    dataset_id: str
    version: str


@dataclass(frozen=True)
class RunSpec:
    run_id_prefix: str
    dataset: DatasetSpec
    rubric_id: str
    judge_engine: str
    output_dir: str
    random_seed: int = 42

    @staticmethod
    def from_yaml(path: str | Path) -> "RunSpec":
        p = Path(path)
        data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))

        ds = data["dataset"]
        dataset = DatasetSpec(
            path=str(ds["path"]),
            dataset_id=str(ds["dataset_id"]),
            version=str(ds["version"]),
        )

        return RunSpec(
            run_id_prefix=str(data["run_id_prefix"]),
            dataset=dataset,
            rubric_id=str(data["rubric_id"]),
            judge_engine=str(data["judge_engine"]),
            output_dir=str(data["output_dir"]),
            random_seed=int(data.get("random_seed", 42)),
        )
