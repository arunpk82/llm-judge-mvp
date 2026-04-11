from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetSpec:
    path: str | None
    dataset_id: str
    version: str


@dataclass(frozen=True)
class SampleSpec:
    """
    Sampling configuration for PR gate / fast runs.

    strategy:
      - "stable_hash": deterministic sampling that remains stable as dataset grows,
                       assuming each row has a stable unique case_id.
    """

    n: int
    seed: int = 42
    strategy: str = "stable_hash"


@dataclass(frozen=True)
class SmokeTestSpec:
    """Fast-fail gate: run N cases, abort if pass rate below threshold."""

    n: int = 10
    min_pass_rate: float = 0.5


@dataclass(frozen=True)
class RunSpec:
    run_id_prefix: str
    dataset: DatasetSpec
    rubric_id: str
    judge_engine: str
    output_dir: str
    random_seed: int = 42
    sample: SampleSpec | None = None
    smoke_test: SmokeTestSpec | None = None

    @staticmethod
    def from_yaml(path: str | Path) -> "RunSpec":
        p = Path(path)
        data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))

        ds = data["dataset"]
        path_val = ds.get("path")
        dataset = DatasetSpec(
            path=str(path_val) if path_val is not None else None,
            dataset_id=str(ds["dataset_id"]),
            version=str(ds["version"]),
        )

        # Optional sampling config (used for PR gate)
        sample_obj: SampleSpec | None = None
        sample_data = data.get("sample")
        if isinstance(sample_data, dict):
            n_val = sample_data.get("n")
            if n_val is not None:
                if not isinstance(n_val, int) or n_val <= 0:
                    raise ValueError("sample.n must be a positive int")

                # Default seed to run random_seed if sample.seed not provided
                seed_val = sample_data.get("seed", data.get("random_seed", 42))
                if not isinstance(seed_val, int):
                    raise ValueError("sample.seed must be an int")

                strategy_val = str(sample_data.get("strategy", "stable_hash"))
                if strategy_val != "stable_hash":
                    raise ValueError(f"Unsupported sample.strategy: {strategy_val}")

                sample_obj = SampleSpec(
                    n=int(n_val), seed=int(seed_val), strategy=strategy_val
                )

        # Optional smoke test config
        smoke_obj: SmokeTestSpec | None = None
        smoke_data = data.get("smoke_test")
        if isinstance(smoke_data, dict):
            smoke_obj = SmokeTestSpec(
                n=int(smoke_data.get("n", 10)),
                min_pass_rate=float(smoke_data.get("min_pass_rate", 0.5)),
            )

        return RunSpec(
            run_id_prefix=str(data["run_id_prefix"]),
            dataset=dataset,
            rubric_id=str(data["rubric_id"]),
            judge_engine=str(data["judge_engine"]),
            output_dir=str(data["output_dir"]),
            random_seed=int(data.get("random_seed", 42)),
            sample=sample_obj,
            smoke_test=smoke_obj,
        )
