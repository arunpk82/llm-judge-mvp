from __future__ import annotations

from pathlib import Path

import pytest

from llm_judge.eval.spec import RunSpec


def _write(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def test_runspec_parses_without_sample(tmp_path: Path) -> None:
    yml = tmp_path / "spec.yaml"
    _write(
        yml,
        """
run_id_prefix: pr-gate
dataset:
  path: datasets/golden/v1.jsonl
  dataset_id: math_basic
  version: v1
rubric_id: chat_quality
judge_engine: deterministic
output_dir: reports/runs
random_seed: 42
""".lstrip(),
    )
    spec = RunSpec.from_yaml(yml)
    assert spec.sample is None
    assert spec.random_seed == 42


def test_runspec_parses_sample(tmp_path: Path) -> None:
    yml = tmp_path / "spec.yaml"
    _write(
        yml,
        """
run_id_prefix: pr-gate
dataset:
  path: datasets/deterministic/math_basic_v2_10k.jsonl
  dataset_id: math_basic
  version: v2_10k
sample:
  n: 300
  seed: 123
  strategy: stable_hash
rubric_id: chat_quality
judge_engine: deterministic
output_dir: reports/runs
random_seed: 42
""".lstrip(),
    )
    spec = RunSpec.from_yaml(yml)
    assert spec.sample is not None
    assert spec.sample.n == 300
    assert spec.sample.seed == 123
    assert spec.sample.strategy == "stable_hash"


def test_runspec_rejects_bad_sample_n(tmp_path: Path) -> None:
    yml = tmp_path / "spec.yaml"
    _write(
        yml,
        """
run_id_prefix: pr-gate
dataset:
  path: x.jsonl
  dataset_id: math_basic
  version: v1
sample:
  n: 0
rubric_id: chat_quality
judge_engine: deterministic
output_dir: reports/runs
""".lstrip(),
    )
    with pytest.raises(ValueError):
        RunSpec.from_yaml(yml)
