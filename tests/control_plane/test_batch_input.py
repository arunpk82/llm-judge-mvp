"""Tests for the batch input file format.

Asserts the round-trip property (model → dump → load equals original)
and that malformed input fails fast with :class:`BatchInputSchemaError`.
The CLI driver in ``tools/run_batch_evaluation.py`` depends on the
error type being stable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llm_judge.control_plane.batch_input import (
    BatchCase,
    BatchInputFile,
    BatchInputSchemaError,
    load_batch_file,
)


def _sample_dict() -> dict:
    return {
        "schema_version": 1,
        "cases": [
            {
                "case_id": "c001",
                "response": "Paris is the capital of France.",
                "source": "Paris is the capital and largest city of France.",
            },
            {
                "case_id": "c002",
                "response": "Water boils at 100 C.",
                "source": "At sea level, water boils at 100 °C.",
                "metadata": {"category": "physics"},
            },
        ],
        "metadata": {"author": "test"},
    }


def test_load_yaml_succeeds(tmp_path: Path) -> None:
    p = tmp_path / "cases.yaml"
    p.write_text(yaml.safe_dump(_sample_dict()), encoding="utf-8")

    loaded = load_batch_file(p)
    assert isinstance(loaded, BatchInputFile)
    assert loaded.schema_version == 1
    assert len(loaded.cases) == 2
    assert loaded.cases[0].case_id == "c001"
    assert loaded.cases[1].metadata == {"category": "physics"}


def test_load_yml_extension_succeeds(tmp_path: Path) -> None:
    p = tmp_path / "cases.yml"
    p.write_text(yaml.safe_dump(_sample_dict()), encoding="utf-8")

    loaded = load_batch_file(p)
    assert len(loaded.cases) == 2


def test_load_json_succeeds(tmp_path: Path) -> None:
    p = tmp_path / "cases.json"
    p.write_text(json.dumps(_sample_dict()), encoding="utf-8")

    loaded = load_batch_file(p)
    assert isinstance(loaded, BatchInputFile)
    assert len(loaded.cases) == 2


def test_round_trip_dump_then_load_preserves_cases(tmp_path: Path) -> None:
    original = BatchInputFile.model_validate(_sample_dict())
    p = tmp_path / "rt.yaml"
    p.write_text(
        yaml.safe_dump(original.model_dump(mode="json")), encoding="utf-8"
    )

    reloaded = load_batch_file(p)
    assert reloaded.model_dump() == original.model_dump()


def test_malformed_yaml_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "broken.yaml"
    p.write_text("schema_version: 1\ncases: [ { unclosed:", encoding="utf-8")

    with pytest.raises(BatchInputSchemaError) as exc_info:
        load_batch_file(p)
    assert str(p) in str(exc_info.value)


def test_missing_required_field_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "missing_response.yaml"
    p.write_text(
        yaml.safe_dump({"cases": [{"case_id": "c001"}]}), encoding="utf-8"
    )

    with pytest.raises(BatchInputSchemaError) as exc_info:
        load_batch_file(p)
    assert "response" in str(exc_info.value)


def test_unsupported_extension_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "cases.toml"
    p.write_text("schema_version = 1", encoding="utf-8")

    with pytest.raises(BatchInputSchemaError) as exc_info:
        load_batch_file(p)
    assert "unsupported extension" in str(exc_info.value)


def test_empty_file_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")

    with pytest.raises(BatchInputSchemaError) as exc_info:
        load_batch_file(p)
    assert "empty" in str(exc_info.value)


def test_missing_file_raises_schema_error(tmp_path: Path) -> None:
    p = tmp_path / "does_not_exist.yaml"
    with pytest.raises(BatchInputSchemaError):
        load_batch_file(p)


def test_batch_case_optional_source_allowed() -> None:
    case = BatchCase(case_id="c001", response="r")
    assert case.source is None


def test_load_accepts_string_path(tmp_path: Path) -> None:
    p = tmp_path / "cases.yaml"
    p.write_text(yaml.safe_dump(_sample_dict()), encoding="utf-8")
    loaded = load_batch_file(str(p))
    assert len(loaded.cases) == 2
