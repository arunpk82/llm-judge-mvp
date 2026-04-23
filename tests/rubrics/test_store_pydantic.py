"""Tests for the Pydantic wiring in ``llm_judge.rubric_store``.

Covers:
  - Valid YAML parses through ``get_rubric`` and returns the
    backward-compatible ``Rubric`` dataclass.
  - Malformed YAML raises ``RubricSchemaError`` naming the file path.
  - Registry loading validates through ``RubricRegistryConfig``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_judge.rubric_store import Rubric, RubricSchemaError, get_rubric

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_get_rubric_returns_backward_compatible_dataclass() -> None:
    rubric = get_rubric("chat_quality")
    assert isinstance(rubric, Rubric)
    # Backward-compatible shape (CP-1 and earlier callers expect this).
    assert rubric.rubric_id == "chat_quality"
    assert rubric.version == "v1"
    assert "relevance" in rubric.dimensions
    assert "clarity" in rubric.dimensions
    assert "correctness" in rubric.dimensions
    assert "tone" in rubric.dimensions


def test_get_rubric_with_explicit_version() -> None:
    rubric = get_rubric("chat_quality@v1")
    assert rubric.version == "v1"


def test_get_rubric_applies_registry_metrics_schema() -> None:
    """The registry declares metrics_schema for chat_quality/v1
    (required=[f1_fail, cohen_kappa]). The returned Rubric must
    carry that contract."""
    rubric = get_rubric("chat_quality")
    assert "f1_fail" in rubric.metrics_required
    assert "cohen_kappa" in rubric.metrics_required


def test_malformed_yaml_raises_rubric_schema_error(tmp_path: Path) -> None:
    """A YAML parse error inside rubric_store surfaces as
    RubricSchemaError with the file path in the message."""
    from llm_judge.rubric_store import _load_yaml

    bad = tmp_path / "broken.yaml"
    bad.write_text("[unclosed: bracket\n", encoding="utf-8")

    with pytest.raises(RubricSchemaError) as exc:
        _load_yaml(bad)
    assert str(bad) in str(exc.value)


def test_registry_validation_catches_bad_latest_type(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If rubrics/registry.yaml has ``latest:`` with a non-string
    value, RubricRegistryConfig validation must reject it."""
    from llm_judge import rubric_store

    fake_rubrics = tmp_path / "rubrics"
    fake_rubrics.mkdir()
    (fake_rubrics / "registry.yaml").write_text(
        yaml.safe_dump({"latest": {"chat_quality": 123}}),  # int, not str
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rubric_store, "_project_root", lambda: tmp_path
    )

    with pytest.raises(RubricSchemaError) as exc:
        rubric_store._load_registry()
    # Message names the file path.
    assert "registry.yaml" in str(exc.value)


def test_get_rubric_missing_rubric_file_raises_value_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the rubric version exists in the registry but the YAML
    file is absent, get_rubric raises ValueError (not
    RubricSchemaError — this is a missing-file, not malformed-schema
    case)."""
    from llm_judge import rubric_store

    fake_rubrics = tmp_path / "rubrics"
    fake_rubrics.mkdir()
    (fake_rubrics / "registry.yaml").write_text(
        yaml.safe_dump({"latest": {"ghost": "v1"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rubric_store, "_project_root", lambda: tmp_path
    )

    with pytest.raises(ValueError) as exc:
        rubric_store.get_rubric("ghost")
    assert "Rubric file not found" in str(exc.value)


def test_rubric_schema_error_is_value_error_subclass() -> None:
    """Backward compatibility: callers that catch ValueError around
    get_rubric must continue to catch schema failures."""
    assert issubclass(RubricSchemaError, ValueError)


def test_get_rubric_propagates_governance_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rubric YAML missing governance fields (owner, status, etc.)
    surfaces through get_rubric as RubricSchemaError naming the file."""
    from llm_judge import rubric_store

    fake_rubrics = tmp_path / "rubrics"
    (fake_rubrics / "partial").mkdir(parents=True)
    (fake_rubrics / "registry.yaml").write_text(
        yaml.safe_dump({"latest": {"partial": "v1"}}),
        encoding="utf-8",
    )
    # Rubric file exists but lacks governance fields.
    (fake_rubrics / "partial" / "v1.yaml").write_text(
        yaml.safe_dump(
            {
                "rubric_id": "partial",
                "version": "v1",
                "dimensions": [{"name": "x"}],
                "decision_policy": {},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        rubric_store, "_project_root", lambda: tmp_path
    )

    with pytest.raises(RubricSchemaError) as exc:
        rubric_store.get_rubric("partial")
    assert "partial" in str(exc.value)
    assert "v1.yaml" in str(exc.value)
