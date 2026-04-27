"""Metrics schema governance tests for CP-1c-b.2 Concern 3."""
from __future__ import annotations

import pytest

from llm_judge.governance import (
    MetricsSchemaViolationError,
    enforce_metrics_schema,
)
from llm_judge.rubrics.lifecycle import (
    UngovernedRubricError,
    check_rubrics_governed,
)


def test_enforce_metrics_schema_passes_when_required_present() -> None:
    """Happy path: all required metrics present (extra keys allowed)."""
    enforce_metrics_schema(
        rubric_ref="chat_quality",
        metrics={"f1_fail": 0.5, "cohen_kappa": 0.7, "extra": 1},
    )


def test_enforce_metrics_schema_fails_when_required_missing() -> None:
    """Required key missing raises MetricsSchemaViolationError."""
    with pytest.raises(MetricsSchemaViolationError) as exc_info:
        enforce_metrics_schema(
            rubric_ref="chat_quality",
            metrics={"f1_fail": 0.5},  # missing cohen_kappa
        )
    msg = str(exc_info.value)
    assert "cohen_kappa" in msg
    assert "chat_quality" in msg


def test_enforce_metrics_schema_fails_for_math_basic_missing_keys() -> None:
    """math_basic requires accuracy + f1_fail; missing accuracy raises."""
    with pytest.raises(MetricsSchemaViolationError) as exc_info:
        enforce_metrics_schema(
            rubric_ref="math_basic",
            metrics={"f1_fail": 0.5},  # missing accuracy
        )
    assert "accuracy" in str(exc_info.value)


def test_check_rubrics_governed_accepts_chat_quality() -> None:
    """Existing rubric with metrics_schema passes governance."""
    check_rubrics_governed("chat_quality", "v1")


def test_check_rubrics_governed_accepts_math_basic() -> None:
    """Existing rubric with metrics_schema passes governance."""
    check_rubrics_governed("math_basic", "v1")


def test_check_rubrics_governed_rejects_missing_metrics_schema(
    tmp_path,
) -> None:
    """A governed rubric without metrics_schema is rejected (CP-1c-b.2)."""
    import shutil
    from pathlib import Path

    # Build a minimal isolated rubrics root: copy chat_quality's lifecycle
    # YAML but write a registry.yaml that declares NO metrics_schema for it.
    src_root = Path(__file__).resolve().parents[2] / "rubrics"
    rubric_id = "chat_quality"
    version = "v1"

    rubrics_root = tmp_path / "rubrics"
    (rubrics_root / rubric_id).mkdir(parents=True)
    shutil.copy(
        src_root / rubric_id / f"{version}.yaml",
        rubrics_root / rubric_id / f"{version}.yaml",
    )
    # registry.yaml that points at chat_quality but omits metrics_schema
    (rubrics_root / "registry.yaml").write_text(
        "latest:\n"
        f"  {rubric_id}: {version}\n"
        "rubrics:\n"
        f"  {rubric_id}:\n"
        "    versions:\n"
        f"      {version}: {{}}\n",
        encoding="utf-8",
    )

    with pytest.raises(UngovernedRubricError) as exc_info:
        check_rubrics_governed(rubric_id, version, rubrics_root=rubrics_root)
    assert "metrics_schema" in str(exc_info.value)
