"""Tests for ``check_rubrics_governed`` — the strict preflight check.

Tests are co-located with the preflight code (CP-1c-a Commit 4).
Every failure mode verifies:
  - ``UngovernedRubricError`` is raised
  - The message names the specific violation and points to the
    rubric file / fix action

Fixtures build self-contained rubric roots under ``tmp_path`` so
no test mutates the real ``rubrics/`` directory.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from llm_judge.rubrics import (
    REVIEW_PERIOD_DAYS,
    UngovernedRubricError,
    check_rubrics_governed,
)


def _valid_rubric_content(
    *,
    rubric_id: str = "fixture",
    version: str = "v1",
    last_reviewed: str = "2026-04-23",
) -> dict[str, Any]:
    """Base content that passes every preflight criterion. Tests
    take this and drop or replace a single field to exercise one
    failure branch at a time."""
    return {
        "rubric_id": rubric_id,
        "version": version,
        "title": f"Fixture Rubric ({version})",
        "owner": "test-owner",
        "created_at": "2026-01-01",
        "status": "production",
        "last_reviewed": last_reviewed,
        "audit_log": [
            {
                "timestamp": "2026-04-23T00:00:00Z",
                "actor": "test",
                "action": "created",
                "details": {},
            }
        ],
        "dimensions": [
            {"name": "relevance", "description": "x"},
            {"name": "clarity", "description": "y"},
        ],
    }


def _write_fixture(
    tmp_path: Path,
    *,
    rubric_content: dict[str, Any] | None = None,
    rubric_id: str = "fixture",
    version: str = "v1",
    registry_content: dict[str, Any] | None = None,
) -> Path:
    """Build a complete rubrics-root fixture: registry.yaml + one
    rubric file. Returns the rubrics_root Path for passing to
    ``check_rubrics_governed``."""
    root = tmp_path / "rubrics"
    (root / rubric_id).mkdir(parents=True, exist_ok=True)
    (root / rubric_id / f"{version}.yaml").write_text(
        yaml.safe_dump(rubric_content or _valid_rubric_content()),
        encoding="utf-8",
    )
    reg = registry_content if registry_content is not None else {
        "latest": {rubric_id: version}
    }
    (root / "registry.yaml").write_text(
        yaml.safe_dump(reg), encoding="utf-8"
    )
    return root


# =====================================================================
# Happy path
# =====================================================================


def test_governed_rubric_passes_chat_quality() -> None:
    """The real chat_quality rubric (migrated in this commit) must
    pass preflight. Uses the repo's rubrics/ root directly."""
    check_rubrics_governed("chat_quality")


def test_governed_rubric_passes_math_basic() -> None:
    """Same for math_basic."""
    check_rubrics_governed("math_basic")


def test_governed_fixture_passes(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)
    # No raise = success
    check_rubrics_governed("fixture", rubrics_root=root)


# =====================================================================
# Failure modes — each strips / mutates ONE field
# =====================================================================


def test_missing_owner_raises(tmp_path: Path) -> None:
    content = _valid_rubric_content()
    del content["owner"]
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "owner" in msg
    # Actionable-fix guidance: points to the file.
    assert "fixture/v1.yaml" in msg


def test_status_draft_raises(tmp_path: Path) -> None:
    content = _valid_rubric_content()
    content["status"] = "draft"
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "draft" in msg
    assert "validated" in msg and "production" in msg
    assert "fixture/v1.yaml" in msg


def test_status_deprecated_raises(tmp_path: Path) -> None:
    content = _valid_rubric_content()
    content["status"] = "deprecated"
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "deprecated" in msg
    assert "validated" in msg and "production" in msg


def test_missing_created_at_raises(tmp_path: Path) -> None:
    content = _valid_rubric_content()
    del content["created_at"]
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "created_at" in msg
    assert "fixture/v1.yaml" in msg


def test_stale_last_reviewed_raises(tmp_path: Path) -> None:
    stale = date.today() - timedelta(days=REVIEW_PERIOD_DAYS + 1)
    content = _valid_rubric_content(last_reviewed=stale.isoformat())
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "last_reviewed" in msg or "last reviewed" in msg
    assert str(REVIEW_PERIOD_DAYS) in msg
    assert "fixture/v1.yaml" in msg


def test_invalid_dimension_reference_raises(tmp_path: Path) -> None:
    content = _valid_rubric_content()
    content["dimensions"] = []  # empty list — structural violation
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "dimensions" in msg
    assert "fixture/v1.yaml" in msg


def test_dimension_missing_name_raises(tmp_path: Path) -> None:
    """Dimension list non-empty but an entry has no 'name'."""
    content = _valid_rubric_content()
    content["dimensions"] = [{"description": "no name here"}]
    root = _write_fixture(tmp_path, rubric_content=content)

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "name" in msg
    assert "fixture/v1.yaml" in msg


def test_orphan_registry_entry_raises(tmp_path: Path) -> None:
    """registry.yaml latest points to a version whose YAML does not
    exist on disk."""
    root = tmp_path / "rubrics"
    root.mkdir()
    (root / "fixture").mkdir()
    # Note: NO rubric file written. Registry references version v1.
    (root / "registry.yaml").write_text(
        yaml.safe_dump({"latest": {"fixture": "v1"}}),
        encoding="utf-8",
    )

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "fixture" in msg
    assert "file missing" in msg or "missing" in msg


def test_registry_missing_entirely_raises(tmp_path: Path) -> None:
    """registry.yaml does not exist at all."""
    root = tmp_path / "rubrics"
    root.mkdir()
    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "registry.yaml" in msg


def test_missing_latest_entry_for_rubric_raises(tmp_path: Path) -> None:
    """registry.yaml exists but has no 'latest' entry for this rubric."""
    root = tmp_path / "rubrics"
    root.mkdir()
    (root / "registry.yaml").write_text(
        yaml.safe_dump({"latest": {"other_rubric": "v1"}}),
        encoding="utf-8",
    )

    with pytest.raises(UngovernedRubricError) as exc:
        check_rubrics_governed("fixture", rubrics_root=root)
    msg = str(exc.value)
    assert "fixture" in msg
    assert "latest" in msg
