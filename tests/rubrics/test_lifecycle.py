"""Tests for ``llm_judge.rubrics.lifecycle`` — schema, lifecycle
walk, per-version status lookup, constant surface."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from llm_judge.rubrics import (
    REVIEW_PERIOD_DAYS,
    RubricAuditEvent,
    RubricLifecycleEntry,
    RubricStatus,
    get_rubric_status,
    list_rubrics,
)

# =====================================================================
# RubricStatus enum
# =====================================================================


def test_rubric_status_has_four_values() -> None:
    names = {s.name for s in RubricStatus}
    assert names == {"DRAFT", "VALIDATED", "PRODUCTION", "DEPRECATED"}
    values = {s.value for s in RubricStatus}
    assert values == {"draft", "validated", "production", "deprecated"}


def test_rubric_status_is_str_enum() -> None:
    # Pydantic coercion depends on RubricStatus behaving as a string enum.
    assert RubricStatus.PRODUCTION == "production"


# =====================================================================
# REVIEW_PERIOD_DAYS surface
# =====================================================================


def test_review_period_days_exposed_and_equals_365() -> None:
    assert REVIEW_PERIOD_DAYS == 365


# =====================================================================
# RubricLifecycleEntry validation
# =====================================================================


def _valid_entry_kwargs() -> dict[str, Any]:
    return {
        "rubric_id": "fixture",
        "version": "v1",
        "status": "production",
        "owner": "test-owner",
        "created_at": "2026-01-01",
        "last_reviewed": "2026-04-01",
        "audit_log": [
            {
                "timestamp": "2026-04-01T00:00:00Z",
                "actor": "test",
                "action": "created",
                "details": {},
            }
        ],
    }


def test_lifecycle_entry_accepts_valid_fields() -> None:
    entry = RubricLifecycleEntry(**_valid_entry_kwargs())
    assert entry.rubric_id == "fixture"
    assert entry.status == RubricStatus.PRODUCTION
    assert isinstance(entry.created_at, datetime)
    assert len(entry.audit_log) == 1
    assert entry.audit_log[0].action == "created"


def test_lifecycle_entry_rejects_empty_owner() -> None:
    kwargs = _valid_entry_kwargs()
    kwargs["owner"] = ""
    with pytest.raises(ValidationError):
        RubricLifecycleEntry(**kwargs)


def test_lifecycle_entry_rejects_missing_status() -> None:
    kwargs = _valid_entry_kwargs()
    del kwargs["status"]
    with pytest.raises(ValidationError):
        RubricLifecycleEntry(**kwargs)


def test_lifecycle_entry_allows_extra_fields() -> None:
    """extra='allow' lets rubric YAML keep scale/decision_policy/
    dimensions alongside governance fields."""
    kwargs = _valid_entry_kwargs()
    kwargs["title"] = "Some Title"
    kwargs["dimensions"] = [{"name": "relevance"}]
    entry = RubricLifecycleEntry(**kwargs)
    # Confirm governance fields parsed correctly; extras are silently kept.
    assert entry.owner == "test-owner"


def test_lifecycle_entry_is_frozen() -> None:
    entry = RubricLifecycleEntry(**_valid_entry_kwargs())
    with pytest.raises(ValidationError):
        entry.owner = "someone-else"  # type: ignore[misc]


# =====================================================================
# RubricAuditEvent
# =====================================================================


def test_audit_event_is_frozen() -> None:
    event = RubricAuditEvent(
        timestamp=datetime(2026, 4, 23, tzinfo=timezone.utc),
        actor="tester",
        action="created",
    )
    with pytest.raises(ValidationError):
        event.actor = "other"  # type: ignore[misc]


def test_audit_event_details_default_empty() -> None:
    event = RubricAuditEvent(
        timestamp=datetime(2026, 4, 23, tzinfo=timezone.utc),
        actor="tester",
        action="created",
    )
    assert event.details == {}


# =====================================================================
# list_rubrics
# =====================================================================


def _write_rubric(root: Path, rubric_id: str, version: str) -> None:
    (root / rubric_id).mkdir(parents=True, exist_ok=True)
    content = _valid_entry_kwargs()
    content["rubric_id"] = rubric_id
    content["version"] = version
    (root / rubric_id / f"{version}.yaml").write_text(
        yaml.safe_dump(content), encoding="utf-8"
    )


def test_list_rubrics_returns_entries_from_directory(tmp_path: Path) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()
    _write_rubric(root, "alpha", "v1")
    _write_rubric(root, "beta", "v1")
    _write_rubric(root, "beta", "v2")

    entries = list_rubrics(rubrics_root=root)
    assert len(entries) == 3
    ids = [(e.rubric_id, e.version) for e in entries]
    assert ("alpha", "v1") in ids
    assert ("beta", "v1") in ids
    assert ("beta", "v2") in ids


def test_list_rubrics_sorted_by_path(tmp_path: Path) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()
    _write_rubric(root, "zebra", "v1")
    _write_rubric(root, "alpha", "v1")

    entries = list_rubrics(rubrics_root=root)
    # iterdir+glob are sorted, so alpha before zebra.
    assert entries[0].rubric_id == "alpha"
    assert entries[-1].rubric_id == "zebra"


def test_list_rubrics_skips_malformed_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()
    _write_rubric(root, "good", "v1")

    (root / "bad").mkdir()
    (root / "bad" / "v1.yaml").write_text(
        yaml.safe_dump({"rubric_id": "bad"}),  # missing governance fields
        encoding="utf-8",
    )

    entries = list_rubrics(rubrics_root=root)
    # bad is skipped, good survives.
    assert len(entries) == 1
    assert entries[0].rubric_id == "good"


def test_list_rubrics_returns_empty_for_missing_root(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    assert list_rubrics(rubrics_root=missing) == []


# =====================================================================
# get_rubric_status
# =====================================================================


def test_get_rubric_status_returns_status(tmp_path: Path) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()
    _write_rubric(root, "alpha", "v1")
    status = get_rubric_status("alpha", "v1", rubrics_root=root)
    assert status == RubricStatus.PRODUCTION


def test_get_rubric_status_raises_file_not_found(tmp_path: Path) -> None:
    root = tmp_path / "rubrics"
    root.mkdir()
    with pytest.raises(FileNotFoundError):
        get_rubric_status("missing", "v1", rubrics_root=root)


def test_get_rubric_status_raises_validation_error_on_malformed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "rubrics"
    (root / "bad").mkdir(parents=True)
    (root / "bad" / "v1.yaml").write_text(
        yaml.safe_dump({"rubric_id": "bad"}),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        get_rubric_status("bad", "v1", rubrics_root=root)
