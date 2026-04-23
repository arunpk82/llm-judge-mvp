"""Migration verification: every rubric currently in the repo passes
preflight, and the two migrated rubrics carry the CP-1c-a audit_log
entry."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_judge.rubrics import (
    RubricLifecycleEntry,
    check_rubrics_governed,
    list_rubrics,
)

REPO_RUBRICS_ROOT = Path(__file__).resolve().parents[2] / "rubrics"


def test_every_rubric_passes_preflight() -> None:
    """Walk the real rubrics/ directory and call
    check_rubrics_governed on each entry — all must pass."""
    entries = list_rubrics(rubrics_root=REPO_RUBRICS_ROOT)
    assert entries, "No rubrics found in repo — test assumes at least one"

    for entry in entries:
        # Strict preflight must not raise.
        check_rubrics_governed(
            entry.rubric_id,
            entry.version,
            rubrics_root=REPO_RUBRICS_ROOT,
        )


def test_migrated_rubrics_carry_cp_1c_a_audit_event() -> None:
    """chat_quality/v1 and math_basic/v1 were migrated in Commit 4.
    Each must carry an audit_log entry with action
    'migrated_to_lifecycle_v1' and actor 'cp-1c-a-migration'."""
    for rubric_id in ("chat_quality", "math_basic"):
        path = REPO_RUBRICS_ROOT / rubric_id / "v1.yaml"
        assert path.exists(), f"expected rubric file: {path}"
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = RubricLifecycleEntry.model_validate(raw)

        migration_events = [
            e
            for e in entry.audit_log
            if e.action == "migrated_to_lifecycle_v1"
        ]
        assert migration_events, (
            f"{rubric_id}/v1 missing migrated_to_lifecycle_v1 "
            f"audit_log entry"
        )
        assert migration_events[0].actor == "cp-1c-a-migration"
