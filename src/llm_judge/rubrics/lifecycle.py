"""Rubric lifecycle module — parallels rules/lifecycle.py.

Provides the governance schema (RubricLifecycleEntry,
RubricAuditEvent, RubricStatus) and helpers for walking the
rubrics/ directory. The strict preflight check
(check_rubrics_governed) and the UngovernedRubricError raising
path are added in Commit 4 of the CP-1c-a packet.

Pattern parallel: src/llm_judge/rules/lifecycle.py.
"""

from __future__ import annotations

import logging
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

# NOTE: future — move to platform config once CP-2
# observability lands. Decision belongs to that
# architecture chat.
REVIEW_PERIOD_DAYS = 365


def _project_root() -> Path:
    # src/llm_judge/rubrics/lifecycle.py → parents[3] is repo root
    return Path(__file__).resolve().parents[3]


def _default_rubrics_root() -> Path:
    return _project_root() / "rubrics"


class RubricStatus(str, Enum):
    DRAFT = "draft"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


class RubricAuditEvent(BaseModel):
    """A single governance audit event attached to a rubric version."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    actor: str
    action: str  # e.g. "created", "status_changed", "migrated_to_lifecycle_v1"
    details: dict[str, Any] = Field(default_factory=dict)


class RubricLifecycleEntry(BaseModel):
    """Governance record for one rubric version.

    Loaded from ``rubrics/<rubric_id>/<version>.yaml``. The YAML
    file may carry additional fields (dimensions, decision_policy,
    scale, title) that ``rubric_store`` consumes — this model is
    permissive about extras so lifecycle validation does not
    require the rubric's scoring schema to be understood here.
    """

    model_config = ConfigDict(frozen=True, extra="allow")

    rubric_id: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    status: RubricStatus
    owner: str = Field(..., min_length=1)
    created_at: datetime
    last_reviewed: datetime
    deprecation_warning_days: int | None = None
    audit_log: list[RubricAuditEvent] = Field(default_factory=list)


class UngovernedRubricError(Exception):
    """Raised when a rubric fails the preflight governance check.

    Message names the specific violation AND the file path so the
    operator can correct it without grepping. The check itself is
    defined in Commit 4 of the CP-1c-a packet.
    """


def _load_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid rubric YAML (expected mapping): {path}")
    return raw


def _rubric_file(rubric_id: str, version: str, rubrics_root: Path) -> Path:
    return rubrics_root / rubric_id / f"{version}.yaml"


def _read_entry(path: Path) -> RubricLifecycleEntry:
    """Load a rubric YAML file as a RubricLifecycleEntry.

    Raises ``pydantic.ValidationError`` if governance fields are
    absent or malformed. Callers (e.g. ``list_rubrics``) decide
    whether to swallow, log, or propagate.
    """
    raw = _load_yaml(path)
    return RubricLifecycleEntry.model_validate(raw)


def list_rubrics(
    rubrics_root: Path | None = None,
) -> list[RubricLifecycleEntry]:
    """List all governed rubrics by walking
    ``rubrics/<rubric_id>/<version>.yaml``.

    Parallels ``rules.lifecycle.load_manifest`` / ``list_rules``.
    Returns entries sorted by (rubric_id, version). Files that
    fail to parse are logged and skipped; the strict contract is
    in :func:`check_rubrics_governed` (Commit 4), which raises
    loudly.
    """
    root = (rubrics_root or _default_rubrics_root()).resolve()
    if not root.exists() or not root.is_dir():
        return []

    entries: list[RubricLifecycleEntry] = []
    for rubric_dir in sorted(root.iterdir()):
        if not rubric_dir.is_dir():
            continue
        for version_file in sorted(rubric_dir.glob("*.yaml")):
            try:
                entries.append(_read_entry(version_file))
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "rubrics.lifecycle.parse_failed",
                    extra={
                        "path": str(version_file),
                        "error": str(exc)[:200],
                    },
                )
    return entries


def get_rubric_status(
    rubric_id: str,
    version: str,
    *,
    rubrics_root: Path | None = None,
) -> RubricStatus:
    """Return the lifecycle status of a specific rubric version.

    Raises ``FileNotFoundError`` if the rubric file is absent or
    ``pydantic.ValidationError`` if governance fields are malformed.
    """
    root = (rubrics_root or _default_rubrics_root()).resolve()
    path = _rubric_file(rubric_id, version, root)
    if not path.exists():
        raise FileNotFoundError(f"Missing rubric file: {path}")
    return _read_entry(path).status


# check_rubrics_governed defined in Commit 4 (CP-1c-a).
