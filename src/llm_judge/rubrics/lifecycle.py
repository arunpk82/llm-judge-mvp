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
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

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


def _load_registry(rubrics_root: Path) -> dict[str, Any]:
    """Load ``rubrics/registry.yaml`` or raise ``UngovernedRubricError``
    with a concrete fix hint."""
    registry_path = rubrics_root / "registry.yaml"
    if not registry_path.exists():
        raise UngovernedRubricError(
            f"Rubric registry missing: {registry_path}. "
            f"Fix: create rubrics/registry.yaml with a 'latest:' map "
            f"naming each rubric's canonical version."
        )
    return _load_yaml(registry_path)


def _resolve_latest_version(
    rubric_id: str, rubrics_root: Path
) -> str:
    registry = _load_registry(rubrics_root)
    latest = registry.get("latest")
    if not isinstance(latest, dict) or rubric_id not in latest:
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}' has no 'latest' entry in "
            f"{rubrics_root / 'registry.yaml'}. Fix: add "
            f"'{rubric_id}: <version>' under 'latest:'."
        )
    value = latest[rubric_id]
    if not isinstance(value, (str, int, float)) or not str(value).strip():
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}' has an invalid latest version "
            f"in registry.yaml: {value!r}. Fix: set a non-empty "
            f"version string."
        )
    return str(value).strip()


def _age_in_days(reviewed: datetime) -> int:
    """Days between ``reviewed`` (datetime, possibly naive) and today.

    Uses date arithmetic so naive/aware timezone differences are
    irrelevant for a day-granularity review window.
    """
    reviewed_date: date = (
        reviewed.date() if isinstance(reviewed, datetime) else reviewed
    )
    return (date.today() - reviewed_date).days


def check_rubrics_governed(
    rubric_id: str,
    version: str | None = None,
    *,
    rubrics_root: Path | None = None,
) -> None:
    """Strict preflight. Raises :class:`UngovernedRubricError` with a
    message naming the violation AND the file path.

    Strict criteria:
      1. ``rubrics/registry.yaml`` exists and carries a ``latest:``
         entry for ``rubric_id`` (if ``version`` is ``None``).
      2. The resolved rubric file at
         ``rubrics/<rubric_id>/<version>.yaml`` exists.
      3. Governance fields validate against
         :class:`RubricLifecycleEntry` (owner, created_at,
         last_reviewed, status, audit_log).
      4. ``status`` is one of ``validated`` or ``production``.
      5. ``last_reviewed`` is within :data:`REVIEW_PERIOD_DAYS`.
      6. ``dimensions`` is a non-empty list and each element carries
         a non-empty ``name`` (structural check; see scope note).

    Scope note on "CAP-7 dimension set": the packet brief called
    for checking that every rubric dimension exists in CAP-7's
    canonical dimension set. CAP-7 (the hallucination pipeline)
    does not today publish a canonical set — the closest list is
    :mod:`llm_judge.calibration.__init__`'s default
    ``["relevance","clarity","correctness","tone"]``, which is not
    a governance surface. Rather than invent a capability
    contract CAP-7 does not expose, this check is scoped to the
    structural shape of ``dimensions``. A semantic dimension-set
    check belongs with CP-1c-b's capability-contract work.
    """
    root = (rubrics_root or _default_rubrics_root()).resolve()

    # 1. registry + version resolution
    if version is None:
        version = _resolve_latest_version(rubric_id, root)

    # 2. rubric file presence
    path = _rubric_file(rubric_id, version, root)
    if not path.exists():
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}@{version}' file missing: {path}. "
            f"Fix: create {path} with the lifecycle schema "
            f"(rubric_id, version, status, owner, created_at, "
            f"last_reviewed, audit_log)."
        )

    raw = _load_yaml(path)

    # 3. governance fields (Pydantic shape)
    try:
        entry = RubricLifecycleEntry.model_validate(raw)
    except ValidationError as exc:
        missing = sorted(
            {
                ".".join(str(p) for p in err.get("loc", ()))
                for err in exc.errors()
            }
        )
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}@{version}' failed governance "
            f"schema: {missing}. Fix: add/repair these fields in "
            f"{path}. See rubrics/{rubric_id}/{version}.yaml."
        ) from exc

    # 4. status gate
    if entry.status not in (
        RubricStatus.VALIDATED,
        RubricStatus.PRODUCTION,
    ):
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}@{version}' status is "
            f"'{entry.status.value}'; must be 'validated' or "
            f"'production' to pass preflight. Fix: update status "
            f"in {path} (and record the change in audit_log)."
        )

    # 5. review freshness
    age_days = _age_in_days(entry.last_reviewed)
    if age_days > REVIEW_PERIOD_DAYS:
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}@{version}' last reviewed "
            f"{age_days} days ago (> {REVIEW_PERIOD_DAYS}-day "
            f"review period). Fix: refresh last_reviewed in "
            f"{path} and append an audit_log entry recording the "
            f"review."
        )

    # 6. dimensions structural shape
    dims = raw.get("dimensions")
    if not isinstance(dims, list) or not dims:
        raise UngovernedRubricError(
            f"Rubric '{rubric_id}@{version}' must declare a "
            f"non-empty 'dimensions' list. Fix: add at least one "
            f"dimension with a 'name' in {path}."
        )
    for i, d in enumerate(dims):
        name: Any
        if isinstance(d, dict):
            name = d.get("name")
        elif isinstance(d, str):
            name = d
        else:
            name = None
        if not isinstance(name, str) or not name.strip():
            raise UngovernedRubricError(
                f"Rubric '{rubric_id}@{version}' dimension[{i}] "
                f"has no non-empty 'name'. Fix: each dimension in "
                f"{path} must carry a string 'name'."
            )
