"""Rubric resolution with Pydantic-validated YAML shapes.

Public entry point: :func:`get_rubric`. Callers receive the same
frozen ``Rubric`` dataclass they always have — the refactor is
internal: registry loading validates via ``RubricRegistryConfig``
and rubric-definition files validate their governance fields via
``RubricLifecycleEntry``. Schema failures surface as
:class:`RubricSchemaError` with the offending file path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from llm_judge.rubric_yaml import RubricRegistryConfig, rubric_definition_model


class RubricSchemaError(ValueError):
    """Raised when a rubric / registry YAML fails schema validation.

    Subclasses ``ValueError`` so existing callers that ``except
    ValueError:`` around ``get_rubric`` continue to work unchanged.
    The message always names the file path so the operator can
    correct the offending YAML without grepping.
    """


@dataclass(frozen=True)
class Rubric:
    rubric_id: str
    version: str
    dimensions: list[str]
    pass_if_overall_score_gte: float
    fail_if_any_dimension_lte: int

    # EPIC-2: governance contract for metrics.json
    metrics_required: list[str]
    metrics_optional: list[str]


def _project_root() -> Path:
    # src/llm_judge/rubric_store.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    """Low-level YAML→dict read. Raises ``RubricSchemaError`` with the
    file path on malformed YAML; never returns non-mappings."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RubricSchemaError(
            f"Malformed YAML in {path}: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise RubricSchemaError(
            f"Invalid YAML structure in {path} (expected mapping)"
        )
    return data


def _registry_path() -> Path:
    return _project_root() / "rubrics" / "registry.yaml"


def _load_registry() -> RubricRegistryConfig:
    """Load and validate ``rubrics/registry.yaml`` via
    :class:`RubricRegistryConfig`.

    Returns the validated model. Raises :class:`RubricSchemaError`
    with the file path on a malformed registry.
    """
    path = _registry_path()
    if not path.exists():
        raise RubricSchemaError(f"Missing rubric registry: {path}")

    raw = _load_yaml(path)
    try:
        return RubricRegistryConfig.model_validate(raw)
    except ValidationError as exc:
        raise RubricSchemaError(
            f"Rubric registry failed schema validation at {path}: {exc}"
        ) from exc


def _resolve_version(rubric_id: str) -> str:
    """Resolve the latest version for ``rubric_id`` from
    ``rubrics/registry.yaml``.

    Raises :class:`ValueError` (a plain ``ValueError``, not
    ``RubricSchemaError``) if the registry has no latest pointer
    for this rubric — that is a governance gap, not a schema
    malformation.
    """
    registry = _load_registry()
    if rubric_id not in registry.latest:
        raise ValueError(
            f"Rubric not registered in registry.yaml latest: {rubric_id}"
        )
    v = registry.latest[rubric_id]
    if not v.strip():
        raise ValueError(
            f"Invalid latest version for rubric '{rubric_id}': {v!r}"
        )
    return v.strip()


def _resolve_metrics_schema_from_registry(
    rubric_id: str, version: str
) -> tuple[list[str], list[str]]:
    """EPIC-2: Resolve declared metric schema for a rubric version via
    the validated registry. Returns ``(required, optional)`` lists.

    Returns empty lists when the registry does not declare a
    schema for this version (backward compatibility).
    """
    registry = _load_registry()
    schema = registry.metrics_schema_for(rubric_id, version)
    if schema is None:
        return [], []

    overlap = set(schema.required) & set(schema.optional)
    if overlap:
        raise RubricSchemaError(
            f"Registry metrics_schema for {rubric_id}@{version} has "
            f"overlap between required and optional: "
            f"{sorted(overlap)}"
        )

    # Dedupe while preserving order.
    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x and x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return _dedupe(schema.required), _dedupe(schema.optional)


def _validate_rubric_definition(path: Path, raw: dict[str, Any]) -> None:
    """Run the rubric definition file's governance schema through
    :class:`RubricLifecycleEntry`. Raises :class:`RubricSchemaError`
    on validation failure, naming the file path.

    The validation is advisory within this function: the rubric
    definition model uses ``extra='allow'`` so scoring fields
    (dimensions, decision_policy, scale) are accepted alongside
    governance fields.
    """
    model_cls = rubric_definition_model()
    try:
        model_cls.model_validate(raw)
    except ValidationError as exc:
        raise RubricSchemaError(
            f"Rubric definition failed schema validation at {path}: {exc}"
        ) from exc


def get_rubric(rubric_ref: str) -> Rubric:
    """Resolve a rubric by id or ``id@version``.

    rubric_ref formats supported:
      - ``"chat_quality"``      → resolves to latest version via ``rubrics/registry.yaml``
      - ``"chat_quality@v1"``   → explicit version

    Returns a :class:`Rubric` with dimensions + decision policy
    (from ``rubrics/<id>/<version>.yaml``) plus metrics schema
    contract (from the registry if declared).
    """
    if "@" in rubric_ref:
        rubric_id, version = rubric_ref.split("@", 1)
        rubric_id = rubric_id.strip()
        version = version.strip()
        if not rubric_id or not version:
            raise ValueError(f"Invalid rubric_ref: {rubric_ref!r}")
    else:
        rubric_id = rubric_ref.strip()
        if not rubric_id:
            raise ValueError(f"Invalid rubric_ref: {rubric_ref!r}")
        version = _resolve_version(rubric_id)

    rubric_path = _project_root() / "rubrics" / rubric_id / f"{version}.yaml"
    if not rubric_path.exists():
        raise ValueError(f"Rubric file not found: {rubric_path}")

    raw = _load_yaml(rubric_path)

    # Governance validation via Pydantic. Raises RubricSchemaError
    # naming the file path on malformed governance fields.
    _validate_rubric_definition(rubric_path, raw)

    dims_raw = raw.get("dimensions", [])
    if not isinstance(dims_raw, list) or not dims_raw:
        raise RubricSchemaError(
            f"Rubric dimensions must be a non-empty list in {rubric_path}"
        )

    dimensions: list[str] = []
    for d in dims_raw:
        if not isinstance(d, dict) or "name" not in d:
            raise RubricSchemaError(
                f"Each dimension must be an object with a 'name' in {rubric_path}"
            )
        name = d["name"]
        if not isinstance(name, str) or not name.strip():
            raise RubricSchemaError(
                f"dimension.name must be a non-empty string in {rubric_path}"
            )
        dimensions.append(name.strip())

    policy = raw.get("decision_policy", {})
    if not isinstance(policy, dict):
        raise RubricSchemaError(
            f"decision_policy must be an object in {rubric_path}"
        )

    # Rubric self-declared metadata (allow fallback to resolved id/version)
    resolved_rubric_id = str(raw.get("rubric_id", rubric_id))
    resolved_version = str(raw.get("version", version))

    # Governance: registry-declared metrics schema (optional until you extend registry.yaml)
    metrics_required, metrics_optional = _resolve_metrics_schema_from_registry(
        rubric_id, version
    )

    return Rubric(
        rubric_id=resolved_rubric_id,
        version=resolved_version,
        dimensions=dimensions,
        pass_if_overall_score_gte=float(policy.get("pass_if_overall_score_gte", 3.0)),
        fail_if_any_dimension_lte=int(policy.get("fail_if_any_dimension_lte", 1)),
        metrics_required=metrics_required,
        metrics_optional=metrics_optional,
    )
