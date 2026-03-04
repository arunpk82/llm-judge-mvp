from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


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
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def _registry_path() -> Path:
    return _project_root() / "rubrics" / "registry.yaml"


def _load_registry() -> dict[str, Any]:
    path = _registry_path()
    if not path.exists():
        raise ValueError(f"Missing rubric registry: {path}")
    return _load_yaml(path)


def _resolve_version(rubric_id: str) -> str:
    """
    Resolve latest version using registry.yaml:

    Backward-compatible contract:
      registry.yaml must contain:
        latest:
          <rubric_id>: <version>
    """
    registry = _load_registry()
    latest = registry.get("latest", {})
    if not isinstance(latest, dict) or rubric_id not in latest:
        raise ValueError(f"Rubric not registered in registry.yaml latest: {rubric_id}")
    v = latest[rubric_id]
    if not isinstance(v, (str, int, float)) or not str(v).strip():
        raise ValueError(f"Invalid latest version for rubric '{rubric_id}': {v!r}")
    return str(v)


def _normalize_str_list(value: Any, *, ctx: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{ctx} must be a list of strings")
    out: list[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{ctx}[{i}] must be a non-empty string")
        out.append(item.strip())
    # de-dupe while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def _resolve_metrics_schema_from_registry(rubric_id: str, version: str) -> tuple[list[str], list[str]]:
    """
    EPIC-2: Resolve declared metric schema for a rubric version.

    Supports extended registry.yaml format:

      rubrics:
        chat_quality:
          versions:
            v1:
              metrics_schema:
                required: [...]
                optional: [...]

    Backward compatibility:
      - If 'rubrics' block doesn't exist, return empty schema (no enforcement possible).
      - If rubric/version exists but metrics_schema absent, return empty lists.

    Note:
      Enforcement should happen in eval.run (or a validator) using this schema.
    """
    registry = _load_registry()
    rubrics = registry.get("rubrics")
    if rubrics is None:
        return [], []
    if not isinstance(rubrics, dict):
        raise ValueError("registry.yaml: 'rubrics' must be a mapping if present")

    r = rubrics.get(rubric_id)
    if r is None:
        return [], []
    if not isinstance(r, dict):
        raise ValueError(f"registry.yaml: rubrics.{rubric_id} must be a mapping")

    versions = r.get("versions")
    if versions is None:
        return [], []
    if not isinstance(versions, dict):
        raise ValueError(f"registry.yaml: rubrics.{rubric_id}.versions must be a mapping")

    v = versions.get(version)
    if v is None:
        return [], []
    if not isinstance(v, dict):
        raise ValueError(f"registry.yaml: rubrics.{rubric_id}.versions.{version} must be a mapping")

    ms = v.get("metrics_schema")
    if ms is None:
        return [], []
    if not isinstance(ms, dict):
        raise ValueError(
            f"registry.yaml: rubrics.{rubric_id}.versions.{version}.metrics_schema must be a mapping"
        )

    required = _normalize_str_list(ms.get("required"), ctx=f"registry.yaml metrics_schema.required for {rubric_id}@{version}")
    optional = _normalize_str_list(ms.get("optional"), ctx=f"registry.yaml metrics_schema.optional for {rubric_id}@{version}")

    # Ensure no overlap
    overlap = set(required) & set(optional)
    if overlap:
        raise ValueError(
            f"registry.yaml: metrics_schema for {rubric_id}@{version} has overlap between required and optional: "
            f"{sorted(list(overlap))}"
        )

    return required, optional


def get_rubric(rubric_ref: str) -> Rubric:
    """
    rubric_ref formats supported:
      - "chat_quality"      -> resolves to latest version via registry.yaml
      - "chat_quality@v1"   -> explicit version

    Returns a Rubric with:
      - dimensions + decision policy (from rubrics/<id>/<version>.yaml)
      - metrics schema contract (from registry.yaml rubrics block, if present)
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

    dims_raw = raw.get("dimensions", [])
    if not isinstance(dims_raw, list) or not dims_raw:
        raise ValueError("Rubric dimensions must be a non-empty list")

    dimensions: list[str] = []
    for d in dims_raw:
        if not isinstance(d, dict) or "name" not in d:
            raise ValueError("Each dimension must be an object with a 'name'")
        name = d["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("dimension.name must be a non-empty string")
        dimensions.append(name.strip())

    policy = raw.get("decision_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("decision_policy must be an object")

    # Rubric self-declared metadata (allow fallback to resolved id/version)
    resolved_rubric_id = str(raw.get("rubric_id", rubric_id))
    resolved_version = str(raw.get("version", version))

    # Governance: registry-declared metrics schema (optional until you extend registry.yaml)
    metrics_required, metrics_optional = _resolve_metrics_schema_from_registry(rubric_id, version)

    return Rubric(
        rubric_id=resolved_rubric_id,
        version=resolved_version,
        dimensions=dimensions,
        pass_if_overall_score_gte=float(policy.get("pass_if_overall_score_gte", 3.0)),
        fail_if_any_dimension_lte=int(policy.get("fail_if_any_dimension_lte", 1)),
        metrics_required=metrics_required,
        metrics_optional=metrics_optional,
    )