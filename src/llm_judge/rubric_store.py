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


def _project_root() -> Path:
    # src/llm_judge/rubric_store.py -> project root is parents[2]
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def _resolve_version(rubric_id: str) -> str:
    registry_path = _project_root() / "rubrics" / "registry.yaml"
    registry = _load_yaml(registry_path)
    latest = registry.get("latest", {})
    if not isinstance(latest, dict) or rubric_id not in latest:
        raise ValueError(f"Rubric not registered: {rubric_id}")
    return str(latest[rubric_id])


def get_rubric(rubric_ref: str) -> Rubric:
    """
    rubric_ref formats supported:
      - "chat_quality"      -> resolves to latest version via registry.yaml
      - "chat_quality@v1"   -> explicit version
    """
    if "@" in rubric_ref:
        rubric_id, version = rubric_ref.split("@", 1)
    else:
        rubric_id = rubric_ref
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
        dimensions.append(str(d["name"]))

    policy = raw.get("decision_policy", {})
    if not isinstance(policy, dict):
        raise ValueError("decision_policy must be an object")

    return Rubric(
        rubric_id=str(raw.get("rubric_id", rubric_id)),
        version=str(raw.get("version", version)),
        dimensions=dimensions,
        pass_if_overall_score_gte=float(policy.get("pass_if_overall_score_gte", 3.0)),
        fail_if_any_dimension_lte=int(policy.get("fail_if_any_dimension_lte", 1)),
    )
