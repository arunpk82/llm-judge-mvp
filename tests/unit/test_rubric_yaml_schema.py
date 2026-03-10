from pathlib import Path

import yaml

from llm_judge.rubric_yaml import RubricRegistryConfig


def _is_registry_yaml(data: object) -> bool:
    """
    Registry YAMLs are expected to have a top-level key `latest`
    mapping rubric_name -> version string.
    """
    if not isinstance(data, dict):
        return False
    if "latest" not in data:
        return False
    return isinstance(data.get("latest"), dict)


def test_rubric_registry_yamls_validate() -> None:
    rubric_dir = Path("rubrics")
    if not rubric_dir.exists():
        return

    yaml_files = sorted(list(rubric_dir.rglob("*.yaml")) + list(rubric_dir.rglob("*.yml")))
    assert yaml_files, "No YAML files found under ./rubrics"

    validated = 0

    for file in yaml_files:
        with file.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Only validate registry-style yaml in this repo (e.g., latest.yaml)
        if not _is_registry_yaml(data):
            continue

        RubricRegistryConfig(**data)
        validated += 1

    # In this repo, rubrics/ appears to contain registry YAMLs (not full rubric definitions).
    assert validated > 0, (
        "No rubric registry YAMLs validated. If your repo stores rubrics elsewhere, "
        "update this test to match that structure."
    )
