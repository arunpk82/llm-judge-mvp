"""Schema-level tests for ``load_plan_for_rubric``'s Pydantic wiring.

The engine parses ``configs/rules/<id>/<version>.yaml`` through
:class:`RulePlanConfig` (CP-1d Commit 2b). Malformed plans now raise
:class:`RulePlanSchemaError` with the offending file path in the
message — replacing the previous ad-hoc dict parsing that silently
dropped bad rows.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llm_judge.rules.engine import RulePlanSchemaError, load_plan_for_rubric


def _write_plan(tmp_path: Path, rubric_id: str, version: str, body: dict) -> Path:
    """Write a rule plan into a temporary ``configs/rules/`` tree and
    return the tmp path so the caller can ``monkeypatch.chdir`` to
    it (``load_plan_for_rubric`` resolves via ``config_root()``)."""
    plan_dir = tmp_path / "configs" / "rules" / rubric_id
    plan_dir.mkdir(parents=True, exist_ok=True)
    (plan_dir / f"{version}.yaml").write_text(
        yaml.safe_dump(body), encoding="utf-8"
    )
    return tmp_path


def test_load_plan_raises_schema_error_on_non_string_rule_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rule entry whose ``id`` is not a string fails the Pydantic
    schema check at the engine boundary."""
    _write_plan(
        tmp_path,
        "fixture_rubric",
        "v1",
        body={
            "rubric_id": "fixture_rubric",
            "version": "v1",
            "rules": [
                # id must be a non-empty string — here it is an int.
                {"id": 12345, "enabled": True, "params": {}}
            ],
        },
    )
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RulePlanSchemaError) as exc:
        load_plan_for_rubric("fixture_rubric", "v1")

    msg = str(exc.value)
    # Message includes the offending path so the operator can fix.
    assert "fixture_rubric" in msg
    assert "v1.yaml" in msg
