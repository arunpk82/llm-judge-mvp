from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from llm_judge.rules.registry import RULE_REGISTRY
from llm_judge.rules.types import Flag, RuleContext, RuleResult


@dataclass(frozen=True)
class RuleSpec:
    id: str
    enabled: bool
    params: dict[str, Any]


@dataclass(frozen=True)
class RulePlan:
    rubric_id: str
    version: str
    rules: list[RuleSpec]


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root (expected mapping): {path}")
    return data


def _parse_plan(data: dict[str, Any]) -> RulePlan:
    rubric_id = str(data.get("rubric_id"))
    version = str(data.get("version"))
    rules_raw = data.get("rules")
    if not isinstance(rules_raw, list):
        raise ValueError("rules must be a list")

    specs: list[RuleSpec] = []
    for r in rules_raw:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("id"))
        enabled = bool(r.get("enabled", True))
        params = r.get("params")
        if not isinstance(params, dict):
            params = {}
        specs.append(RuleSpec(id=rid, enabled=enabled, params=dict(params)))

    return RulePlan(rubric_id=rubric_id, version=version, rules=specs)


@lru_cache(maxsize=32)
def load_plan_for_rubric(rubric_id: str, version: str) -> RulePlan:
    path = Path("configs/rules") / f"{rubric_id}_{version}.yaml"
    data = _load_yaml(path)
    plan = _parse_plan(data)
    if plan.rubric_id != rubric_id or plan.version != version:
        raise ValueError(f"Rule plan mismatch for {rubric_id}/{version}: {path}")
    return plan


def run_rules(ctx: RuleContext, plan: RulePlan) -> RuleResult:
    flags: list[Flag] = []
    for spec in plan.rules:
        if not spec.enabled:
            continue
        fn = RULE_REGISTRY.get(spec.id)
        if fn is None:
            # Deterministic warning flag instead of raising.
            flags.append(
                Flag(
                    id="engine.unknown_rule",
                    severity="weak",
                    details={"rule_id": spec.id},
                    evidence=[json.dumps({"rule_id": spec.id})],
                )
            )
            continue
        out = fn(ctx, spec.params)
        flags.extend(out.flags)

    # Stable ordering for determinism
    flags = sorted(flags, key=lambda f: (f.id, f.severity, json.dumps(f.details, sort_keys=True)))
    return RuleResult(flags=flags)
