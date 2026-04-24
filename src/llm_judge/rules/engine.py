from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from llm_judge.paths import config_root
from llm_judge.rule_plan_yaml import RulePlanConfig
from llm_judge.rules.registry import get_rule
from llm_judge.rules.types import RuleContext


class RulePlanSchemaError(ValueError):
    """Raised when a rule-plan YAML fails Pydantic schema validation.

    Subclasses ``ValueError`` so existing callers that catch
    ``ValueError`` around ``load_plan_for_rubric`` continue to work
    unchanged. The message names the file path so operators can
    correct the offending YAML without grepping.
    """

logger = logging.getLogger(__name__)

# RuleResult is a project type; we try to construct it if possible.
# If its constructor signature changes, we fall back to a simple object with .flags.
try:
    from llm_judge.rules.types import RuleResult
except Exception:  # pragma: no cover
    RuleResult = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class RuleSpec:
    id: str
    enabled: bool = True
    params: dict[str, Any] | None = None


@dataclass(frozen=True)
class RulePlan:
    rubric_id: str
    version: str
    rules: list[dict[str, Any]]


def _as_rule_id(spec: object) -> str | None:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        rid = spec.get("id")
        return rid if isinstance(rid, str) else None
    rid = getattr(spec, "id", None)
    return rid if isinstance(rid, str) else None


def _as_rule_params(spec: object) -> dict[str, Any]:
    if isinstance(spec, Mapping):
        params = spec.get("params")
        return params if isinstance(params, dict) else {}
    params = getattr(spec, "params", None)
    return params if isinstance(params, dict) else {}


def _resolve_rule(rule_id: str):
    """
    Resolve from registry.
    Supports fully-qualified ids (quality.nonsense_basic) and short ids (nonsense_basic).
    """
    try:
        return get_rule(rule_id), rule_id
    except Exception:
        if "." in rule_id:
            short_id = rule_id.split(".", 1)[-1]
            return get_rule(short_id), rule_id  # keep invoked_id for prefixing flags
        raise


def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load YAML using PyYAML if available; otherwise use a minimal parser sufficient
    for our test YAML structure.
    """
    try:
        import yaml
    except Exception:  # pragma: no cover
        yaml = None  # type: ignore

    if yaml is not None:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    # Minimal fallback parser (handles the exact simple structure used in tests)
    # Not intended to be a general YAML parser.
    out: dict[str, Any] = {}
    rules: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    in_rules = False

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        if line.startswith("rubric_id:"):
            out["rubric_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("version:"):
            out["version"] = line.split(":", 1)[1].strip()
        elif line.startswith("rules:"):
            in_rules = True
        elif in_rules and line.lstrip().startswith("- "):
            # new rule entry
            if current:
                rules.append(current)
            current = {}
            # "- id: xxx"
            rest = line.lstrip()[2:].strip()
            if rest.startswith("id:"):
                current["id"] = rest.split(":", 1)[1].strip()
        elif in_rules and current is not None:
            s = line.strip()
            if s.startswith("enabled:"):
                v = s.split(":", 1)[1].strip().lower()
                current["enabled"] = v == "true"
            elif s.startswith("params:"):
                current["params"] = {}
            elif ":" in s and isinstance(current.get("params"), dict):
                k, v = s.split(":", 1)
                k = k.strip()
                v = v.strip()
                # best-effort typing
                if v.isdigit():
                    current["params"][k] = int(v)
                else:
                    current["params"][k] = v

    if current:
        rules.append(current)
    out["rules"] = rules
    return out


@dataclass(frozen=True)
class RuleEngine:
    """
    Lightweight deterministic rule runner.

    rules: accepts list of:
      - "rule_id"
      - {"id": "rule_id", "params": {...}}
    """

    rules: Sequence[object]

    def run(self, ctx: RuleContext) -> list[Any]:
        combined_flags: list[Any] = []

        # Ensure ctx.flags exists and is list[str]
        if not hasattr(ctx, "flags") or not isinstance(getattr(ctx, "flags"), list):
            try:
                setattr(ctx, "flags", [])
            except Exception:
                pass

        for spec in self.rules:
            rule_id = _as_rule_id(spec)
            if not rule_id:
                continue

            params = _as_rule_params(spec)

            try:
                rule, invoked_id = _resolve_rule(rule_id)
            except Exception:
                # deterministic judge must never crash due to unknown rules
                continue

            try:
                rr = rule.apply(ctx, params)
            except Exception:
                continue

            flags = getattr(rr, "flags", None)
            if not isinstance(flags, list):
                continue

            for f in flags:
                # 1) keep structured flags for engine-level tests
                combined_flags.append(f)

                # 2) also write normalized string flags for scorer
                fid = getattr(f, "id", None)
                sev = getattr(f, "severity", None)

                if isinstance(fid, str) and isinstance(sev, str):
                    # If invoked with "quality.xxx" but rule emits "xxx", prefix it
                    if "." in invoked_id and "." not in fid:
                        prefix = invoked_id.rsplit(".", 1)[0]
                        fid_full = f"{prefix}.{fid}"
                        try:
                            setattr(f, "id", fid_full)  # optional, best effort
                        except Exception:
                            pass
                        fid = fid_full

                    flag_str = f"{fid}:{sev}"
                else:
                    flag_str = str(f)

                try:
                    if isinstance(ctx.flags, list) and flag_str not in ctx.flags:
                        ctx.flags.append(flag_str)
                except Exception:
                    pass

        return combined_flags


def load_plan_for_rubric(rubric_id: str, version: str) -> RulePlan:
    """
    Load plan YAML from: configs/rules/{rubric_id}/{version}.yaml

    Validates the rule-plan YAML through ``RulePlanConfig`` (CP-1d
    Commit 2b). Schema failures raise :class:`RulePlanSchemaError`
    naming the file path.

    EPIC-3.2: Rules that are deprecated AND past their warning period
    are automatically excluded from the plan.
    """
    path = config_root() / "rules" / rubric_id / f"{version}.yaml"
    data = _load_yaml(path)

    try:
        parsed = RulePlanConfig.model_validate(data)
    except ValidationError as exc:
        raise RulePlanSchemaError(
            f"Rule plan failed schema validation at {path}: {exc}"
        ) from exc

    # Determine which rules are deprecated-enforced (past warning period)
    excluded: set[str] = set()
    try:
        from llm_judge.rules.lifecycle import get_deprecated_enforced_rules

        excluded = get_deprecated_enforced_rules()
    except Exception:
        pass  # graceful — if lifecycle unavailable, skip filtering

    rules: list[dict[str, Any]] = []
    for rule in parsed.rules:
        if not rule.enabled:
            continue
        # EPIC-3.2: Skip deprecated-enforced rules
        if rule.id in excluded:
            logger.info(
                "rule.excluded.deprecated",
                extra={"rule_id": rule.id, "rubric_id": rubric_id},
            )
            continue
        rules.append({"id": rule.id, "params": dict(rule.params)})

    return RulePlan(
        rubric_id=parsed.rubric_id or rubric_id,
        version=parsed.version or version,
        rules=rules,
    )


def run_rules(ctx: RuleContext, plan: RulePlan) -> Any:
    """
    Execute a RulePlan against context and return a RuleResult-like object.

    Test expectation:
      rr = run_rules(...); ids = {f.id for f in rr.flags}
    """
    engine = RuleEngine(rules=plan.rules)
    flags = engine.run(ctx)

    # Try to construct real RuleResult if available and compatible
    try:
        return RuleResult(flags=flags)
    except Exception:
        # Fallback: return a simple object with .flags attribute
        class _RR:  # noqa: D401
            def __init__(self, flags: list[Any]) -> None:
                self.flags = flags

        return _RR(flags)
