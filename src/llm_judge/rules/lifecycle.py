from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path("rules/manifest.yaml")
VALID_STATUSES = {"draft", "validated", "production", "deprecated"}


@dataclass(frozen=True)
class RuleMeta:
    name: str
    version: int
    owner: str
    status: str
    introduced: str


def _read_yaml(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid manifest YAML: {path}")
    return raw


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, RuleMeta]:
    if not path.exists():
        raise FileNotFoundError(f"Missing rule manifest: {path}")

    raw = _read_yaml(path)

    schema_version = str(raw.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError(f"rules manifest missing schema_version: {path}")

    rules = raw.get("rules", {})
    if not isinstance(rules, dict):
        raise ValueError(f"rules manifest must contain a 'rules' mapping: {path}")

    out: dict[str, RuleMeta] = {}
    for name, meta in rules.items():
        if not isinstance(meta, dict):
            raise ValueError(f"Rule '{name}' metadata must be a mapping")

        out[str(name)] = RuleMeta(
            name=str(name),
            version=int(meta["version"]),
            owner=str(meta["owner"]),
            status=str(meta["status"]),
            introduced=str(meta["introduced"]),
        )

    return out


def discover_runtime_rules() -> set[str]:
    """
    Discover rule ids from the runtime rule registry.

    This is the key governance protection:
    if a rule exists in code/registry but not in rules/manifest.yaml,
    validation must fail.
    """
    try:
        from llm_judge.rules.registry import RULE_REGISTRY
    except Exception as e:
        raise RuntimeError(
            "Could not import llm_judge.rules.registry.RULE_REGISTRY for rule discovery"
        ) from e

    if not isinstance(RULE_REGISTRY, dict):
        raise ValueError("RULE_REGISTRY must be a dict")

    discovered: set[str] = set()
    for key in RULE_REGISTRY.keys():
        discovered.add(str(key).strip())

    return {r for r in discovered if r}


def list_rules() -> None:
    manifest_rules = load_manifest()
    runtime_rules = discover_runtime_rules()

    print("")
    print("Registered Rules")
    print("---------------------------------------------------------------------")
    print(f"{'RULE':28} {'VER':<5} {'STATUS':<12} {'OWNER':<16} {'IN_RUNTIME':<10}")
    print("---------------------------------------------------------------------")

    for name in sorted(manifest_rules.keys()):
        r = manifest_rules[name]
        in_runtime = "yes" if name in runtime_rules else "no"
        print(
            f"{r.name:28} "
            f"{r.version:<5} "
            f"{r.status:<12} "
            f"{r.owner:<16} "
            f"{in_runtime:<10}"
        )

    missing_in_manifest = sorted(runtime_rules - set(manifest_rules.keys()))
    if missing_in_manifest:
        print("")
        print("Runtime rules missing in manifest:")
        for rule_name in missing_in_manifest:
            print(f"- {rule_name}")


def validate_rules() -> None:
    manifest_rules = load_manifest()
    runtime_rules = discover_runtime_rules()

    errors: list[str] = []

    # Metadata validation
    for r in manifest_rules.values():
        if r.status not in VALID_STATUSES:
            errors.append(f"{r.name}: invalid status '{r.status}'")

        if r.version <= 0:
            errors.append(f"{r.name}: version must be > 0")

        if not r.owner.strip():
            errors.append(f"{r.name}: owner must be non-empty")

        if not r.introduced.strip():
            errors.append(f"{r.name}: introduced must be non-empty")

    manifest_rule_names = set(manifest_rules.keys())

    # Runtime rule exists but not governed
    missing_in_manifest = sorted(runtime_rules - manifest_rule_names)
    for name in missing_in_manifest:
        errors.append(f"{name}: present in RULE_REGISTRY but missing in rules/manifest.yaml")

    # Manifest rule does not exist in code/registry
    missing_in_runtime = sorted(manifest_rule_names - runtime_rules)
    for name in missing_in_runtime:
        errors.append(f"{name}: present in rules/manifest.yaml but missing in RULE_REGISTRY")

    if errors:
        print("Rule lifecycle validation FAILED")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("Rule lifecycle validation OK")


def check_rules_governed() -> list[str]:
    """
    EPIC-3.3: Embeddable rule governance check.

    Returns a list of error strings (empty = all rules governed).
    Unlike validate_rules(), this does NOT print or exit — it returns
    errors for the caller to handle. Used by eval runner pre-flight
    to enforce governance at runtime, not just during preflight.
    """
    errors: list[str] = []

    try:
        manifest_rules = load_manifest()
    except (FileNotFoundError, ValueError) as e:
        return [f"Rule manifest not found or invalid: {e}"]

    try:
        runtime_rules = discover_runtime_rules()
    except RuntimeError as e:
        return [f"Could not discover runtime rules: {e}"]

    manifest_rule_names = set(manifest_rules.keys())

    # Rules in code but not governed
    for name in sorted(runtime_rules - manifest_rule_names):
        errors.append(f"Ungoverned rule: '{name}' exists in code but not in rules/manifest.yaml")

    # Rules declared but missing from code
    for name in sorted(manifest_rule_names - runtime_rules):
        errors.append(f"Stale manifest entry: '{name}' in rules/manifest.yaml but not in RULE_REGISTRY")

    # Metadata quality
    for r in manifest_rules.values():
        if r.status not in VALID_STATUSES:
            errors.append(f"Invalid status for '{r.name}': '{r.status}'")

    return errors


def emit_rule_snapshot(*, actor: str = "system", registry_path: Path | None = None) -> None:
    """
    EPIC-5.1: Record current rule state as a governance event.

    Called during eval runner pre-flight to create a traceable record
    of which rules were active for each evaluation. Drift detection
    (EPIC 6.1) uses these events to correlate metric changes with
    rule state changes.
    """
    try:
        manifest_rules = load_manifest()
        runtime_rules = discover_runtime_rules()
    except Exception:
        return  # best-effort — don't block eval runs

    rule_snapshot = {
        name: {
            "version": meta.version,
            "status": meta.status,
            "owner": meta.owner,
            "in_runtime": name in runtime_rules,
        }
        for name, meta in manifest_rules.items()
    }
    ungoverned = sorted(runtime_rules - set(manifest_rules.keys()))

    try:
        from llm_judge.eval.event_registry import append_event

        kwargs: dict[str, Any] = {
            "event_type": "rule_change",
            "source": "rules/lifecycle.py",
            "actor": actor,
            "related_ids": {},
            "payload": {
                "action": "snapshot",
                "rule_count": len(manifest_rules),
                "rules": rule_snapshot,
                "ungoverned_rules": ungoverned,
            },
        }
        if registry_path is not None:
            kwargs["registry_path"] = registry_path
        append_event(**kwargs)
    except Exception:
        pass  # best-effort


def export_json(out_path: Path) -> None:
    manifest_rules = load_manifest()
    runtime_rules = discover_runtime_rules()

    data = {
        "schema_version": "1",
        "rules": {
            r.name: {
                "version": r.version,
                "owner": r.owner,
                "status": r.status,
                "introduced": r.introduced,
                "in_runtime_registry": r.name in runtime_rules,
            }
            for r in manifest_rules.values()
        },
        "runtime_only_rules": sorted(list(runtime_rules - set(manifest_rules.keys()))),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule lifecycle management for LLM-Judge.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List rules from manifest with runtime presence")
    sub.add_parser("validate", help="Validate manifest + runtime rule alignment")

    p = sub.add_parser("export-json", help="Export lifecycle metadata as JSON")
    p.add_argument("--out", required=True)

    args = parser.parse_args()

    if args.cmd == "list":
        list_rules()
        return 0

    if args.cmd == "validate":
        validate_rules()
        return 0

    if args.cmd == "export-json":
        export_json(Path(args.out))
        return 0

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())