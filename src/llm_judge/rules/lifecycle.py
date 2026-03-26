from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml

MANIFEST_PATH = Path("rules/manifest.yaml")
AUDIT_LOG_PATH = Path("reports/rule_audit.jsonl")
VALID_STATUSES = {"draft", "validated", "production", "deprecated"}

# Default review period if not specified per-rule
DEFAULT_REVIEW_PERIOD_DAYS = 365


@dataclass(frozen=True)
class RuleMeta:
    name: str
    version: int
    owner: str
    status: str
    introduced: str

    # Aging fields (optional — backward compatible)
    review_period_days: int = DEFAULT_REVIEW_PERIOD_DAYS
    last_reviewed: str | None = None       # ISO date: "2026-03-06"
    deprecated_at: str | None = None       # ISO date when deprecated
    deprecation_warning_days: int = 30     # grace period before enforcement


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
            review_period_days=int(meta.get("review_period_days", DEFAULT_REVIEW_PERIOD_DAYS)),
            last_reviewed=str(meta["last_reviewed"]) if meta.get("last_reviewed") else None,
            deprecated_at=str(meta["deprecated_at"]) if meta.get("deprecated_at") else None,
            deprecation_warning_days=int(meta.get("deprecation_warning_days", 30)),
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


# =====================================================================
# Aging & Deprecation (EPIC 3.2)
# =====================================================================

def _parse_date(s: str) -> date | None:
    """Parse ISO date string (YYYY-MM-DD). Returns None on failure."""
    try:
        return date.fromisoformat(s.strip())
    except (ValueError, AttributeError):
        return None


def _today() -> date:
    """Current UTC date. Separate function for testability."""
    return datetime.now(timezone.utc).date()


@dataclass(frozen=True)
class AgingReport:
    """Aging status for a single rule."""
    rule_name: str
    status: str
    introduced: str
    last_reviewed: str | None
    review_period_days: int
    days_since_review: int | None
    stale: bool
    deprecated: bool
    deprecation_enforced: bool  # past warning period


def compute_aging(rule: RuleMeta) -> AgingReport:
    """Compute aging status for a single rule."""
    today = _today()

    # Determine review anchor: last_reviewed if set, otherwise introduced
    review_anchor = _parse_date(rule.last_reviewed) if rule.last_reviewed else _parse_date(rule.introduced)
    days_since = (today - review_anchor).days if review_anchor else None

    stale = False
    if days_since is not None and days_since > rule.review_period_days:
        stale = True

    # Deprecation enforcement: past the warning period?
    deprecated = rule.status == "deprecated"
    enforced = False
    if deprecated and rule.deprecated_at:
        dep_date = _parse_date(rule.deprecated_at)
        if dep_date:
            days_since_deprecated = (today - dep_date).days
            enforced = days_since_deprecated > rule.deprecation_warning_days

    return AgingReport(
        rule_name=rule.name,
        status=rule.status,
        introduced=rule.introduced,
        last_reviewed=rule.last_reviewed,
        review_period_days=rule.review_period_days,
        days_since_review=days_since,
        stale=stale,
        deprecated=deprecated,
        deprecation_enforced=enforced,
    )


def check_aging(path: Path = MANIFEST_PATH) -> list[AgingReport]:
    """Compute aging reports for all rules. Returns list sorted by staleness."""
    manifest = load_manifest(path)
    reports = [compute_aging(r) for r in manifest.values()]
    # Sort: stale first, then by days_since_review descending
    reports.sort(key=lambda r: (not r.stale, -(r.days_since_review or 0)))
    return reports


def get_deprecated_enforced_rules(path: Path = MANIFEST_PATH) -> set[str]:
    """
    Return rule names that are deprecated AND past their warning period.

    Used by RuleEngine to exclude enforced-deprecated rules from execution.
    """
    reports = check_aging(path)
    return {r.rule_name for r in reports if r.deprecation_enforced}


# =====================================================================
# Audit Trail (EPIC 3.1)
# =====================================================================

def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def append_audit_entry(
    *,
    rule_id: str,
    action: str,
    actor: str = "system",
    details: dict[str, Any] | None = None,
    audit_path: Path = AUDIT_LOG_PATH,
) -> None:
    """
    Append a structured audit entry to the rule audit log.

    Actions: created, updated, deprecated, reviewed, status_changed, deleted
    """
    entry = {
        "timestamp": _utc_now_iso(),
        "rule_id": rule_id,
        "action": action,
        "actor": actor,
        "details": details or {},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_audit_log(
    *,
    rule_id: str | None = None,
    audit_path: Path = AUDIT_LOG_PATH,
) -> list[dict[str, Any]]:
    """Read audit log, optionally filtered by rule_id."""
    if not audit_path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with audit_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if rule_id and obj.get("rule_id") != rule_id:
                continue
            entries.append(obj)
    return entries


def list_rules() -> None:
    manifest_rules = load_manifest()
    runtime_rules = discover_runtime_rules()

    print("")
    print("Registered Rules")
    print("-" * 95)
    print(
        f"{'RULE':28} {'VER':<5} {'STATUS':<12} {'OWNER':<16} "
        f"{'IN_RT':<6} {'DAYS':<6} {'STALE':<6}"
    )
    print("-" * 95)

    for name in sorted(manifest_rules.keys()):
        r = manifest_rules[name]
        in_runtime = "yes" if name in runtime_rules else "no"
        aging = compute_aging(r)
        days_str = str(aging.days_since_review) if aging.days_since_review is not None else "?"
        stale_str = "YES" if aging.stale else ("-" if aging.deprecated else "no")
        print(
            f"{r.name:28} "
            f"{r.version:<5} "
            f"{r.status:<12} "
            f"{r.owner:<16} "
            f"{in_runtime:<6} "
            f"{days_str:<6} "
            f"{stale_str:<6}"
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

    sub.add_parser("list", help="List rules from manifest with runtime presence and aging")
    sub.add_parser("validate", help="Validate manifest + runtime rule alignment")
    sub.add_parser("aging", help="Show rule aging report (stale rules, deprecation status)")

    p = sub.add_parser("export-json", help="Export lifecycle metadata as JSON")
    p.add_argument("--out", required=True)

    p_audit = sub.add_parser("audit", help="Show rule audit log")
    p_audit.add_argument("--rule-id", default=None, help="Filter by rule ID")

    args = parser.parse_args()

    if args.cmd == "list":
        list_rules()
        return 0

    if args.cmd == "validate":
        validate_rules()
        return 0

    if args.cmd == "aging":
        reports = check_aging()
        print("")
        print("Rule Aging Report")
        print("-" * 90)
        print(
            f"{'RULE':28} {'STATUS':<12} {'DAYS':<8} "
            f"{'REVIEW_PERIOD':<14} {'STALE':<7} {'DEPRECATED':<12}"
        )
        print("-" * 90)
        stale_count = 0
        for r in reports:
            days_str = str(r.days_since_review) if r.days_since_review is not None else "?"
            stale_str = "YES" if r.stale else "no"
            dep_str = "ENFORCED" if r.deprecation_enforced else ("warning" if r.deprecated else "-")
            if r.stale:
                stale_count += 1
            print(
                f"{r.rule_name:28} {r.status:<12} {days_str:<8} "
                f"{r.review_period_days:<14} {stale_str:<7} {dep_str:<12}"
            )
        print("")
        if stale_count:
            print(f"  WARNING: {stale_count} rule(s) exceed review period.")
            print("  Action: review and update last_reviewed in rules/manifest.yaml")
        else:
            print("  All rules within review period.")
        return 0

    if args.cmd == "audit":
        entries = read_audit_log(rule_id=args.rule_id)
        if not entries:
            print("No audit entries found.")
            return 0
        print("")
        print(f"Rule Audit Log ({len(entries)} entries)")
        print("-" * 80)
        for e in entries:
            print(
                f"  {e.get('timestamp', '?'):<22} "
                f"{e.get('rule_id', '?'):<28} "
                f"{e.get('action', '?'):<16} "
                f"{e.get('actor', '?')}"
            )
        return 0

    if args.cmd == "export-json":
        export_json(Path(args.out))
        return 0

    parser.print_help()
    return 1

if __name__ == "__main__":
    raise SystemExit(main())