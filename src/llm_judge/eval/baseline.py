"""Baseline management for LLM Judge eval governance.

This module provides functionality to promote eval runs to baselines,
enabling regression detection in CI pipelines.

Usage (CLI):
    # Create baseline from a specific run
    python -m llm_judge.eval.baseline create --run-dir reports/runs/pr-gate-20260303-030618-670487

    # Create baseline from the latest run matching a prefix
    python -m llm_judge.eval.baseline create --latest pr-gate

    # List existing baselines
    python -m llm_judge.eval.baseline list

    # Show baseline details
    python -m llm_judge.eval.baseline show --suite golden --rubric chat_quality

    # Validate baseline integrity (latest pointer + required snapshot artifacts)
    python -m llm_judge.eval.baseline validate --suite golden --rubric-id chat_quality
    python -m llm_judge.eval.baseline validate --all
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml

from llm_judge.paths import baselines_root, state_root

logger = logging.getLogger(__name__)

# Default paths (can be overridden)
DEFAULT_BASELINES_DIR = baselines_root()
DEFAULT_RUNS_DIR = state_root() / "runs"


@dataclass(frozen=True)
class BaselineRef:
    """Reference to a baseline snapshot."""

    suite: str
    rubric_id: str
    baseline_id: str
    created_at_utc: str
    source_run_dir: str

    def snapshot_dir(self, baselines_dir: Path) -> Path:
        return baselines_dir / self.suite / self.rubric_id / "snapshots" / self.baseline_id


class BaselineError(Exception):
    """Base exception for baseline operations."""

    pass


class ValidationError(BaselineError):
    """Raised when run artifacts fail validation."""

    pass


class BaselineExistsError(BaselineError):
    """Raised when attempting to create a baseline that already exists."""

    pass


class BaselineIntegrityError(BaselineError):
    """Raised when an existing baseline (latest pointer/snapshot) is invalid."""

    pass


def _utc_now_compact() -> str:
    """Generate compact UTC timestamp: 20260224T154455Z"""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _require_file(path: Path, context: str = "") -> None:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if context:
            msg = f"{context}: {msg}"
        raise FileNotFoundError(msg)


def _count_jsonl_lines(path: Path) -> int:
    """Count non-empty lines in a JSONL file."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def infer_suite_from_manifest(manifest: dict[str, Any]) -> str:
    """Extract suite identifier from manifest, with fallback chain."""
    # Check runspec first (most reliable)
    runspec = manifest.get("runspec", {})
    if isinstance(runspec, dict):
        dataset = runspec.get("dataset", {})
        if isinstance(dataset, dict):
            for key in ("dataset_id", "suite"):
                val = dataset.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip()

    # Direct keys
    for key in ("suite", "dataset_id", "dataset", "dataset_name"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return "default"


def infer_rubric_id_from_manifest(manifest: dict[str, Any]) -> str:
    """Extract rubric ID from manifest, with fallback chain."""
    # Check runspec first
    runspec = manifest.get("runspec", {})
    if isinstance(runspec, dict):
        val = runspec.get("rubric_id")
        if isinstance(val, str) and val.strip():
            return val.strip()

    # Nested rubric object
    rubric = manifest.get("rubric")
    if isinstance(rubric, dict):
        rid = rubric.get("id") or rubric.get("rubric_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()

    # Direct keys
    for key in ("rubric_id", "rubric"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return "default"


def _resolve_case_artifact(run_dir: Path) -> tuple[Path, str]:
    """
    Find the per-case judgments artifact.

    Prefers: judgments.jsonl (current)
    Fallback: results.jsonl (legacy)

    Returns: (path, filename)
    """
    for name in ("judgments.jsonl", "results.jsonl"):
        path = run_dir / name
        if path.exists():
            return path, name

    raise FileNotFoundError(
        f"Missing per-case artifact in {run_dir}. "
        f"Expected judgments.jsonl or results.jsonl"
    )


def validate_run_artifacts(
    run_dir: Path,
    *,
    min_cases: int = 1,
    require_metrics: bool = True,
) -> dict[str, Any]:
    """
    Validate that a run directory contains valid artifacts for baseline creation.

    Returns manifest dict if valid, raises ValidationError otherwise.
    """
    manifest_path = run_dir / "manifest.json"
    metrics_path = run_dir / "metrics.json"

    _require_file(manifest_path, "Run validation")

    try:
        manifest = _read_json(manifest_path)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid manifest.json: {e}")

    if require_metrics:
        _require_file(metrics_path, "Run validation")
        try:
            metrics = _read_json(metrics_path)
            if not metrics:
                raise ValidationError("metrics.json is empty")
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid metrics.json: {e}")

    case_path, case_name = _resolve_case_artifact(run_dir)
    n_cases = _count_jsonl_lines(case_path)

    if n_cases < min_cases:
        raise ValidationError(
            f"Insufficient cases in {case_name}: found {n_cases}, require >= {min_cases}"
        )

    logger.info(f"Validated run: {run_dir} ({n_cases} cases)")
    return manifest


# ---------------------------------------------------------------------------
# Baseline integrity validation (P0 governance)
# ---------------------------------------------------------------------------

def _parse_latest_json(latest_path: Path) -> dict[str, Any]:
    _require_file(latest_path, "Baseline validation")
    try:
        return _read_json(latest_path)
    except json.JSONDecodeError as e:
        raise BaselineIntegrityError(f"Invalid latest.json at {latest_path}: {e}")


def _extract_baseline_id(latest: dict[str, Any]) -> str:
    """
    Supports existing format:
      { "baseline_id": "...", ... }

    Also tolerates future-proof aliases if ever introduced.
    """
    for key in ("baseline_id", "latest", "snapshot_id", "snapshot"):
        v = latest.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    raise BaselineIntegrityError("latest.json missing baseline identifier (baseline_id/latest/snapshot_id)")


def validate_baseline_snapshot_dir(snapshot_dir: Path) -> None:
    """
    World-class baseline snapshot contract.
    Required:
      - manifest.json
      - metrics.json
      - judgments.jsonl (canonical)
    Optional:
      - results.jsonl (legacy compat)
    """
    required = ["manifest.json", "metrics.json", "judgments.jsonl"]
    missing = [name for name in required if not (snapshot_dir / name).exists()]
    if missing:
        raise BaselineIntegrityError(f"Snapshot missing required files: {missing} in {snapshot_dir}")

    # Lightweight sanity checks: non-empty artifacts
    if _count_jsonl_lines(snapshot_dir / "judgments.jsonl") <= 0:
        raise BaselineIntegrityError(f"judgments.jsonl is empty in {snapshot_dir}")

    try:
        m = _read_json(snapshot_dir / "metrics.json")
        if not m:
            raise BaselineIntegrityError(f"metrics.json is empty in {snapshot_dir}")
    except json.JSONDecodeError as e:
        raise BaselineIntegrityError(f"Invalid metrics.json in {snapshot_dir}: {e}")

    try:
        _ = _read_json(snapshot_dir / "manifest.json")
    except json.JSONDecodeError as e:
        raise BaselineIntegrityError(f"Invalid manifest.json in {snapshot_dir}: {e}")


def validate_latest_baseline(
    *,
    suite: str,
    rubric_id: str,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
) -> Path:
    """
    Validate that baselines/<suite>/<rubric_id>/latest.json points to a valid snapshot.
    Returns the snapshot_dir if valid.
    """
    baselines_dir = baselines_dir.resolve()
    latest_path = baselines_dir / suite / rubric_id / "latest.json"
    latest = _parse_latest_json(latest_path)

    baseline_id = _extract_baseline_id(latest)
    snapshot_dir = baselines_dir / suite / rubric_id / "snapshots" / baseline_id

    if not snapshot_dir.exists() or not snapshot_dir.is_dir():
        raise BaselineIntegrityError(f"latest.json points to missing snapshot dir: {snapshot_dir}")

    validate_baseline_snapshot_dir(snapshot_dir)
    return snapshot_dir


def _iter_latest_files(baselines_dir: Path) -> Iterable[Path]:
    return baselines_dir.resolve().rglob("latest.json")

def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}

# ---------------------------------------------------------------------------
# Baseline creation / listing / show (existing behavior)
# ---------------------------------------------------------------------------

def create_baseline_from_run(
    *,
    run_dir: Path,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    suite: str | None = None,
    rubric_id: str | None = None,
    baseline_id: str | None = None,
    set_latest: bool = True,
    overwrite: bool = False,
    min_cases: int = 1,
) -> BaselineRef:
    """
    Promote a run directory into a baseline snapshot.
    """
    run_dir = run_dir.resolve()
    baselines_dir = baselines_dir.resolve()

    # Validate before creating anything
    manifest = validate_run_artifacts(run_dir, min_cases=min_cases)

    case_path, case_name = _resolve_case_artifact(run_dir)

    suite_final = (suite or infer_suite_from_manifest(manifest)).strip()
    rubric_final = (rubric_id or infer_rubric_id_from_manifest(manifest)).strip()

    # Single timestamp for consistency
    ts = _utc_now_compact()
    baseline_final = (baseline_id or ts).strip()

    dst_dir = baselines_dir / suite_final / rubric_final / "snapshots" / baseline_final

    if dst_dir.exists():
        if overwrite:
            logger.warning(f"Overwriting existing baseline: {dst_dir}")
            shutil.rmtree(dst_dir)
        else:
            raise BaselineExistsError(
                f"Baseline already exists: {dst_dir}. Use --overwrite to replace."
            )

    dst_dir.mkdir(parents=True, exist_ok=False)

    # Copy all artifacts (canonical snapshot format)
    shutil.copy2(run_dir / "manifest.json", dst_dir / "manifest.json")
    shutil.copy2(run_dir / "metrics.json", dst_dir / "metrics.json")
    shutil.copy2(case_path, dst_dir / "judgments.jsonl")

    # Compat file for older tooling expecting results.jsonl
    results_compat = dst_dir / "results.jsonl"
    if not results_compat.exists():
        shutil.copy2(case_path, results_compat)

    ref = BaselineRef(
        suite=suite_final,
        rubric_id=rubric_final,
        baseline_id=baseline_final,
        created_at_utc=ts,
        source_run_dir=str(run_dir),
    )

    if set_latest:
        latest_path = baselines_dir / suite_final / rubric_final / "latest.json"
        _write_json(
            latest_path,
            {
                "suite": ref.suite,
                "rubric_id": ref.rubric_id,
                "baseline_id": ref.baseline_id,
                "created_at_utc": ref.created_at_utc,
                "source_run_dir": ref.source_run_dir,
                "case_artifact_source": case_name,
                "case_artifact_snapshot": "judgments.jsonl",
            },
        )
        logger.info(f"Updated latest pointer: {latest_path}")

    logger.info(f"Created baseline: {dst_dir}")
    return ref


def find_latest_run(prefix: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Path:
    """Find the most recent run directory matching a prefix."""
    runs_dir = runs_dir.resolve()
    matches = sorted(runs_dir.glob(f"{prefix}*"), reverse=True)

    if not matches:
        raise FileNotFoundError(f"No runs found matching prefix '{prefix}' in {runs_dir}")

    # Return most recent (sorted descending by name, which includes timestamp)
    return matches[0]


def list_baselines(baselines_dir: Path = DEFAULT_BASELINES_DIR) -> list[dict[str, Any]]:
    """List all baselines with their metadata."""
    baselines_dir = baselines_dir.resolve()
    results: list[dict[str, Any]] = []

    for latest_file in baselines_dir.rglob("latest.json"):
        try:
            data = _read_json(latest_file)
            data["_path"] = str(latest_file.parent)
            results.append(data)
        except Exception as e:
            logger.warning(f"Failed to read {latest_file}: {e}")

    return results


def get_baseline_info(
    suite: str,
    rubric_id: str,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
) -> dict[str, Any]:
    """Get detailed info about a specific baseline."""
    latest_path = baselines_dir / suite / rubric_id / "latest.json"
    _require_file(latest_path, f"Baseline {suite}/{rubric_id}")

    info = _read_json(latest_path)
    snapshot_dir = baselines_dir / suite / rubric_id / "snapshots" / info["baseline_id"]

    # Add metrics summary
    metrics_path = snapshot_dir / "metrics.json"
    if metrics_path.exists():
        info["metrics"] = _read_json(metrics_path)

    # Add case count
    judgments_path = snapshot_dir / "judgments.jsonl"
    if judgments_path.exists():
        info["n_cases"] = _count_jsonl_lines(judgments_path)

    return info


# =============================================================================
# CLI Interface
# =============================================================================

def _cmd_create(args: argparse.Namespace) -> int:
    """Handle 'create' subcommand."""
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.latest:
        run_dir = find_latest_run(args.latest, Path(args.runs_dir))
        print(f"Found latest run: {run_dir}")
    else:
        print("Error: Must specify --run-dir or --latest", file=sys.stderr)
        return 1

    try:
        ref = create_baseline_from_run(
            run_dir=run_dir,
            baselines_dir=Path(args.baselines_dir),
            suite=args.suite,
            rubric_id=args.rubric_id,
            baseline_id=args.baseline_id,
            set_latest=not args.no_latest,
            overwrite=args.overwrite,
            min_cases=args.min_cases,
        )
        print(f"Created baseline: {ref.suite}/{ref.rubric_id}/{ref.baseline_id}")
        print(f"  Source: {ref.source_run_dir}")
        print(f"  Created: {ref.created_at_utc}")
        return 0
    except BaselineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle 'list' subcommand."""
    baselines = list_baselines(Path(args.baselines_dir))

    if not baselines:
        print("No baselines found.")
        return 0

    if args.json:
        print(json.dumps(baselines, indent=2))
    else:
        print(f"{'Suite':<20} {'Rubric':<20} {'Baseline ID':<25} {'Created'}")
        print("-" * 85)
        for b in baselines:
            print(
                f"{b.get('suite', 'N/A'):<20} "
                f"{b.get('rubric_id', 'N/A'):<20} "
                f"{b.get('baseline_id', 'N/A'):<25} "
                f"{b.get('created_at_utc', 'N/A')}"
            )
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    """Handle 'show' subcommand."""
    try:
        info = get_baseline_info(
            args.suite,
            args.rubric_id,
            Path(args.baselines_dir),
        )
        print(json.dumps(info, indent=2))
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    """Handle 'validate' subcommand."""
    baselines_dir = Path(args.baselines_dir)

    try:
        if args.all:
            failures: list[str] = []
            for latest_path in _iter_latest_files(baselines_dir):
                # latest_path: baselines/<suite>/<rubric_id>/latest.json
                try:
                    rubric_dir = latest_path.parent
                    rubric_id = rubric_dir.name
                    suite = rubric_dir.parent.name
                    snap = validate_latest_baseline(
                        suite=suite,
                        rubric_id=rubric_id,
                        baselines_dir=baselines_dir,
                    )
                    print(f"OK  {suite}/{rubric_id} -> {snap}")
                except Exception as e:
                    failures.append(f"{latest_path}: {e}")
                    print(f"ERR {latest_path}: {e}")

            if failures:
                return 1
            return 0

        if not args.suite or not args.rubric_id:
            print("Error: Must pass --suite and --rubric-id (or use --all)", file=sys.stderr)
            return 1

        snap = validate_latest_baseline(
            suite=args.suite,
            rubric_id=args.rubric_id,
            baselines_dir=baselines_dir,
        )
        print(f"OK {args.suite}/{args.rubric_id} -> {snap}")
        return 0

    except BaselineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _cmd_promote(args: argparse.Namespace) -> int:
    """Handle 'promote' subcommand."""
    try:
        ref = promote_baseline_from_run(
            run_dir=Path(args.run_dir),
            policy_path=Path(args.policy),
            baselines_dir=Path(args.baselines_dir),
            suite=args.suite,
            rubric_id=args.rubric_id,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            print(f"Dry run complete: {ref.suite}/{ref.rubric_id}")
        else:
            print(f"Baseline promoted: baselines/{ref.suite}/{ref.rubric_id}/snapshots/{ref.baseline_id}")
            print(f"Latest updated:    baselines/{ref.suite}/{ref.rubric_id}/latest.json")
        return 0
    except BaselineError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m llm_judge.eval.baseline",
        description="Manage eval baselines for regression detection",
    )
    parser.add_argument(
        "--baselines-dir",
        default="baselines",
        help="Root directory for baselines (default: baselines/)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create subcommand
    create_parser = subparsers.add_parser("create", help="Create baseline from a run")
    create_parser.add_argument("--run-dir", help="Path to run directory")
    create_parser.add_argument("--latest", help="Use latest run matching this prefix")
    create_parser.add_argument("--runs-dir", default="reports/runs", help="Runs directory")
    create_parser.add_argument("--suite", help="Override suite name")
    create_parser.add_argument("--rubric-id", help="Override rubric ID")
    create_parser.add_argument("--baseline-id", help="Override baseline ID")
    create_parser.add_argument("--no-latest", action="store_true", help="Don't update latest.json")
    create_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing baseline")
    create_parser.add_argument("--min-cases", type=int, default=1, help="Minimum cases required")
    create_parser.set_defaults(func=_cmd_create)

    # list subcommand
    list_parser = subparsers.add_parser("list", help="List all baselines")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=_cmd_list)

    # show subcommand
    show_parser = subparsers.add_parser("show", help="Show baseline details")
    show_parser.add_argument("--suite", required=True, help="Suite name")
    show_parser.add_argument("--rubric-id", required=True, help="Rubric ID")
    show_parser.set_defaults(func=_cmd_show)

    # validate subcommand
    validate_parser = subparsers.add_parser("validate", help="Validate baseline integrity")
    validate_parser.add_argument("--suite", help="Suite name (required unless --all)")
    validate_parser.add_argument("--rubric-id", help="Rubric ID (required unless --all)")
    validate_parser.add_argument("--all", action="store_true", help="Validate all latest.json pointers found")
    validate_parser.set_defaults(func=_cmd_validate)

    # promote subcommand
    promote_parser = subparsers.add_parser("promote", help="Policy-gated baseline promotion")
    promote_parser.add_argument("--run-dir", required=True, help="Eval run output dir")
    promote_parser.add_argument("--policy", required=True, help="Promotion policy YAML")
    promote_parser.add_argument("--baselines-dir", default="baselines", help="Baselines root (default: baselines)")
    promote_parser.add_argument("--suite", help="Suite override (optional)")
    promote_parser.add_argument("--rubric-id", help="Rubric override (optional)")
    promote_parser.add_argument("--dry-run", action="store_true", help="Evaluate policy but do not write snapshot")
    promote_parser.set_defaults(func=_cmd_promote)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    return args.func(args)


def _policy_check(*, policy: dict[str, Any], diff_summary: dict[str, Any]) -> tuple[bool, list[str]]:
    violations: list[str] = []

    required = policy.get("required_metrics") or []
    if not isinstance(required, list):
        required = []

    metric_diffs = diff_summary.get("metrics", {})
    deltas = metric_diffs.get("deltas", {}) if isinstance(metric_diffs, dict) else {}

    # Required metrics must appear in delta set OR at least be present in both metrics files.
    # P0: enforce presence in deltas if numeric.
    for k in required:
        if str(k) not in deltas:
            violations.append(f"Missing required metric delta: {k}")

    # Max drop checks: baseline - candidate <= tolerance  => (candidate - baseline) >= -tolerance
    max_drop = policy.get("max_metric_drop") or {}
    if isinstance(max_drop, dict):
        for k, tol_any in max_drop.items():
            try:
                tol = float(tol_any)
            except Exception:
                continue
            d = deltas.get(str(k))
            if isinstance(d, dict):
                delta = d.get("delta")
                if isinstance(delta, (int, float)) and delta < -tol:
                    violations.append(
                        f"Metric drop too large: {k} baseline={d.get('baseline')} candidate={d.get('candidate')} "
                        f"drop={-delta} (tolerance={tol})"
                    )

    # Flip cap
    max_flips_any = policy.get("max_decision_flips")
    if max_flips_any is not None:
        try:
            max_flips = int(max_flips_any)
            flips = diff_summary.get("judgments", {}).get("decision_flips", [])
            if isinstance(flips, list) and len(flips) > max_flips:
                violations.append(f"Too many decision flips: {len(flips)} (max={max_flips})")
        except Exception:
            pass

    return (len(violations) == 0), violations
    
def promote_baseline_from_run(
    *,
    run_dir: Path,
    policy_path: Path,
    baselines_dir: Path = DEFAULT_BASELINES_DIR,
    suite: str | None = None,
    rubric_id: str | None = None,
    dry_run: bool = False,
) -> BaselineRef:
    from llm_judge.eval.diff import compute_diff_summary, resolve_baseline, resolve_run_dir

    manifest = validate_run_artifacts(run_dir)

    suite_eff = suite or infer_suite_from_manifest(manifest)
    rubric_eff = rubric_id or infer_rubric_id_from_manifest(manifest)

    # Load promotion policy
    policy = _read_yaml(policy_path)

    # Resolve baseline pointer if exists
    latest_path = baselines_dir / suite_eff / rubric_eff / "latest.json"
    allow_missing = bool(policy.get("allow_missing_baseline", True))

    diff_summary: dict[str, Any] | None = None
    if latest_path.exists():
        baseline_run = resolve_baseline(latest_path)
        candidate_run = resolve_run_dir(run_dir)
        diff_summary = compute_diff_summary(baseline=baseline_run, candidate=candidate_run)

        ok, violations = _policy_check(policy=policy, diff_summary=diff_summary)
        if not ok:
            raise BaselineError("POLICY: VIOLATION\n- " + "\n- ".join(violations))
    else:
        if not allow_missing:
            raise BaselineError(f"Missing latest baseline pointer: {latest_path}")

    if dry_run:
        print("Promotion decision: PASS (dry-run)" if diff_summary is not None else "Promotion decision: PASS (no prior baseline)")
        return BaselineRef(
            suite=suite_eff,
            rubric_id=rubric_eff,
            baseline_id="DRY_RUN",
            created_at_utc=_utc_now_compact(),
            source_run_dir=str(run_dir),
        )

    # Create snapshot (reuse existing baseline creator)
    ref = create_baseline_from_run(
        run_dir=run_dir,
        baselines_dir=baselines_dir,
        suite=suite_eff,
        rubric_id=rubric_eff,
        baseline_id=None,
        set_latest=True,
    )

    # Write diff summary into snapshot if we had a baseline to compare
    if diff_summary is not None:
        _write_json(ref.snapshot_dir(baselines_dir) / "diff_summary.json", diff_summary)

    # --- EPIC-5.1: Emit baseline_promotion event to cross-capability registry ---
    try:
        from llm_judge.eval.event_registry import append_event

        append_event(
            event_type="baseline_promotion",
            source="eval/baseline.py",
            actor="operator",
            related_ids={
                "baseline_id": ref.baseline_id,
                "suite": ref.suite,
                "rubric_id": ref.rubric_id,
                "source_run_dir": ref.source_run_dir,
            },
            payload={
                "had_prior_baseline": diff_summary is not None,
                "violations": [],
                "dry_run": False,
            },
        )
    except Exception:
        logger.warning("Could not emit baseline_promotion event to event registry")

    return ref
    

if __name__ == "__main__":
    sys.exit(main())