from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from .baseline import create_baseline_from_run, promote_baseline_from_run


def _cmd_baseline_create(args: argparse.Namespace) -> int:
    ref = create_baseline_from_run(
        run_dir=Path(args.run_dir),
        baselines_dir=Path(args.baselines_dir),
        suite=args.suite,
        rubric_id=args.rubric_id,
        baseline_id=args.baseline_id,
        set_latest=not args.no_latest,
    )
    print(
        f"Baseline created: baselines/{ref.suite}/{ref.rubric_id}/snapshots/{ref.baseline_id}"
    )
    if not args.no_latest:
        print(f"Latest updated:  baselines/{ref.suite}/{ref.rubric_id}/latest.json")
    return 0


def _cmd_baseline_promote(args: argparse.Namespace) -> int:
    try:
        ref = promote_baseline_from_run(
            run_dir=Path(args.run_dir),
            policy_path=Path(args.policy),
            baselines_dir=Path(args.baselines_dir),
            suite=args.suite,
            rubric_id=args.rubric_id,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(str(e))
        return 2
    print(
        f"Baseline promoted: baselines/{ref.suite}/{ref.rubric_id}/snapshots/{ref.baseline_id}"
    )
    print(f"Latest updated:    baselines/{ref.suite}/{ref.rubric_id}/latest.json")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """
    Delegate to the eval runner CLI implemented in llm_judge.eval.run.
    """
    cmd = [sys.executable, "-m", "llm_judge.eval.run", *args.run_args]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def _cmd_diff(args: argparse.Namespace) -> int:
    """
    Delegate to the eval diff CLI implemented in llm_judge.eval.diff.
    """
    cmd = [sys.executable, "-m", "llm_judge.eval.diff", *args.diff_args]
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llm-judge-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- run (delegates to llm_judge.eval.run) ----
    p_run = sub.add_parser(
        "run",
        help="Run evaluation (delegates to python -m llm_judge.eval.run ...)",
        prefix_chars="-",
    )
    p_run.add_argument(
        "run_args",
        nargs="*",
        help="Arguments passed through to llm_judge.eval.run (e.g., --spec configs/runspecs/pr_gate.yaml)",
    )
    p_run.set_defaults(func=_cmd_run)

    # ---- diff (delegates to llm_judge.eval.diff) ----
    p_diff = sub.add_parser(
        "diff",
        help="Diff evaluation runs (delegates to python -m llm_judge.eval.diff ...)",
        prefix_chars="-",
    )
    p_diff.add_argument(
        "diff_args",
        nargs="*",
        help="Arguments passed through to llm_judge.eval.diff (e.g., --baseline ... --candidate ...)",
    )
    p_diff.set_defaults(func=_cmd_diff)

    # ---- baseline-create ----
    p_bc = sub.add_parser(
        "baseline-create",
        help="Promote an eval run dir into a baseline snapshot",
    )
    p_bc.add_argument(
        "--run-dir",
        required=True,
        help="Path to eval run output directory (contains manifest.json/judgments.jsonl/metrics.json)",
    )
    p_bc.add_argument(
        "--baselines-dir",
        default="baselines",
        help="Baselines root directory (default: baselines)",
    )
    p_bc.add_argument(
        "--suite",
        default=None,
        help="Suite name override (optional; inferred from manifest if omitted)",
    )
    p_bc.add_argument(
        "--rubric-id",
        default=None,
        help="Rubric id override (optional; inferred from manifest if omitted)",
    )
    p_bc.add_argument(
        "--baseline-id",
        default=None,
        help="Baseline id override (default: UTC timestamp)",
    )
    p_bc.add_argument(
        "--no-latest",
        action="store_true",
        help="Do not update latest.json",
    )
    p_bc.set_defaults(func=_cmd_baseline_create)

    # ---- baseline-promote ----
    p_bp = sub.add_parser(
        "baseline-promote",
        help="Policy-gated baseline promotion",
    )
    p_bp.add_argument(
        "--run-dir",
        required=True,
        help="Eval run output dir",
    )
    p_bp.add_argument(
        "--policy",
        required=True,
        help="Promotion policy YAML",
    )
    p_bp.add_argument(
        "--baselines-dir",
        default="baselines",
        help="Baselines root (default: baselines)",
    )
    p_bp.add_argument(
        "--suite",
        default=None,
        help="Suite override (optional)",
    )
    p_bp.add_argument(
        "--rubric-id",
        default=None,
        help="Rubric override (optional)",
    )
    p_bp.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate policy but do not write snapshot",
    )
    p_bp.set_defaults(func=_cmd_baseline_promote)

    return p


def main() -> int:
    # Fast path passthrough: avoid argparse touching flags for delegated CLIs.
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        passthrough = sys.argv[2:]
        args = argparse.Namespace(func=_cmd_run, run_args=passthrough)
        return int(args.func(args))

    if len(sys.argv) >= 2 and sys.argv[1] == "diff":
        passthrough = sys.argv[2:]
        args = argparse.Namespace(func=_cmd_diff, diff_args=passthrough)
        return int(args.func(args))

    # All other subcommands parse normally.
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
