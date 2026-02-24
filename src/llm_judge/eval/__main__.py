from __future__ import annotations

import argparse
from pathlib import Path

from .baseline import create_baseline_from_run


def _cmd_baseline_create(args: argparse.Namespace) -> int:
    ref = create_baseline_from_run(
        run_dir=Path(args.run_dir),
        baselines_dir=Path(args.baselines_dir),
        suite=args.suite,
        rubric_id=args.rubric_id,
        baseline_id=args.baseline_id,
        set_latest=not args.no_latest,
    )
    print(f"Baseline created: baselines/{ref.suite}/{ref.rubric_id}/snapshots/{ref.baseline_id}")
    if not args.no_latest:
        print(f"Latest updated:  baselines/{ref.suite}/{ref.rubric_id}/latest.json")
    return 0


def build_cli() -> argparse.ArgumentParser:
    # IMPORTANT: if your __main__.py already defines build_cli(), merge this in
    # rather than duplicating. Keep existing subcommands.
    p = argparse.ArgumentParser(prog="llm-judge-eval")
    sub = p.add_subparsers(dest="cmd", required=True)

    # baseline create
    p_bc = sub.add_parser(
        "baseline-create", help="Promote an eval run dir into a baseline snapshot"
    )
    p_bc.add_argument(
        "--run-dir",
        required=True,
        help="Path to eval run output directory (contains manifest.json/results.jsonl/metrics.json)",
    )
    p_bc.add_argument(
        "--baselines-dir", default="baselines", help="Baselines root directory (default: baselines)"
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
        "--baseline-id", default=None, help="Baseline id override (default: UTC timestamp)"
    )
    p_bc.add_argument("--no-latest", action="store_true", help="Do not update latest.json")
    p_bc.set_defaults(func=_cmd_baseline_create)

    return p


def main() -> int:
    parser = build_cli()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
