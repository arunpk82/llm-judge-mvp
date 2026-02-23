from __future__ import annotations

import argparse

from llm_judge.eval.runner import run_from_runspec


def main() -> None:
    ap = argparse.ArgumentParser(description="Run deterministic LLM-judge evaluation")
    ap.add_argument(
        "--runspec",
        default="configs/runspecs/pr_gate.yaml",
        help="Path to YAML runspec",
    )
    args = ap.parse_args()

    out_dir = run_from_runspec(args.runspec)
    print(f"Run complete: {out_dir}")


if __name__ == "__main__":
    main()
