"""
Dataset CLI — list and validate registered datasets.

Usage:
    python -m llm_judge.datasets list
    python -m llm_judge.datasets validate
    python -m llm_judge.datasets inspect <dataset_id> <version>
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATASETS_DIR = Path("datasets")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file."""
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _find_datasets(root: Path) -> list[dict[str, Any]]:
    """Scan for all dataset.yaml files under root."""
    datasets: list[dict[str, Any]] = []
    if not root.exists():
        return datasets

    for ds_dir in sorted(root.iterdir()):
        meta_path = ds_dir / "dataset.yaml"
        if not meta_path.exists():
            continue

        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            continue

        data_file = meta.get("data_file", "")
        data_path = ds_dir / data_file if data_file else None
        cases = 0
        if data_path and data_path.exists():
            cases = _count_lines(data_path)

        datasets.append({
            "dataset_id": meta.get("dataset_id", "?"),
            "version": meta.get("version", "?"),
            "data_file": data_file,
            "data_path": str(data_path) if data_path else "?",
            "cases": cases,
            "content_hash": meta.get("content_hash"),
            "owner": meta.get("owner"),
            "task_type": meta.get("task_type"),
            "dir": str(ds_dir),
        })

    return datasets


def cmd_list(args: argparse.Namespace) -> int:
    datasets = _find_datasets(Path(args.root))
    if not datasets:
        print("No datasets found.")
        return 0

    print("")
    print("Registered Datasets")
    print("-" * 80)
    print(
        f"{'DATASET_ID':<18} {'VER':<6} {'CASES':>6}  "
        f"{'OWNER':<12} {'HASH':<24}"
    )
    print("-" * 80)

    for ds in datasets:
        hash_str = ds["content_hash"][:22] + "..." if ds["content_hash"] else "not set"
        print(
            f"{ds['dataset_id']:<18} {ds['version']:<6} {ds['cases']:>6}  "
            f"{ds.get('owner') or '-':<12} {hash_str:<24}"
        )

    print(f"\n{len(datasets)} dataset(s) registered.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from llm_judge.datasets.registry import DatasetRegistry

    datasets = _find_datasets(Path(args.root))
    if not datasets:
        print("No datasets found.")
        return 0

    errors = 0
    reg = DatasetRegistry(root_dir=Path(args.root))

    print("")
    print("Validating datasets...")
    print("-" * 60)

    for ds in datasets:
        try:
            reg.resolve(
                dataset_id=ds["dataset_id"],
                version=ds["version"],
            )
            print(f"  OK   {ds['dataset_id']}@{ds['version']} ({ds['cases']} cases)")
        except Exception as e:
            errors += 1
            print(f"  FAIL {ds['dataset_id']}@{ds['version']}: {e}")

    print("")
    if errors:
        print(f"{errors} dataset(s) failed validation.")
        return 1

    print(f"All {len(datasets)} dataset(s) valid.")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    from llm_judge.datasets.registry import DatasetRegistry

    reg = DatasetRegistry(root_dir=Path(args.root))
    try:
        resolved = reg.resolve(dataset_id=args.dataset_id, version=args.version)
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    data_path = resolved.data_path
    actual_hash = _sha256_file(data_path) if data_path.exists() else "file not found"

    print(f"\nDataset: {resolved.metadata.dataset_id}@{resolved.metadata.version}")
    print("-" * 60)
    print(f"  Directory:    {resolved.dataset_dir}")
    print(f"  Data file:    {resolved.metadata.data_file}")
    print(f"  Data path:    {data_path}")
    print(f"  Cases:        {_count_lines(data_path) if data_path.exists() else '?'}")
    print(f"  Owner:        {resolved.metadata.owner or '-'}")
    print(f"  Task type:    {resolved.metadata.task_type or '-'}")
    print(f"  Stored hash:  {resolved.metadata.content_hash or 'not set'}")
    print(f"  Actual hash:  {actual_hash}")

    if resolved.metadata.content_hash and actual_hash != resolved.metadata.content_hash:
        print("\n  WARNING: Hash mismatch! Data file may have been modified.")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset management for LLM-Judge.")
    parser.add_argument("--root", default=str(DEFAULT_DATASETS_DIR), help="Datasets root directory")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all registered datasets")
    sub.add_parser("validate", help="Validate all datasets (schema, integrity, hash)")

    p_inspect = sub.add_parser("inspect", help="Inspect a specific dataset")
    p_inspect.add_argument("dataset_id", help="Dataset ID")
    p_inspect.add_argument("version", help="Dataset version")

    args = parser.parse_args()

    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "validate":
        return cmd_validate(args)
    if args.cmd == "inspect":
        return cmd_inspect(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
