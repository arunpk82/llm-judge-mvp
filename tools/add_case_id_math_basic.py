#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Add case_id to a JSONL dataset (math_basic_v1) safely."
    )
    p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Input JSONL path (e.g., datasets/deterministic/math_basic_v1.jsonl)",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        required=True,
        help="Output JSONL path (e.g., datasets/deterministic/math_basic_v1_with_ids.jsonl)",
    )
    p.add_argument(
        "--prefix",
        default="mbv1",
        help="case_id prefix (default: mbv1)",
    )
    p.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting index (default: 1)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=8,
        help="Zero-padding width (default: 8 -> mbv1_00000001)",
    )
    p.add_argument(
        "--set-schema-version",
        default=None,
        help='If provided, set row["schema_version"] to this value (e.g., deterministic.case.v1)',
    )
    p.add_argument(
        "--migrate-expected",
        action="store_true",
        help="If set, migrate expected_decision/expected_flags -> expected.{decision,flags}",
    )
    p.add_argument(
        "--fail-if-case-id-exists",
        action="store_true",
        help="If set, abort if any row already has case_id.",
    )
    return p.parse_args()


def make_case_id(prefix: str, idx: int, width: int) -> str:
    return f"{prefix}_{idx:0{width}d}"


def migrate_expected_fields(row: dict[str, Any]) -> None:
    """
    Convert:
      expected_decision: "pass"|"fail"
      expected_flags: [...]
    into:
      expected: { "decision": ..., "flags": [...] }

    Leaves existing `expected` untouched if already present.
    """
    if "expected" in row and isinstance(row["expected"], dict):
        return

    decision = row.pop("expected_decision", None)
    flags = row.pop("expected_flags", None)

    expected: dict[str, Any] = {}
    if decision is not None:
        expected["decision"] = decision
    if flags is not None:
        expected["flags"] = flags

    if expected:
        row["expected"] = expected


def main() -> int:
    args = parse_args()
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    if not in_path.exists():
        raise SystemExit(f"ERROR: input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0

    with (
        in_path.open("r", encoding="utf-8") as fin,
        out_path.open("w", encoding="utf-8") as fout,
    ):
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"ERROR: {in_path}:{line_no}: invalid JSON: {e}"
                ) from e

            if not isinstance(row, dict):
                raise SystemExit(
                    f"ERROR: {in_path}:{line_no}: expected JSON object per line"
                )

            n_in += 1

            if args.fail_if_case_id_exists and "case_id" in row:
                raise SystemExit(
                    f"ERROR: {in_path}:{line_no}: case_id already exists; aborting."
                )

            # Add / overwrite case_id deterministically
            cid = make_case_id(args.prefix, args.start + (n_in - 1), args.width)
            row["case_id"] = cid

            # Optional schema_version set
            if args.set_schema_version is not None:
                row["schema_version"] = args.set_schema_version

            # Optional expected migration
            if args.migrate_expected:
                migrate_expected_fields(row)

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"Done. Read {n_in} rows, wrote {n_out} rows.")
    print(f"Output: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
