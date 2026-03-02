from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IN_PATH = Path("datasets/deterministic/math_basic_v1.jsonl")
OUT_PATH = Path("datasets/deterministic/math_basic_v1.jsonl.new")

CASE_ID_PREFIX = "mbv1"
WIDTH = 8


def _make_case_id(i: int) -> str:
    return f"{CASE_ID_PREFIX}_{i:0{WIDTH}d}"


def _migrate_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    # Add schema + case_id
    row["schema_version"] = "deterministic.case.v1"
    row["case_id"] = _make_case_id(idx)

    # Migrate expected_* -> expected
    if "expected" not in row:
        expected: dict[str, Any] = {}
        if "expected_decision" in row:
            expected["decision"] = row.pop("expected_decision")
        if "expected_flags" in row:
            expected["flags"] = row.pop("expected_flags")
        if expected:
            row["expected"] = expected

    return row


def main() -> int:
    if not IN_PATH.exists():
        raise SystemExit(f"Missing input file: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{IN_PATH}:{line_no}: invalid JSON: {e}") from e

            if not isinstance(row, dict):
                raise SystemExit(f"{IN_PATH}:{line_no}: expected JSON object per line")

            n += 1
            row = _migrate_row(row, n)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"OK: migrated {n} rows")
    print(f"Output written to: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
