from __future__ import annotations

import json
from pathlib import Path
from typing import Any

IN_PATH = Path("datasets/deterministic/math_basic_v1.jsonl")
OUT_PATH = Path("datasets/deterministic/math_basic_v1.jsonl.new")

CASE_ID_PREFIX = "mbv1"
WIDTH = 8

# IMPORTANT: math-only rubric to avoid chat_quality relevance/format rules
FORCE_RUBRIC_ID = "math_basic"

# Optional schema version stamp (safe to add; ignored by existing code unless validated)
SCHEMA_VERSION = "deterministic.case.v1"


def _make_case_id(i: int) -> str:
    return f"{CASE_ID_PREFIX}_{i:0{WIDTH}d}"


def _migrate_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    # case_id + schema
    row["case_id"] = _make_case_id(idx)
    row["schema_version"] = SCHEMA_VERSION

    # force rubric_id for this dataset
    row["rubric_id"] = FORCE_RUBRIC_ID

    # normalize expected_* -> expected
    # Your current file uses expected_decision / expected_flags
    if "expected" not in row or not isinstance(row.get("expected"), dict):
        expected: dict[str, Any] = {}
        if "expected_decision" in row:
            expected["decision"] = row.pop("expected_decision")
        if "expected_flags" in row:
            expected["flags"] = row.pop("expected_flags")
        # ensure flags always present as list
        if "flags" not in expected or not isinstance(expected.get("flags"), list):
            expected["flags"] = []
        row["expected"] = expected
    else:
        # if expected already exists, ensure flags list exists
        exp = row["expected"]
        if "flags" not in exp or not isinstance(exp.get("flags"), list):
            exp["flags"] = []
        row["expected"] = exp

    return row


def main() -> int:
    if not IN_PATH.exists():
        raise SystemExit(f"Missing input file: {IN_PATH}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with (
        IN_PATH.open("r", encoding="utf-8") as fin,
        OUT_PATH.open("w", encoding="utf-8") as fout,
    ):
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
    print("")
    print("Next step:")
    print(f"  mv {OUT_PATH} {IN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
