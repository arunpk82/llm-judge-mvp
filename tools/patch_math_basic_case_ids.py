# tools/patch_math_basic_case_ids.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"Invalid JSON on line {i} of {path}: {e}") from e
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _normalize_expected(row: dict[str, Any]) -> None:
    """
    Supports either:
      - expected: { decision, flags }
      - expected_decision + expected_flags
      - expected_decision only
    Produces/ensures: expected: { decision: str, flags: list[str] }
    """
    if isinstance(row.get("expected"), dict):
        exp = row["expected"]
        # Ensure keys exist with correct types
        decision = exp.get("decision")
        flags = exp.get("flags")
        if not isinstance(decision, str):
            # fall back to old fields if present
            decision = row.get("expected_decision")
        if not isinstance(flags, list):
            flags = row.get("expected_flags", [])
        if not isinstance(flags, list):
            flags = []
        row["expected"] = {"decision": str(decision), "flags": [str(x) for x in flags]}
        return

    # old schema
    decision = row.get("expected_decision")
    if decision is None:
        # sometimes older data used "expected" as a string; tolerate it
        if isinstance(row.get("expected"), str):
            decision = row["expected"]
        else:
            raise SystemExit(
                f"Row missing expected/expected_decision: keys={list(row.keys())}"
            )

    flags = row.get("expected_flags", [])
    if not isinstance(flags, list):
        flags = []
    row["expected"] = {"decision": str(decision), "flags": [str(x) for x in flags]}

    # Optional cleanup (keep for backward-compat if you want)
    if "expected_decision" in row:
        del row["expected_decision"]
    if "expected_flags" in row:
        del row["expected_flags"]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Patch math_basic_v1.jsonl with case_id (+ optional rubric_id/expected normalization)."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--prefix", default="mbv1_", help="case_id prefix (default: mbv1_)")
    ap.add_argument(
        "--start", type=int, default=1, help="starting sequence number (default: 1)"
    )
    ap.add_argument("--width", type=int, default=8, help="zero-pad width (default: 8)")
    ap.add_argument(
        "--force-rubric-id",
        default=None,
        help='If set, overwrite rubric_id for all rows (e.g., "math_basic")',
    )
    ap.add_argument(
        "--normalize-expected",
        action="store_true",
        help="Normalize expected fields into expected:{decision,flags}",
    )

    args = ap.parse_args()
    inp = Path(args.inp)
    out = Path(args.out)

    rows = _read_jsonl(inp)

    n_case_added = 0
    n_rubric_forced = 0
    n_expected_norm = 0

    seq = args.start
    for row in rows:
        # case_id
        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            row["case_id"] = f"{args.prefix}{seq:0{args.width}d}"
            n_case_added += 1
        seq += 1

        # rubric_id
        if args.force_rubric_id is not None:
            if row.get("rubric_id") != args.force_rubric_id:
                row["rubric_id"] = args.force_rubric_id
                n_rubric_forced += 1

        # expected normalization
        if args.normalize_expected:
            _normalize_expected(row)
            n_expected_norm += 1

    _write_jsonl(out, rows)

    print("Patch complete")
    print(f"- rows: {len(rows)}")
    print(f"- case_id added: {n_case_added}")
    if args.force_rubric_id is not None:
        print(f"- rubric_id forced to '{args.force_rubric_id}': {n_rubric_forced}")
    if args.normalize_expected:
        print(f"- expected normalized: {n_expected_norm}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
