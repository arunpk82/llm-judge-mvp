from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalize_expected(row: dict[str, Any]) -> None:
    """
    Normalize expected format to:
      expected: { decision: "pass|fail", flags: [...] }

    Supports legacy keys:
      expected_decision, expected_flags
    """
    if isinstance(row.get("expected"), dict):
        exp = row["expected"]
        # Ensure keys exist
        exp.setdefault("decision", row.get("expected_decision"))
        exp.setdefault("flags", row.get("expected_flags", []))
        # Clean up
        if exp.get("decision") is None and isinstance(row.get("expected_decision"), str):
            exp["decision"] = row["expected_decision"]
        if exp.get("flags") is None:
            exp["flags"] = []
        row["expected"] = {"decision": exp["decision"], "flags": list(exp.get("flags", []))}
    else:
        row["expected"] = {
            "decision": row.get("expected_decision"),
            "flags": list(row.get("expected_flags", [])),
        }

    # Remove legacy keys if present
    row.pop("expected_decision", None)
    row.pop("expected_flags", None)

    if row["expected"]["decision"] not in {"pass", "fail"}:
        raise ValueError(f"Invalid expected.decision: {row['expected']['decision']}")


def normalize_file(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)

        # 1) case_id
        cid = row.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            row["case_id"] = f"mbv1_{idx:08d}"

        # 2) force rubric_id to math_basic
        row["rubric_id"] = "math_basic"

        # 3) expected schema normalization
        _normalize_expected(row)

        out_lines.append(json.dumps(row, ensure_ascii=False))

    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    dataset_path = Path("datasets/deterministic/math_basic_v1.jsonl")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    normalize_file(dataset_path)
    print(f"OK: normalized {dataset_path}")