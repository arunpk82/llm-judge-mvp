from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def best_effort_git_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_manifest(*, runspec: Any, dataset_path: Path) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "git_sha": best_effort_git_sha(),
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "runspec": asdict(runspec),
        "env": {
            # capture only non-sensitive env hints
            "judge_engine": os.getenv("JUDGE_ENGINE"),
            "judge_timeout_ms": os.getenv("JUDGE_TIMEOUT_MS"),
        },
    }
