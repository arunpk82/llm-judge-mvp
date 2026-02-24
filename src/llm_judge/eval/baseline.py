from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaselineRef:
    suite: str
    rubric_id: str
    baseline_id: str
    created_at_utc: str
    source_run_dir: str


def _utc_now_compact() -> str:
    # Example: 20260224T154455Z
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def infer_suite_from_manifest(manifest: dict[str, Any]) -> str:
    """
    Best-effort inference. We try common keys found in run manifests.
    Fall back to 'unknown_suite' if not present.

    You can tighten this once we confirm manifest schema in Phase 2.
    """
    for key in ("suite", "dataset_id", "dataset", "dataset_name", "runspec_name"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "unknown_suite"


def infer_rubric_id_from_manifest(manifest: dict[str, Any]) -> str:
    """
    Milestone A added rubric info into manifest v2 via eval/io.py.
    We support a few common shapes.
    """
    rubric = manifest.get("rubric")
    if isinstance(rubric, dict):
        rid = rubric.get("id") or rubric.get("rubric_id")
        if isinstance(rid, str) and rid.strip():
            return rid.strip()

    # fallback keys
    for key in ("rubric_id", "rubric"):
        val = manifest.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    return "unknown_rubric"


def create_baseline_from_run(
    *,
    run_dir: Path,
    baselines_dir: Path,
    suite: str | None = None,
    rubric_id: str | None = None,
    baseline_id: str | None = None,
    set_latest: bool = True,
) -> BaselineRef:
    """
    Promote a run directory into a baseline snapshot.

    run_dir must contain:
      - manifest.json
      - results.jsonl
      - metrics.json

    Writes:
      baselines/<suite>/<rubric_id>/snapshots/<baseline_id>/{manifest.json,results.jsonl,metrics.json}
      baselines/<suite>/<rubric_id>/latest.json  (if set_latest)
    """
    run_dir = run_dir.resolve()
    baselines_dir = baselines_dir.resolve()

    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    metrics_path = run_dir / "metrics.json"

    _require_file(manifest_path)
    _require_file(results_path)
    _require_file(metrics_path)

    manifest = _read_json(manifest_path)

    suite_final = (suite or infer_suite_from_manifest(manifest)).strip()
    rubric_final = (rubric_id or infer_rubric_id_from_manifest(manifest)).strip()
    baseline_final = (baseline_id or _utc_now_compact()).strip()

    dst_dir = baselines_dir / suite_final / rubric_final / "snapshots" / baseline_final
    dst_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(manifest_path, dst_dir / "manifest.json")
    shutil.copy2(results_path, dst_dir / "results.jsonl")
    shutil.copy2(metrics_path, dst_dir / "metrics.json")

    ref = BaselineRef(
        suite=suite_final,
        rubric_id=rubric_final,
        baseline_id=baseline_final,
        created_at_utc=_utc_now_compact(),
        source_run_dir=str(run_dir),
    )

    if set_latest:
        latest_path = baselines_dir / suite_final / rubric_final / "latest.json"
        _write_json(
            latest_path,
            {
                "suite": ref.suite,
                "rubric_id": ref.rubric_id,
                "baseline_id": ref.baseline_id,
                "created_at_utc": ref.created_at_utc,
                "source_run_dir": ref.source_run_dir,
            },
        )

    return ref
