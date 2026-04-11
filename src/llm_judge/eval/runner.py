from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from llm_judge.eval.harness import load_jsonl
from llm_judge.eval.metrics import compute_metrics
from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import PredictRequest


@dataclass(frozen=True)
class RunSpec:
    run_id_prefix: str
    dataset_path: str
    dataset_id: str
    dataset_version: str
    rubric_id: str
    judge_engine: str
    output_dir: str
    random_seed: int


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root: {path}")
    return data


def load_runspec(path: str) -> RunSpec:
    p = Path(path)
    raw = _read_yaml(p)
    dataset = raw.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("runspec.dataset must be an object")

    return RunSpec(
        run_id_prefix=str(raw.get("run_id_prefix", "run")),
        dataset_path=str(dataset.get("path")),
        dataset_id=str(dataset.get("dataset_id", "dataset")),
        dataset_version=str(dataset.get("version", "v1")),
        rubric_id=str(raw.get("rubric_id")),
        judge_engine=str(raw.get("judge_engine", "deterministic")),
        output_dir=str(raw.get("output_dir", "reports/runs")),
        random_seed=int(raw.get("random_seed", 42)),
    )


def _git_head() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_from_runspec(runspec_path: str) -> Path:
    spec = load_runspec(runspec_path)

    # Ensure deterministic run ordering.
    random.seed(spec.random_seed)

    os.environ["JUDGE_ENGINE"] = spec.judge_engine

    rows = load_jsonl(spec.dataset_path)
    engine = get_judge_engine()

    run_id = f"{spec.run_id_prefix}-{spec.dataset_id}-{spec.dataset_version}-{spec.rubric_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(spec.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    judgments_path = out_dir / "judgments.jsonl"
    judgments: list[dict[str, Any]] = []
    with open(judgments_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            req = PredictRequest(
                conversation=row["conversation"],
                candidate_answer=row["candidate_answer"],
                rubric_id=row["rubric_id"],
            )
            res = engine.evaluate(req)

            j: dict[str, Any] = {
                "case_id": row.get("case_id", i),
                "rubric_id": row.get("rubric_id"),
                "human_decision": row.get("human_decision"),
                "judge_decision": res.decision,
                "overall_score": res.overall_score,
                "scores": res.scores,
                "confidence": res.confidence,
                "flags": res.flags,
            }
            judgments.append(j)
            f.write(json.dumps(j, ensure_ascii=False) + "\n")

    metrics = compute_metrics(judgments)
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    manifest = {
        "run_id": run_id,
        "created_at": _utc_now_iso(),
        "git_head": _git_head(),
        "runspec_path": runspec_path,
        "dataset": {
            "path": spec.dataset_path,
            "dataset_id": spec.dataset_id,
            "version": spec.dataset_version,
        },
        "rubric_id": spec.rubric_id,
        "judge_engine": spec.judge_engine,
        "artifacts": {
            "judgments": str(judgments_path),
            "metrics": str(metrics_path),
        },
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return out_dir
