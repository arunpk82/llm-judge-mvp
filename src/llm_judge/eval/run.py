from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import Message, PredictRequest

from .io import build_manifest, ensure_dir, write_json, write_jsonl
from .metrics import compute_metrics
from .spec import RunSpec


def _load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _make_run_id(prefix: str) -> str:
    # deterministic-ish: prefix + UTC timestamp + random suffix
    # random seeded from runspec for stable-ish runs if desired
    import datetime as _dt

    ts = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    suffix = f"{random.randint(0, 999999):06d}"
    return f"{prefix}-{ts}-{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible evaluation benchmark.")
    parser.add_argument("--spec", required=True, help="Path to RunSpec YAML.")
    args = parser.parse_args()

    spec = RunSpec.from_yaml(args.spec)

    # Make run deterministic-ish
    random.seed(spec.random_seed)

    # Force engine via env to match repo contract
    os.environ["JUDGE_ENGINE"] = spec.judge_engine

    dataset_path = Path(spec.dataset.path)
    rows = _load_jsonl(dataset_path)

    engine = get_judge_engine()

    run_id = _make_run_id(spec.run_id_prefix)
    run_dir = Path(spec.output_dir) / run_id
    ensure_dir(run_dir)

    manifest = build_manifest(runspec=spec, dataset_path=dataset_path)
    write_json(run_dir / "manifest.json", manifest)

    judgments: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        # Convert conversation dicts to Message objects
        conv = [Message(**m) for m in row["conversation"]]

        req = PredictRequest(
            conversation=conv,
            candidate_answer=str(row["candidate_answer"]),
            rubric_id=str(row.get("rubric_id", spec.rubric_id)),
        )

        res = engine.evaluate(req)

        judgments.append(
            {
                "case_index": i,
                "rubric_id": req.rubric_id,
                "human_decision": row.get("human_decision"),
                "human_scores": row.get("human_scores"),
                "judge_decision": getattr(res, "decision", None),
                "judge_scores": getattr(res, "scores", None),
                "judge_flags": getattr(res, "flags", None),
            }
        )

    write_jsonl(run_dir / "judgments.jsonl", judgments)

    # PR4: compute & persist metrics
    metrics = compute_metrics(judgments)
    write_json(run_dir / "metrics.json", metrics)

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
