from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _stable_hash_u64(seed: int, s: str) -> int:
    h = hashlib.sha256(f"{seed}:{s}".encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _sample_rows_stable_hash(
    rows: List[dict[str, Any]],
    *,
    n: int,
    seed: int,
) -> Tuple[List[dict[str, Any]], dict[str, Any]]:
    """
    Deterministic sampling that remains stable even if dataset grows.

    Rule:
      - derive a stable key per row: prefer row['case_id'], else fallback to f"idx:{i}"
      - sort by sha(seed, key) ascending
      - take first n

    Returns sampled_rows and sampling_metadata.
    """
    if n <= 0:
        return [], {"strategy": "stable_hash", "n": 0, "seed": seed}

    keyed: List[Tuple[int, int, dict[str, Any]]] = []
    for i, row in enumerate(rows):
        cid = row.get("case_id")
        key = str(cid) if isinstance(cid, str) and cid.strip() else f"idx:{i}"
        hv = _stable_hash_u64(seed, key)
        keyed.append((hv, i, row))

    keyed.sort(key=lambda t: (t[0], t[1]))
    sampled = [t[2] for t in keyed[: min(n, len(keyed))]]

    meta = {
        "strategy": "stable_hash",
        "requested_n": n,
        "actual_n": len(sampled),
        "seed": seed,
        "requires_case_id_for_growth_stability": True,
    }
    return sampled, meta


def _get_sample_config(spec: Any) -> Optional[dict[str, Any]]:
    """
    Backward compatible:
    - If RunSpec has attribute .sample (dict-like), use it.
    - Otherwise return None.

    This lets you land C1 code even before RunSpec is upgraded,
    as long as pr_gate.yaml remains unchanged. Once RunSpec gains
    `sample`, this automatically activates.
    """
    sample = getattr(spec, "sample", None)
    if sample is None:
        return None
    if isinstance(sample, dict):
        return sample
    # If you model sample as a dataclass later, try to read fields
    try:
        return {
            "n": getattr(sample, "n", None),
            "seed": getattr(sample, "seed", None),
            "strategy": getattr(sample, "strategy", None),
        }
    except Exception:
        return None


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

    # Optional sampling for PR gate
    sampling_meta: Optional[dict[str, Any]] = None
    sample_cfg = _get_sample_config(spec)
    if sample_cfg:
        strategy = str(sample_cfg.get("strategy") or "stable_hash")
        n = sample_cfg.get("n")
        seed = sample_cfg.get("seed", spec.random_seed)
        if isinstance(n, int) and n > 0:
            if strategy != "stable_hash":
                raise ValueError(f"Unsupported sample.strategy: {strategy}")
            if not isinstance(seed, int):
                raise ValueError("sample.seed must be an int")
            rows, sampling_meta = _sample_rows_stable_hash(rows, n=n, seed=seed)

    engine = get_judge_engine()

    run_id = _make_run_id(spec.run_id_prefix)
    run_dir = Path(spec.output_dir) / run_id
    ensure_dir(run_dir)

    manifest = build_manifest(runspec=spec, dataset_path=dataset_path)
    if sampling_meta:
        manifest["sampling"] = sampling_meta
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
                # carry through for traceability/debugging
                "case_id": row.get("case_id"),
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
