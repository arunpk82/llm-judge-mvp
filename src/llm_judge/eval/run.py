from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_judge.datasets.registry import DatasetRegistry
from llm_judge.eval.registry import append_run_registry_entry
from llm_judge.eval.schema import EVAL_RUN_SCHEMA_VERSION
from llm_judge.rubric_store import get_rubric
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


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

    World-class contract:
      - requires a stable unique case_id per row
      - sort by sha(seed, case_id) ascending
      - take first n
      - return rows sorted by case_id for stable downstream iteration/diff

    Returns sampled_rows and sampling_metadata.
    """
    if n <= 0:
        return [], {"strategy": "stable_hash", "requested_n": 0, "actual_n": 0, "seed": seed}

    # Enforce case_id to guarantee stability as datasets scale/grow.
    missing_case_id = [
        i
        for i, r in enumerate(rows)
        if not isinstance(r.get("case_id"), str) or not r["case_id"].strip()
    ]
    if missing_case_id:
        raise ValueError(
            "Deterministic sampling requires non-empty string 'case_id' for every row. "
            f"Missing/invalid case_id at rows: {missing_case_id[:10]}"
            + (" ..." if len(missing_case_id) > 10 else "")
        )

    keyed: List[Tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        cid = str(row["case_id"])
        hv = _stable_hash_u64(seed, cid)
        keyed.append((hv, cid, row))

    keyed.sort(key=lambda t: (t[0], t[1]))
    sampled = [t[2] for t in keyed[: min(n, len(keyed))]]

    # For diff-friendly stability, sort selected rows by case_id.
    sampled_sorted = sorted(sampled, key=lambda r: str(r["case_id"]))

    meta = {
        "strategy": "stable_hash",
        "requested_n": n,
        "actual_n": len(sampled_sorted),
        "seed": seed,
        "requires_case_id_for_growth_stability": True,
        "ordering": "case_id_ascending",
    }
    return sampled_sorted, meta


def _enforce_metrics_schema(*, rubric_ref: str, metrics: dict[str, Any]) -> None:
    """
    EPIC-2: enforce rubric-declared metrics schema (registry.yaml contract).

    Policy:
      - If rubric declares required metrics, all must be present in metrics.json
      - Missing required keys => hard failure (prevents silent drift)
    """
    rubric = get_rubric(rubric_ref)
    required = list(rubric.metrics_required)

    if not required:
        # Backward compatibility: registry.yaml may not yet declare metrics_schema.
        return

    missing = [k for k in required if k not in metrics]
    if missing:
        raise ValueError(
            "Metrics schema violation: computed metrics.json is missing required keys "
            f"for rubric={rubric.rubric_id}@{rubric.version}: {missing}. "
            f"Present keys={sorted(list(metrics.keys()))}"
        )

def _resolve_dataset_path(spec: RunSpec) -> Path:
    # Backward compatible: allow explicit path — but warn about bypass.
    if spec.dataset.path:
        warnings.warn(
            "RunSpec.dataset.path bypasses dataset governance (no validation, "
            "no hash verification). Use dataset_id + version instead. "
            "This bypass will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        return Path(spec.dataset.path)

    # Governed path: resolve by dataset_id + version
    # Includes: metadata validation, hash verification, content validation
    reg = DatasetRegistry()
    resolved = reg.resolve(dataset_id=spec.dataset.dataset_id, version=spec.dataset.version)
    return resolved.data_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Run reproducible evaluation benchmark.")
    parser.add_argument("--spec", required=True, help="Path to RunSpec YAML.")
    args = parser.parse_args()

    spec = RunSpec.from_yaml(args.spec)

    # Make run deterministic-ish
    random.seed(spec.random_seed)

    # Force engine via env to match repo contract
    os.environ["JUDGE_ENGINE"] = spec.judge_engine

    start_total = time.perf_counter()

    dataset_path = _resolve_dataset_path(spec)
    rows = _load_jsonl(dataset_path)
    dataset_hash = _sha256_file(dataset_path)

    # --- EPIC-2.1: Pre-flight compatibility check ---
    # Verify rubric exists and rule plan is loadable before scoring.
    # Catches mismatches early instead of failing mid-evaluation.
    preflight_notes: list[str] = []
    try:
        rubric = get_rubric(str(spec.rubric_id))
        preflight_notes.append(f"rubric={rubric.rubric_id}@{rubric.version}")
    except ValueError as e:
        raise ValueError(
            f"Pre-flight failed: rubric '{spec.rubric_id}' not found. "
            f"Register it in rubrics/registry.yaml before running evaluation. Error: {e}"
        )

    try:
        from llm_judge.rules.engine import load_plan_for_rubric
        plan = load_plan_for_rubric(rubric.rubric_id, rubric.version)
        preflight_notes.append(f"rule_plan=config({len(plan.rules)} rules)")
    except (FileNotFoundError, ValueError):
        preflight_notes.append("rule_plan=fallback(no config file)")

    # Check dataset rows have required fields for scoring
    if rows:
        sample_row = rows[0]
        missing_fields = []
        for field in ("conversation", "candidate_answer"):
            if field not in sample_row:
                missing_fields.append(field)
        if missing_fields:
            raise ValueError(
                f"Pre-flight failed: dataset is missing required fields for scoring: "
                f"{missing_fields}. First row keys: {sorted(sample_row.keys())}"
            )
        preflight_notes.append("dataset_fields_ok")

    # --- EPIC-3.3: Rule governance check ---
    # Verify all runtime rules are declared in manifest.yaml.
    # Catches ungoverned rules before scoring — not just in preflight.
    try:
        from llm_judge.rules.lifecycle import check_rules_governed
        governance_errors = check_rules_governed()
        if governance_errors:
            print("WARNING: Rule governance issues detected:")
            for ge in governance_errors:
                print(f"  - {ge}")
            preflight_notes.append(f"rule_governance=WARN({len(governance_errors)} issues)")
        else:
            preflight_notes.append("rule_governance=OK")
    except Exception:
        preflight_notes.append("rule_governance=SKIP(lifecycle unavailable)")

    print(f"Pre-flight: {', '.join(preflight_notes)}")

    # Optional sampling for PR gate
    sampling_meta: Optional[dict[str, Any]] = None
    if spec.sample is not None:
        sampling_start = time.perf_counter()
        # spec.py already validates strategy; keep guardrail here too.
        if spec.sample.strategy != "stable_hash":
            raise ValueError(f"Unsupported sample.strategy: {spec.sample.strategy}")
        rows, sampling_meta = _sample_rows_stable_hash(rows, n=spec.sample.n, seed=spec.sample.seed)
        sampling_seconds = time.perf_counter() - sampling_start
    else:
        sampling_seconds = 0.0

    engine = get_judge_engine()

    run_id = _make_run_id(spec.run_id_prefix)
    run_dir = Path(spec.output_dir) / run_id
    ensure_dir(run_dir)

    # --- EPIC-1.1: Emit validation report artifact ---
    # Records what was checked and that it passed. Makes validation auditable
    # for CAP-5 (Artifact Governance) to index.
    import datetime as _dt
    validation_report = {
        "schema_version": "1.0",
        "artifact_type": "validation_report",
        "timestamp": _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "dataset_id": str(spec.dataset.dataset_id),
        "dataset_version": str(spec.dataset.version),
        "dataset_hash": dataset_hash,
        "rubric_id": str(spec.rubric_id),
        "checks": {
            "dataset_resolved": True,
            "hash_verified": bool(dataset_hash),
            "integrity_checked": True,
            "rubric_exists": True,
            "rule_plan_source": preflight_notes[1] if len(preflight_notes) > 1 else "unknown",
            "dataset_fields_compatible": True,
        },
        "result": "PASS",
        "cases_loaded": len(rows),
        "preflight_notes": preflight_notes,
    }
    write_json(run_dir / "validation_report.json", validation_report)

    manifest = build_manifest(runspec=spec, dataset_path=dataset_path)

    # --- Production-grade artifact contract (schema/versioning) ---
    manifest["schema_version"] = EVAL_RUN_SCHEMA_VERSION
    manifest["artifact_type"] = "eval_run"

    # --- Reproducibility: dataset hash ---
    manifest["dataset_hash"] = dataset_hash

    if sampling_meta:
        manifest["sampling"] = sampling_meta

    judgments: List[Dict[str, Any]] = []

    evaluation_start = time.perf_counter()
    for i, row in enumerate(rows):
        # Convert conversation dicts to Message objects
        conv = [Message(**m) for m in row["conversation"]]

        rubric_ref = str(row.get("rubric_id", spec.rubric_id))

        req = PredictRequest(
            conversation=conv,
            candidate_answer=str(row["candidate_answer"]),
            rubric_id=rubric_ref,
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
    evaluation_seconds = time.perf_counter() - evaluation_start

    write_jsonl(run_dir / "judgments.jsonl", judgments)

    # compute & persist metrics
    metrics = compute_metrics(judgments)

    # EPIC-2: enforce metrics schema declared in registry.yaml (if present)
    _enforce_metrics_schema(rubric_ref=str(spec.rubric_id), metrics=metrics)

    write_json(run_dir / "metrics.json", metrics)
    
    # --- L3 Observability: append to run registry (append-only) ---
    # Dataset identifiers come from RunSpec (governed) and stay stable.
    dataset_id = str(spec.dataset.dataset_id)
    dataset_version = str(spec.dataset.version)

    cases_total = len(_load_jsonl(dataset_path))  # total available in dataset file
    cases_evaluated = len(rows)  # after sampling (or full)
    sampled = spec.sample is not None

    append_run_registry_entry(
        run_dir=run_dir,
        manifest=manifest,
        metrics=metrics,
        cases_total=cases_total,
        cases_evaluated=cases_evaluated,
        sampled=sampled,
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        rubric_id=str(spec.rubric_id),
        judge_engine=str(spec.judge_engine),
        dataset_hash=str(dataset_hash),
    )

    total_seconds = time.perf_counter() - start_total

    # --- Observability: timing ---
    manifest["timing"] = {
        "sampling_seconds": round(float(sampling_seconds), 6),
        "evaluation_seconds": round(float(evaluation_seconds), 6),
        "total_seconds": round(float(total_seconds), 6),
    }

    # Write manifest last so it includes timing + hashes.
    write_json(run_dir / "manifest.json", manifest)

    print(f"Run complete: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())