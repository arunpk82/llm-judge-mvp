from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from llm_judge.datasets_manifest import iter_jsonl, load_manifest, stable_sample
from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import PredictRequest

MANIFEST = Path("datasets/deterministic/_index.json")


def _validate_case_schema(row: dict[str, Any]) -> None:
    assert isinstance(row.get("case_id"), str) and row["case_id"].strip()
    assert isinstance(row.get("rubric_id"), str) and row["rubric_id"].strip()
    assert isinstance(row.get("conversation"), list) and row["conversation"]
    assert isinstance(row.get("candidate_answer"), str)
    assert isinstance(row.get("expected"), dict)
    assert isinstance(row["expected"].get("decision"), str) and row["expected"]["decision"] in {
        "pass",
        "fail",
    }


def _run_cases(rows: list[dict[str, Any]], dataset_id: str, mode: str) -> None:
    engine = get_judge_engine()

    start = time.time()
    failures = 0

    for r in rows:
        _validate_case_schema(r)

        req = PredictRequest(
            conversation=r["conversation"],
            candidate_answer=r["candidate_answer"],
            rubric_id=r["rubric_id"],
        )
        out = engine.evaluate(req)

        exp = r["expected"]

        if out.decision != exp["decision"]:
            failures += 1
            print("\n===== DETERMINISTIC CASE MISMATCH =====")
            print(f"case_id        : {r['case_id']}")
            print(f"rubric_id      : {r['rubric_id']}")
            print(f"prompt         : {r['conversation']}")
            print(f"candidate      : {r['candidate_answer']}")
            print(f"expected_dec   : {exp['decision']}")
            print(f"actual_dec     : {out.decision}")
            print(f"expected_flags : {exp.get('flags')}")
            print(f"actual_flags   : {getattr(out, 'flags', None)}")
            print(f"actual_scores  : {getattr(out, 'scores', None)}")
            print("========================================\n")

        assert out.decision == exp["decision"], f"case_id={r['case_id']}"

    duration = time.time() - start

    # Low-noise but high-signal run summary (shows up in CI logs)
    print("\n=== Deterministic Dataset Summary ===")
    print(f"Dataset        : {dataset_id}")
    print(f"Mode           : {mode}")
    print(f"Executed rows  : {len(rows)}")
    print(f"Failures       : {failures}")
    print(f"Duration (sec) : {duration:.2f}")
    print("=====================================\n")


def test_deterministic_datasets_ci_sampled() -> None:
    """Fast-lane regression: sample each dataset deterministically for PR CI."""
    specs = load_manifest(MANIFEST)
    assert specs, "manifest must list at least one dataset"

    for spec in specs:
        rows = list(iter_jsonl(spec.path))
        assert rows, f"dataset is empty: {spec.dataset_id}"
        sampled = stable_sample(rows, sample_size=spec.ci_sample_size)
        _run_cases(sampled, dataset_id=spec.dataset_id, mode=f"sampled(ci_sample_size={spec.ci_sample_size})")


@pytest.mark.nightly
def test_deterministic_datasets_nightly_full() -> None:
    """Deep-lane regression: full dataset run (scheduled CI only)."""
    # Allow opting out locally if someone accidentally triggers the marker.
    if os.getenv("LLM_JUDGE_DATASET_FULL", "1").strip().lower() not in {"1", "true", "yes"}:
        pytest.skip("LLM_JUDGE_DATASET_FULL disabled")

    specs = load_manifest(MANIFEST)
    assert specs, "manifest must list at least one dataset"

    for spec in specs:
        if not spec.nightly_full_run:
            continue
        rows = list(iter_jsonl(spec.path))
        _run_cases(rows, dataset_id=spec.dataset_id, mode="full(nightly)")