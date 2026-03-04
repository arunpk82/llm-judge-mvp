from __future__ import annotations

import random

from llm_judge.eval.run import _sample_rows_stable_hash


def _mk_rows(n: int, *, prefix: str) -> list[dict]:
    # Sampling depends only on case_id.
    return [{"case_id": f"{prefix}-{i:05d}"} for i in range(n)]


def test_stable_hash_sampling_is_deterministic_for_same_input() -> None:
    """
    EPIC-1 (scale): Deterministic sampling must be reproducible for the same dataset + seed.
    """
    seed = 123
    n_sample = 50

    rows = _mk_rows(500, prefix="base")
    s1, m1 = _sample_rows_stable_hash(rows, n=n_sample, seed=seed)
    s2, m2 = _sample_rows_stable_hash(rows, n=n_sample, seed=seed)

    assert [r["case_id"] for r in s1] == [r["case_id"] for r in s2]
    assert m1["strategy"] == "stable_hash"
    assert m2["strategy"] == "stable_hash"
    assert m1["requested_n"] == n_sample
    assert m2["requested_n"] == n_sample
    assert m1["actual_n"] == n_sample
    assert m2["actual_n"] == n_sample


def test_stable_hash_sampling_is_order_independent() -> None:
    """
    EPIC-1: Dataset ordering must not affect which cases are selected.
    """
    seed = 123
    n_sample = 50

    rows = _mk_rows(500, prefix="base")
    shuffled = rows[:]
    random.Random(999).shuffle(shuffled)

    s1, _ = _sample_rows_stable_hash(rows, n=n_sample, seed=seed)
    s2, _ = _sample_rows_stable_hash(shuffled, n=n_sample, seed=seed)

    assert [r["case_id"] for r in s1] == [r["case_id"] for r in s2]


def test_stable_hash_sampling_is_sorted_by_case_id_for_diff_stability() -> None:
    """
    EPIC-1: Downstream diffs should be stable; sampled rows must be sorted by case_id.
    """
    seed = 999
    n_sample = 20

    rows = _mk_rows(200, prefix="x")
    sampled, meta = _sample_rows_stable_hash(rows, n=n_sample, seed=seed)

    ids = [r["case_id"] for r in sampled]
    assert ids == sorted(ids), "Sampled rows must be sorted by case_id for stable diffs"
    assert meta["ordering"] == "case_id_ascending"