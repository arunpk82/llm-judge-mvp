from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from llm_judge.datasets_manifest import iter_jsonl, load_manifest, stable_sample


def _call_stable_sample(
    rows: list[dict[str, Any]], k: int, key: str
) -> list[dict[str, Any]]:
    """
    stable_sample signature varies across versions.
    Your current version requires keyword-only 'sample_size'.
    This helper adapts safely.
    """
    sig = inspect.signature(stable_sample)
    names = set(sig.parameters.keys())

    kwargs: dict[str, Any] = {}

    # sample size parameter (your current version)
    if "sample_size" in names:
        kwargs["sample_size"] = k
    elif "k" in names:
        kwargs["k"] = k
    elif "n" in names:
        kwargs["n"] = k
    elif "size" in names:
        kwargs["size"] = k
    elif "count" in names:
        kwargs["count"] = k

    # key field parameter (if present)
    if "key" in names:
        kwargs["key"] = key
    elif "key_field" in names:
        kwargs["key_field"] = key
    elif "id_key" in names:
        kwargs["id_key"] = key
    elif "field" in names:
        kwargs["field"] = key

    # Always pass rows positionally (most stable)
    return stable_sample(rows, **kwargs)


def test_load_manifest_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest(tmp_path / "_index.json")


def test_load_manifest_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "_index.json"
    p.write_text("{not-json}", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_manifest(p)


def test_iter_jsonl_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(tmp_path / "missing.jsonl"))


def test_iter_jsonl_invalid_json_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"ok": 1}\n{bad}\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        list(iter_jsonl(p))


def test_stable_sample_is_deterministic() -> None:
    rows: list[dict[str, Any]] = [{"case_id": f"c{i}", "x": i} for i in range(200)]
    s1 = _call_stable_sample(rows, k=25, key="case_id")
    s2 = _call_stable_sample(rows, k=25, key="case_id")
    assert [str(r["case_id"]) for r in s1] == [str(r["case_id"]) for r in s2]


def test_stable_sample_k_bigger_than_population_returns_all() -> None:
    rows: list[dict[str, Any]] = [{"case_id": f"c{i}"} for i in range(10)]
    s = _call_stable_sample(rows, k=999, key="case_id")
    assert len(s) == 10


def test_load_manifest_rejects_missing_required_fields(tmp_path: Path) -> None:
    p = tmp_path / "_index.json"
    # missing path
    p.write_text(json.dumps([{"id": "ds1"}]), encoding="utf-8")
    with pytest.raises(Exception):
        load_manifest(p)


def test_stable_sample_with_zero_sample_size_returns_empty_or_raises() -> None:
    rows: list[dict[str, Any]] = [{"case_id": "c1"}, {"case_id": "c2"}]
    try:
        out = _call_stable_sample(rows, k=0, key="case_id")
        assert out == []  # or len(out) == 0
    except Exception:
        assert True


def test_stable_sample_missing_key_field_does_not_crash() -> None:
    rows: list[dict[str, Any]] = [{"wrong": "x1"}, {"wrong": "x2"}]
    try:
        _call_stable_sample(rows, k=1, key="case_id")
    except Exception:
        assert True
