from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

DATASET_PATH = Path("datasets/golden/v1.jsonl")
ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_DECISIONS = {"pass", "fail"}


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _validate_case(obj: dict[str, Any], *, line_no: int) -> None:
    # Required keys (v1 baseline)
    for key in ("rubric_id", "conversation", "candidate_answer", "human_decision"):
        assert key in obj, f"line {line_no}: missing required key '{key}'"

    # case_id is strongly recommended; optional for backward compatibility
    if "case_id" in obj and obj["case_id"] is not None:
        assert _is_nonempty_str(
            obj["case_id"]
        ), f"line {line_no}: case_id must be a non-empty string"

    assert (
        obj["rubric_id"] == "chat_quality"
    ), f"line {line_no}: rubric_id must be 'chat_quality' for v1"

    conv = obj["conversation"]
    assert (
        isinstance(conv, list) and len(conv) >= 1
    ), f"line {line_no}: conversation must be a non-empty list"
    for i, msg in enumerate(conv):
        assert isinstance(
            msg, dict
        ), f"line {line_no}: conversation[{i}] must be an object"
        assert (
            msg.get("role") in ALLOWED_ROLES
        ), f"line {line_no}: conversation[{i}].role invalid"
        assert _is_nonempty_str(
            msg.get("content")
        ), f"line {line_no}: conversation[{i}].content must be non-empty"

    assert _is_nonempty_str(
        obj["candidate_answer"]
    ), f"line {line_no}: candidate_answer must be non-empty"
    assert (
        obj["human_decision"] in ALLOWED_DECISIONS
    ), f"line {line_no}: human_decision must be pass/fail"

    # Optional human_scores validation
    if "human_scores" in obj and obj["human_scores"] is not None:
        hs = obj["human_scores"]
        assert isinstance(hs, dict), f"line {line_no}: human_scores must be an object"
        for k in ("relevance", "clarity", "correctness", "tone"):
            if k in hs:
                v = hs[k]
                assert isinstance(
                    v, int
                ), f"line {line_no}: human_scores.{k} must be int"
                assert (
                    1 <= v <= 5
                ), f"line {line_no}: human_scores.{k} must be in [1..5]"

    # Optional tags validation
    if "tags" in obj and obj["tags"] is not None:
        tags = obj["tags"]
        assert isinstance(tags, list), f"line {line_no}: tags must be a list"
        for t in tags:
            assert _is_nonempty_str(
                t
            ), f"line {line_no}: each tag must be non-empty string"

    # Optional rationale validation (you already use this)
    if "rationale" in obj and obj["rationale"] is not None:
        assert _is_nonempty_str(
            obj["rationale"]
        ), f"line {line_no}: rationale must be non-empty string"


@pytest.mark.unit
def test_golden_dataset_v1_jsonl_is_valid() -> None:
    assert DATASET_PATH.exists(), f"Missing dataset file: {DATASET_PATH}"

    seen_ids: set[str] = set()
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            assert line, f"line {line_no}: empty line is not allowed in JSONL"
            obj = json.loads(line)
            assert isinstance(
                obj, dict
            ), f"line {line_no}: each JSONL row must be an object"

            _validate_case(obj, line_no=line_no)

            cid = obj.get("case_id")
            if cid is not None:
                assert cid not in seen_ids, f"line {line_no}: duplicate case_id '{cid}'"
                seen_ids.add(cid)
