from __future__ import annotations

from pathlib import Path
from typing import Any

DATASET_PATH = Path("datasets/golden/v1.jsonl")
ALLOWED_ROLES = {"system", "user", "assistant"}
ALLOWED_DECISIONS = {"pass", "fail"}


def _is_nonempty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _validate_case(obj: dict[str, Any], *, line_no: int) -> None:
    # Required keys (case_id is now required)
    for key in ("case_id", "rubric_id", "conversation", "candidate_answer", "human_decision"):
        assert key in obj, f"line {line_no}: missing required key '{key}'"

    assert _is_nonempty_str(obj["case_id"]), f"line {line_no}: case_id must be a non-empty string"
    assert obj["rubric_id"] == "chat_quality", (
        f"line {line_no}: rubric_id must be 'chat_quality' for v1"
    )

    conv = obj["conversation"]
    assert isinstance(conv, list) and len(conv) >= 1, (
        f"line {line_no}: conversation must be a non-empty list"
    )
    for i, msg in enumerate(conv):
        assert isinstance(msg, dict), f"line {line_no}: conversation[{i}] must be an object"
        assert msg.get("role") in ALLOWED_ROLES, f"line {line_no}: conversation[{i}].role invalid"
        assert _is_nonempty_str(msg.get("content")), (
            f"line {line_no}: conversation[{i}].content must be non-empty"
        )

    assert _is_nonempty_str(obj["candidate_answer"]), (
        f"line {line_no}: candidate_answer must be non-empty"
    )
    assert obj["human_decision"] in ALLOWED_DECISIONS, (
        f"line {line_no}: human_decision must be pass/fail"
    )

    if "human_scores" in obj and obj["human_scores"] is not None:
        hs = obj["human_scores"]
        assert isinstance(hs, dict), f"line {line_no}: human_scores must be an object"
        for k in ("relevance", "clarity", "correctness", "tone"):
            if k in hs:
                v = hs[k]
                assert isinstance(v, int), f"line {line_no}: human_scores.{k} must be int"
                assert 1 <= v <= 5, f"line {line_no}: human_scores.{k} must be in [1..5]"

    if "rationale" in obj and obj["rationale"] is not None:
        assert _is_nonempty_str(obj["rationale"]), (
            f"line {line_no}: rationale must be non-empty string"
        )
