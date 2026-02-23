from __future__ import annotations

import json
from pathlib import Path

from llm_judge.rubric_store import get_rubric
from llm_judge.rules.correctness_basic import CorrectnessBasicRule
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import Message, PredictRequest

DATASET = Path("datasets/deterministic/math_basic_v1.jsonl")


def _base_flags(flags: list[str]) -> set[str]:
    return {f.split(":", 1)[0] for f in flags}


def _flags_from_rule_result(rr: object) -> list[str]:
    """
    Supports both possible rule APIs:
    - rule returns RuleResult with `.flags` as list[Flag]
    - rule returns RuleResult with `.flags` as list[str]
    - rule mutates ctx.flags and returns None
    """
    if rr is None:
        return []

    flags = getattr(rr, "flags", None)
    if not flags:
        return []

    out: list[str] = []
    for f in flags:
        # Flag dataclass -> has .id
        fid = getattr(f, "id", None)
        if isinstance(fid, str):
            out.append(fid)
        elif isinstance(f, str):
            out.append(f)
    return out


def test_math_basic_deterministic_dataset_regression() -> None:
    """
    Rule-level deterministic regression at scale.

    NOTE:
    We do NOT call score_candidate() here because score_candidate follows the rubric plan.
    This test validates the correctness.basic rule behavior directly across many cases.
    """
    assert DATASET.exists(), f"Missing dataset file: {DATASET}"

    rubric = get_rubric("chat_quality")

    # CI sampling; raise later or run full dataset in nightly
    max_rows = 300

    rows = 0
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            req = PredictRequest(
                conversation=[Message(**m) for m in obj["conversation"]],
                candidate_answer=obj["candidate_answer"],
                rubric_id=obj["rubric_id"],
            )

            ctx = RuleContext(request=req, rubric=rubric)

            # Support both rule contracts: return RuleResult and/or mutate ctx.flags
            rr = CorrectnessBasicRule().apply(ctx)

            got_from_ctx = _base_flags(list(getattr(ctx, "flags", [])))
            got_from_rr = _base_flags(_flags_from_rule_result(rr))

            got = got_from_ctx | got_from_rr

            expected_flags = set(obj.get("expected_flags", []))
            expected_decision = obj.get("expected_decision")

            # Dataset semantics:
            # - fail rows must contain expected flags
            # - pass rows must NOT contain expected flags
            if expected_decision == "fail":
                missing = expected_flags - got
                assert not missing, {
                    "missing_flags": sorted(missing),
                    "got_flags": sorted(got),
                    "meta": obj.get("meta", {}),
                    "ctx_flags": list(getattr(ctx, "flags", [])),
                    "rr_flags": sorted(got_from_rr),
                }
            elif expected_decision == "pass":
                unexpected = expected_flags & got
                assert not unexpected, {
                    "unexpected_flags": sorted(unexpected),
                    "got_flags": sorted(got),
                    "meta": obj.get("meta", {}),
                    "ctx_flags": list(getattr(ctx, "flags", [])),
                    "rr_flags": sorted(got_from_rr),
                }

            rows += 1
            if rows >= max_rows:
                break

    assert rows > 0