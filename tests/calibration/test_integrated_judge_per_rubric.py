"""Per-rubric prompt loading tests for CP-1c-b.2 Concern 2.

Verifies that :class:`IntegratedJudge` resolves prompts per
``rubric_id`` rather than caching a single hardcoded template at
init. Uses :mod:`unittest.mock` to stand in for the prompt loader so
the tests do not require a config fixture for every rubric on disk
(today only ``chat_quality`` has versioned prompts in
``configs/prompts/``).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_judge.calibration.prompts import PromptTemplate
from llm_judge.integrated_judge import IntegratedJudge


def _template(rubric_id: str) -> PromptTemplate:
    return PromptTemplate(
        prompt_id=rubric_id,
        version="v1",
        system_prompt=f"system::{rubric_id}",
        user_template="conv: {conversation}\nans: {candidate_answer}\ndims: {dimensions}",
        dimensions=["relevance", "clarity"],
        created_at="2026-04-27",
        author="test",
    )


def test_chat_quality_rubric_loads_chat_quality_prompt() -> None:
    """Loading prompt for chat_quality returns chat_quality's template."""
    judge = IntegratedJudge()
    template = judge._ensure_prompt_loaded("chat_quality")
    assert template is not None
    assert template.prompt_id == "chat_quality"


def test_per_rubric_loader_passes_rubric_id_through() -> None:
    """The loader is called with the requested rubric_id, not a hardcode."""
    judge = IntegratedJudge()
    fake = _template("math_basic")
    with patch(
        "llm_judge.integrated_judge.load_latest_prompt",
        return_value=fake,
    ) as mock_loader:
        result = judge._ensure_prompt_loaded("math_basic")
    mock_loader.assert_called_once_with("math_basic")
    assert result is fake


def test_per_rubric_caching_works() -> None:
    """Loading the same rubric twice hits the cache (loader called once)."""
    judge = IntegratedJudge()
    fake = _template("chat_quality")
    with patch(
        "llm_judge.integrated_judge.load_latest_prompt",
        return_value=fake,
    ) as mock_loader:
        judge._ensure_prompt_loaded("chat_quality")
        judge._ensure_prompt_loaded("chat_quality")
    assert mock_loader.call_count == 1


def test_multi_rubric_session_no_cross_contamination() -> None:
    """Distinct rubrics keep distinct cached templates."""
    judge = IntegratedJudge()
    cq = _template("chat_quality")
    mb = _template("math_basic")

    def fake_loader(rubric_id: str) -> PromptTemplate:
        return {"chat_quality": cq, "math_basic": mb}[rubric_id]

    with patch(
        "llm_judge.integrated_judge.load_latest_prompt",
        side_effect=fake_loader,
    ):
        cq1 = judge._ensure_prompt_loaded("chat_quality")
        mb1 = judge._ensure_prompt_loaded("math_basic")
        cq2 = judge._ensure_prompt_loaded("chat_quality")

    assert cq1 is cq2  # cache hit
    assert cq1 is not mb1  # distinct templates
    assert cq1.prompt_id == "chat_quality"
    assert mb1.prompt_id == "math_basic"


def test_unregistered_rubric_raises() -> None:
    """Loading prompt for unregistered rubric surfaces FileNotFoundError."""
    judge = IntegratedJudge()
    with pytest.raises(FileNotFoundError):
        judge._ensure_prompt_loaded("does_not_exist_12345")


def test_resolve_prompt_falls_back_to_none_on_missing() -> None:
    """The graceful resolver returns None when no prompt is on disk."""
    judge = IntegratedJudge()
    assert judge._resolve_prompt("does_not_exist_12345") is None
