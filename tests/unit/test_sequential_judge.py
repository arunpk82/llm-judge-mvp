"""Tests for Gate 2 pipeline: SequentialJudge, LLM adapters, runtime wiring."""
from __future__ import annotations

import json
from typing import Literal

import pytest

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import Message, PredictRequest, PredictResponse
from llm_judge.sequential_judge import SequentialJudge, _is_low_confidence

# ================================================================
# Helpers
# ================================================================


def _make_request(question: str = "How do I reset my password?") -> PredictRequest:
    return PredictRequest(
        conversation=[Message(role="user", content=question)],
        candidate_answer="Please restart your router.",
        rubric_id="chat_quality",
    )


def _make_response(
    scores: dict[str, int] | None = None,
    flags: list[str] | None = None,
    decision: Literal["pass", "fail"] = "pass",
    overall: float = 3.5,
    confidence: float = 0.7,
) -> PredictResponse:
    return PredictResponse(
        decision=decision,
        overall_score=overall,
        scores=scores or {"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4},
        confidence=confidence,
        flags=flags or [],
    )


class _StubJudge(JudgeEngine):
    def __init__(self, response: PredictResponse) -> None:
        self._response = response
        self.call_count = 0

    def evaluate(self, request: PredictRequest) -> PredictResponse:
        self.call_count += 1
        return self._response


class _FailingJudge(JudgeEngine):
    def evaluate(self, request: PredictRequest) -> PredictResponse:
        raise RuntimeError("LLM unavailable")


# ================================================================
# SequentialJudge tests
# ================================================================


def test_gate1_confident_with_flags_skips_gate2() -> None:
    """When Gate 1 fires quality flags, Gate 2 is not called."""
    gate1_resp = _make_response(
        scores={"relevance": 1, "clarity": 1, "correctness": 1, "tone": 3},
        flags=["quality.nonsense_basic:strong"],
        decision="fail",
        overall=1.5,
    )
    gate2 = _StubJudge(_make_response(decision="pass"))
    judge = SequentialJudge(gate1=_StubJudge(gate1_resp), gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "fail"
    assert gate2.call_count == 0


def test_gate1_confident_with_correctness_flags() -> None:
    """Correctness flags also signal confidence."""
    gate1_resp = _make_response(
        scores={"relevance": 4, "clarity": 4, "correctness": 2, "tone": 4},
        flags=["correctness.definition_sanity:strong"],
        decision="fail",
        overall=3.0,
    )
    gate2 = _StubJudge(_make_response())
    judge = SequentialJudge(gate1=_StubJudge(gate1_resp), gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "fail"
    assert gate2.call_count == 0


def test_gate1_confident_with_extreme_high_scores() -> None:
    """All 5s = confident, no escalation."""
    gate1_resp = _make_response(
        scores={"relevance": 5, "clarity": 5, "correctness": 5, "tone": 5},
        decision="pass",
        overall=5.0,
    )
    gate2 = _StubJudge(_make_response())
    judge = SequentialJudge(gate1=_StubJudge(gate1_resp), gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "pass"
    assert gate2.call_count == 0


def test_gate1_confident_with_extreme_low_score() -> None:
    """Any score <=2 = confident, no escalation."""
    gate1_resp = _make_response(
        scores={"relevance": 2, "clarity": 4, "correctness": 4, "tone": 4},
        decision="fail",
        overall=3.5,
    )
    gate2 = _StubJudge(_make_response())
    judge = SequentialJudge(gate1=_StubJudge(gate1_resp), gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "fail"
    assert gate2.call_count == 0


def test_low_confidence_escalates_to_gate2() -> None:
    """Default scores, no flags, borderline overall → escalate."""
    gate1_resp = _make_response(
        scores={"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4},
        flags=[],
        decision="pass",
        overall=3.5,
    )
    gate2_resp = _make_response(
        scores={"relevance": 1, "clarity": 4, "correctness": 4, "tone": 1},
        decision="fail",
        overall=2.5,
    )

    gate1 = _StubJudge(gate1_resp)
    gate2 = _StubJudge(gate2_resp)
    judge = SequentialJudge(gate1=gate1, gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "fail"
    assert gate1.call_count == 1
    assert gate2.call_count == 1


def test_low_confidence_all_threes() -> None:
    """All 3s with borderline overall → escalate."""
    gate1_resp = _make_response(
        scores={"relevance": 3, "clarity": 3, "correctness": 3, "tone": 3},
        overall=3.0,
    )
    gate2_resp = _make_response(decision="fail", overall=2.0)

    gate2 = _StubJudge(gate2_resp)
    judge = SequentialJudge(gate1=_StubJudge(gate1_resp), gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "fail"
    assert gate2.call_count == 1


def test_gate2_failure_falls_back_to_gate1() -> None:
    """When Gate 2 raises an exception, Gate 1's result is returned."""
    gate1_resp = _make_response(
        scores={"relevance": 4, "clarity": 3, "correctness": 4, "tone": 3},
        flags=[],
        decision="pass",
        overall=3.5,
    )

    gate1 = _StubJudge(gate1_resp)
    gate2 = _FailingJudge()
    judge = SequentialJudge(gate1=gate1, gate2=gate2)

    result = judge.evaluate(_make_request())

    assert result.decision == "pass"
    assert gate1.call_count == 1


def test_gate1_always_runs() -> None:
    """Gate 1 runs on every request regardless of outcome."""
    gate1_resp = _make_response(decision="pass", overall=4.5)
    gate1 = _StubJudge(gate1_resp)
    gate2 = _StubJudge(_make_response())
    judge = SequentialJudge(gate1=gate1, gate2=gate2)

    judge.evaluate(_make_request())
    judge.evaluate(_make_request())
    judge.evaluate(_make_request())

    assert gate1.call_count == 3


# ================================================================
# _is_low_confidence tests
# ================================================================


def test_is_low_confidence_with_quality_flags() -> None:
    resp = _make_response(flags=["quality.nonsense_basic:strong"])
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_with_correctness_flags() -> None:
    resp = _make_response(flags=["correctness.definition_sanity:weak"])
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_extreme_low() -> None:
    resp = _make_response(
        scores={"relevance": 1, "clarity": 4, "correctness": 4, "tone": 4},
    )
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_extreme_high() -> None:
    resp = _make_response(
        scores={"relevance": 5, "clarity": 4, "correctness": 4, "tone": 4},
    )
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_defaults_borderline_true() -> None:
    resp = _make_response(
        scores={"relevance": 4, "clarity": 4, "correctness": 3, "tone": 4},
        overall=3.5,
    )
    assert _is_low_confidence(resp) is True


def test_is_low_confidence_defaults_high_overall_false() -> None:
    resp = _make_response(
        scores={"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4},
        overall=4.0,
    )
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_defaults_low_overall_false() -> None:
    resp = _make_response(
        scores={"relevance": 3, "clarity": 3, "correctness": 3, "tone": 3},
        overall=2.0,
    )
    assert _is_low_confidence(resp) is False


def test_is_low_confidence_mixed_non_default_scores() -> None:
    resp = _make_response(
        scores={"relevance": 4, "clarity": 4, "correctness": 4, "tone": 2},
        overall=3.5,
    )
    assert _is_low_confidence(resp) is False


# ================================================================
# LLM Judge adapter and helper tests
# ================================================================


def test_extract_json_plain() -> None:
    from llm_judge.llm_judge import _extract_json

    result = _extract_json('{"decision": "pass", "score": 4}')
    assert result["decision"] == "pass"
    assert result["score"] == 4


def test_extract_json_with_markdown_fences() -> None:
    from llm_judge.llm_judge import _extract_json

    result = _extract_json('```json\n{"decision": "fail"}\n```')
    assert result["decision"] == "fail"


def test_extract_json_with_bare_fences() -> None:
    from llm_judge.llm_judge import _extract_json

    result = _extract_json('```\n{"key": "value"}\n```')
    assert result["key"] == "value"


def test_extract_json_invalid_raises() -> None:
    from llm_judge.llm_judge import _extract_json

    with pytest.raises(json.JSONDecodeError):
        _extract_json("NOT VALID JSON")


def test_sanitize_scores_clamps_values() -> None:
    from llm_judge.llm_judge import _sanitize_scores

    parsed = {
        "decision": "pass",
        "overall_score": 7.0,
        "scores": {"relevance": 0, "clarity": 8, "correctness": 3, "tone": -1},
        "confidence": 1.5,
        "flags": [],
    }
    result = _sanitize_scores(parsed)

    assert result.overall_score == 5.0
    assert result.scores["relevance"] == 1
    assert result.scores["clarity"] == 5
    assert result.scores["correctness"] == 3
    assert result.scores["tone"] == 1
    assert result.confidence == 1.0


def test_sanitize_scores_defaults() -> None:
    from llm_judge.llm_judge import _sanitize_scores

    result = _sanitize_scores({})
    assert result.decision == "fail"
    assert result.overall_score == 3.0
    assert result.confidence == 0.7


def test_build_prompt_contains_conversation() -> None:
    from llm_judge.llm_judge import _build_prompt

    req = _make_request("What is the return policy?")
    prompt = _build_prompt(req)
    assert "USER: What is the return policy?" in prompt
    assert "Candidate answer:" in prompt


def test_build_eval_prompt_contains_conversation() -> None:
    from llm_judge.llm_judge import _build_eval_prompt

    req = _make_request("What is the return policy?")
    prompt = _build_eval_prompt(req)
    assert "USER: What is the return policy?" in prompt
    assert "Candidate answer:" in prompt


# ================================================================
# Adapter build_request / parse_response tests
# ================================================================


def test_openai_adapter_builds_correct_request() -> None:
    from llm_judge.llm_judge import OpenAIAdapter

    adapter = OpenAIAdapter(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4",
    )
    url, headers, payload = adapter.build_request("test prompt", "system prompt")

    assert url == "https://api.openai.com/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "gpt-4"
    assert payload["messages"][1]["content"] == "test prompt"
    assert payload["temperature"] == 0


def test_openai_adapter_parses_response() -> None:
    from llm_judge.llm_judge import OpenAIAdapter

    adapter = OpenAIAdapter(
        base_url="https://api.openai.com/v1",
        api_key="test-key",
        model="gpt-4",
    )
    data = {"choices": [{"message": {"content": "hello"}}]}
    assert adapter.parse_response(data) == "hello"


def test_gemini_adapter_builds_correct_request() -> None:
    from llm_judge.llm_judge import GeminiAdapter

    adapter = GeminiAdapter(api_key="gem-key", model="gemini-2.5-flash")
    url, headers, payload = adapter.build_request("test prompt", "system prompt")

    assert "gemini-2.5-flash" in url
    assert "gem-key" in url
    assert payload["contents"][0]["parts"][0]["text"] == "test prompt"
    assert payload["generationConfig"]["temperature"] == 0.0


def test_gemini_adapter_parses_response() -> None:
    from llm_judge.llm_judge import GeminiAdapter

    adapter = GeminiAdapter(api_key="key", model="model")
    data = {"candidates": [{"content": {"parts": [{"text": "result"}]}}]}
    assert adapter.parse_response(data) == "result"


def test_ollama_adapter_builds_correct_request() -> None:
    from llm_judge.llm_judge import OllamaAdapter

    adapter = OllamaAdapter(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
    )
    url, headers, payload = adapter.build_request("test prompt", "system prompt")

    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "llama3.1:8b"
    assert payload["stream"] is False
    assert payload["format"] == "json"
    assert payload["messages"][1]["content"] == "test prompt"


def test_ollama_adapter_parses_response() -> None:
    from llm_judge.llm_judge import OllamaAdapter

    adapter = OllamaAdapter(base_url="http://localhost:11434", model="model")
    data = {"message": {"content": "output"}}
    assert adapter.parse_response(data) == "output"


# ================================================================
# _get_adapter tests
# ================================================================


def test_get_adapter_gemini(monkeypatch) -> None:
    from llm_judge.llm_judge import GeminiAdapter, _get_adapter

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    adapter = _get_adapter("gemini")
    assert isinstance(adapter, GeminiAdapter)


def test_get_adapter_groq(monkeypatch) -> None:
    from llm_judge.llm_judge import OpenAIAdapter, _get_adapter

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    adapter = _get_adapter("groq")
    assert isinstance(adapter, OpenAIAdapter)


def test_get_adapter_ollama() -> None:
    from llm_judge.llm_judge import OllamaAdapter, _get_adapter

    adapter = _get_adapter("ollama")
    assert isinstance(adapter, OllamaAdapter)


def test_get_adapter_openai(monkeypatch) -> None:
    from llm_judge.llm_judge import OpenAIAdapter, _get_adapter

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    adapter = _get_adapter("openai")
    assert isinstance(adapter, OpenAIAdapter)


def test_get_adapter_gemini_missing_key(monkeypatch) -> None:
    from llm_judge.llm_judge import _get_adapter

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        _get_adapter("gemini")


def test_get_adapter_groq_missing_key(monkeypatch) -> None:
    from llm_judge.llm_judge import _get_adapter

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _get_adapter("groq")


def test_get_adapter_openai_missing_key(monkeypatch) -> None:
    from llm_judge.llm_judge import _get_adapter

    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="LLM_API_KEY"):
        _get_adapter("openai")


# ================================================================
# Runtime sequential mode test
# ================================================================


def test_runtime_sequential_mode(monkeypatch) -> None:
    """JUDGE_ENGINE=sequential creates a SequentialJudge."""
    from llm_judge.runtime import get_judge_engine
    from llm_judge.sequential_judge import SequentialJudge

    monkeypatch.setenv("JUDGE_ENGINE", "sequential")
    monkeypatch.setenv("GATE2_ENGINE", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    engine = get_judge_engine()
    assert isinstance(engine, SequentialJudge)


def test_runtime_deterministic_mode(monkeypatch) -> None:
    from llm_judge.deterministic_judge import DeterministicJudge
    from llm_judge.runtime import get_judge_engine

    monkeypatch.setenv("JUDGE_ENGINE", "deterministic")
    engine = get_judge_engine()
    assert isinstance(engine, DeterministicJudge)


def test_runtime_llm_mode_maps_to_openai(monkeypatch) -> None:
    from llm_judge.runtime import FallbackJudge, get_judge_engine

    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    engine = get_judge_engine()
    assert isinstance(engine, FallbackJudge)
