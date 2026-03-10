import json

import httpx

from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import Message, PredictRequest


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        # OpenAI-compatible envelope expected by LLMJudge
        return {
            "choices": [
                {
                    "message": {
                        "content": self._content,
                    }
                }
            ]
        }


class _FakeClient:
    def __init__(self, content: str) -> None:
        self._content = content

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, headers: dict, json: dict) -> _FakeResponse:
        return _FakeResponse(self._content)


def test_llm_judge_success_path_returns_llm_response(monkeypatch) -> None:
    # Configure engine selection
    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("JUDGE_TIMEOUT_MS", "5000")

    # Required for LLMJudge
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_S", "1")

    # Mock httpx.Client to avoid network calls
    llm_payload = {
        "decision": "pass",
        "overall_score": 4.2,
        "scores": {"relevance": 4, "clarity": 4, "correctness": 4, "tone": 5},
        "confidence": 0.9,
        "flags": ["llm_path"],
        "explanations": {
            "relevance": "Relevant.",
            "clarity": "Clear.",
            "correctness": "Consistent.",
            "tone": "Polite.",
        },
    }
    monkeypatch.setattr(
        httpx, "Client", lambda *args, **kwargs: _FakeClient(json.dumps(llm_payload))
    )

    engine = get_judge_engine()

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi there!",
        rubric_id="chat_quality",
    )

    res = engine.evaluate(req)

    # Assert we returned the LLM response (not fallback)
    assert res.flags == ["llm_path"]
    assert res.decision in ("pass", "fail")
    assert 0.0 <= res.overall_score <= 5.0
    assert res.scores["tone"] == 5


def test_llm_invalid_json_triggers_fallback(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("JUDGE_TIMEOUT_MS", "5000")

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_TIMEOUT_S", "1")

    # LLM returns invalid JSON -> LLMJudge raises -> fallback judge should be used
    monkeypatch.setattr(httpx, "Client", lambda timeout: _FakeClient("NOT_JSON"))

    engine = get_judge_engine()

    req = PredictRequest(
        conversation=[Message(role="user", content="How do I reset my router?")],
        candidate_answer="Please restart your router.",
        rubric_id="chat_quality",
    )

    res = engine.evaluate(req)

    # Fallback judge returns deterministic dimension keys
    assert "relevance" in res.scores
    assert res.decision in ("pass", "fail")


def test_engine_selector_returns_llm_engine_when_configured(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    engine = get_judge_engine()

    # We can't directly check internal wrapped types cleanly without coupling,
    # so we assert behavior: LLM path works when JSON is returned.
    llm_payload = {
        "decision": "pass",
        "overall_score": 4.0,
        "scores": {"relevance": 4, "clarity": 4, "correctness": 4, "tone": 4},
        "confidence": 0.8,
        "flags": ["llm_selected"],
        "explanations": {"relevance": "ok"},
    }

    monkeypatch.setattr(
        httpx, "Client", lambda *args, **kwargs: _FakeClient(json.dumps(llm_payload))
    )

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi",
        rubric_id="chat_quality",
    )
    res = engine.evaluate(req)
    assert res.flags == ["llm_selected"]


def test_llm_http_error_triggers_fallback(monkeypatch) -> None:
    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    import httpx

    class _ErrorClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def post(self, url, headers, json):
            raise httpx.HTTPError("boom")

    monkeypatch.setattr(httpx, "Client", lambda timeout: _ErrorClient())

    engine = get_judge_engine()

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi",
        rubric_id="chat_quality",
    )

    res = engine.evaluate(req)

    # fallback deterministic judge should run
    assert "relevance" in res.scores
