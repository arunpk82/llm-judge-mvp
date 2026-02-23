from fastapi.testclient import TestClient

from llm_judge.deterministic_judge import DeterministicJudge
from llm_judge.main import app
from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import Message, PredictRequest

client = TestClient(app, raise_server_exceptions=True)


def test_app_lifespan() -> None:
    with TestClient(app) as c:
        resp = c.get("/health")
        assert resp.status_code == 200


def test_health_ok() -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_happy_path() -> None:
    payload = {
        "conversation": [{"role": "user", "content": "Hello"}],
        "candidate_answer": "Hi there",
        "rubric_id": "chat_quality",
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in ("pass", "fail")
    assert 0.0 <= float(body["overall_score"]) <= 5.0
    assert isinstance(body["scores"], dict)
    assert set(body["scores"].keys()) == {"relevance", "clarity", "correctness", "tone"}
    for v in body["scores"].values():
        assert 1 <= int(v) <= 5

    assert 0.0 <= float(body["confidence"]) <= 1.0
    assert isinstance(body["flags"], list)


def test_predict_missing_required_fields_fails_fast() -> None:
    payload = {"candidate_answer": "Hi"}
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422
    body = resp.json()
    assert "detail" in body


def test_logging_configuration_runs() -> None:
    from llm_judge.logging import configure_logging

    # This should not raise
    configure_logging()


def test_chat_quality_rubric_exists() -> None:
    from llm_judge.rubrics import RUBRICS

    assert "chat_quality" in RUBRICS
    rubric = RUBRICS["chat_quality"]
    assert rubric.version == "v1"
    assert "clarity" in rubric.dimensions


def test_scorer_unknown_rubric_fails_cleanly() -> None:
    from llm_judge.scorer import score_candidate

    req = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi",
        rubric_id="does_not_exist",
    )
    res = score_candidate(req)
    assert res.decision == "fail"
    assert "unknown_rubric" in res.flags


def test_scorer_low_relevance_flag_when_unrelated() -> None:
    from llm_judge.scorer import score_candidate

    req = PredictRequest(
        conversation=[Message(role="user", content="How do I reset my router?")],
        candidate_answer="Bananas are yellow and grow in bunches.",
        rubric_id="chat_quality",
    )
    res = score_candidate(req)

    assert res.scores["relevance"] <= 2
    assert "low_relevance" in res.flags


def test_scorer_rude_tone_flag_when_rude_words_present() -> None:
    from llm_judge.scorer import score_candidate

    req = PredictRequest(
        conversation=[Message(role="user", content="Can you help me?")],
        candidate_answer="That is a stupid question. Shut up.",
        rubric_id="chat_quality",
    )
    res = score_candidate(req)

    assert res.scores["tone"] <= 2
    assert "rude_tone" in res.flags


def test_scorer_polite_language_improves_tone() -> None:
    from llm_judge.scorer import score_candidate

    req = PredictRequest(
        conversation=[Message(role="user", content="My internet is slow")],
        candidate_answer="Please restart your router. Thanks!",
        rubric_id="chat_quality",
    )
    res = score_candidate(req)

    assert res.scores["tone"] >= 4


def test_predict_response_contains_explanations() -> None:
    payload = {
        "conversation": [{"role": "user", "content": "How do I reset my router?"}],
        "candidate_answer": "Please restart it and try again.",
        "rubric_id": "chat_quality",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert "explanations" in body
    assert isinstance(body["explanations"], dict)
    assert "relevance" in body["explanations"]
    assert "tone" in body["explanations"]


def test_predict_uses_judge_engine_contract() -> None:
    payload = {
        "conversation": [{"role": "user", "content": "Hello"}],
        "candidate_answer": "Hi there!",
        "rubric_id": "chat_quality",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200

    body = resp.json()
    assert "decision" in body
    assert "scores" in body


def test_engine_selector_defaults_to_deterministic(monkeypatch) -> None:
    monkeypatch.delenv("JUDGE_ENGINE", raising=False)
    engine = get_judge_engine()
    assert isinstance(engine, DeterministicJudge)


def test_llm_timeout_falls_back_to_deterministic(monkeypatch) -> None:
    # Force LLM engine with a very small timeout
    monkeypatch.setenv("JUDGE_ENGINE", "llm")
    monkeypatch.setenv("JUDGE_TIMEOUT_MS", "1")
    monkeypatch.setenv("SIMULATE_LLM_TIMEOUT", "1")

    engine = get_judge_engine()

    # Evaluate via engine directly (fast, deterministic)

    req = PredictRequest(
        conversation=[Message(role="user", content="How do I reset my router?")],
        candidate_answer="Please restart your router.",
        rubric_id="chat_quality",
    )

    res = engine.evaluate(req)

    # DeterministicJudge produces these keys
    assert "relevance" in res.scores
    assert res.decision in ("pass", "fail")
