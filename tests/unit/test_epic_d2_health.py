"""
EPIC-D2: Service Health & Readiness Probes — Tests.

Acceptance criteria verified:
  AC-1: /ready returns 503 with details if any check fails
  AC-2: /health/dependencies checks LLM provider when JUDGE_ENGINE=llm
  AC-3: All responses include per-check detail in JSON body
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from llm_judge.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# =====================================================================
# /health — liveness (always 200)
# =====================================================================


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_health_always_ok(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Health is a liveness check — env vars don't affect it."""
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", "/nonexistent")
        resp = client.get("/health")
        assert resp.status_code == 200


# =====================================================================
# /ready — readiness (503 on failure)
# =====================================================================


class TestReadyEndpoint:
    def test_ready_returns_200_when_all_ok(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "configs"
        cfg.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        (data / "reports").mkdir()
        (data / "datasets").mkdir()

        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(data))

        resp = client.get("/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ready"
        assert body["ok"] is True
        assert body["config_root"]["ok"] is True
        assert body["state_root"]["ok"] is True
        assert body["state_root"]["writable"] is True

    def test_ready_returns_503_missing_config(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(tmp_path / "nonexistent"))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path))
        (tmp_path / "reports").mkdir()
        (tmp_path / "datasets").mkdir()

        resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "not_ready"
        assert body["ok"] is False
        assert body["config_root"]["ok"] is False

    def test_ready_returns_503_missing_state(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = tmp_path / "configs"
        cfg.mkdir()
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path / "missing"))

        resp = client.get("/ready")
        assert resp.status_code == 503
        body = resp.json()
        assert body["state_root"]["ok"] is False

    def test_ready_includes_resolved_paths(
        self,
        client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC-3: Responses include per-check detail — including resolved paths."""
        cfg = tmp_path / "configs"
        cfg.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        (data / "reports").mkdir()
        (data / "datasets").mkdir()

        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(data))

        resp = client.get("/ready")
        body = resp.json()

        # All checks include path and exists fields
        for key in ("config_root", "state_root", "baselines_root", "datasets_root"):
            assert "path" in body[key], f"{key} missing 'path'"
            assert "exists" in body[key], f"{key} missing 'exists'"
            assert "ok" in body[key], f"{key} missing 'ok'"


# =====================================================================
# /health/dependencies — LLM provider check
# =====================================================================


class TestHealthDependenciesEndpoint:
    def test_deterministic_engine_returns_200_not_configured(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JUDGE_ENGINE", "deterministic")
        resp = client.get("/health/dependencies")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["dependencies"]["llm_provider"]["status"] == "not_configured"
        assert body["dependencies"]["llm_provider"]["ok"] is True

    def test_default_engine_is_deterministic(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("JUDGE_ENGINE", raising=False)
        resp = client.get("/health/dependencies")
        assert resp.status_code == 200
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["engine"] == "deterministic"

    def test_llm_engine_no_api_key_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JUDGE_ENGINE", "openai")
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        resp = client.get("/health/dependencies")
        assert resp.status_code == 503
        body = resp.json()
        assert body["status"] == "unhealthy"
        assert body["dependencies"]["llm_provider"]["ok"] is False
        assert "LLM_API_KEY" in body["dependencies"]["llm_provider"]["error"]

    def test_gemini_no_api_key_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JUDGE_ENGINE", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        resp = client.get("/health/dependencies")
        assert resp.status_code == 503
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["ok"] is False
        assert "GEMINI_API_KEY" in body["dependencies"]["llm_provider"]["error"]

    def test_groq_no_api_key_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JUDGE_ENGINE", "groq")
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        resp = client.get("/health/dependencies")
        assert resp.status_code == 503
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["ok"] is False

    def test_unknown_engine_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("JUDGE_ENGINE", "unknown_provider")
        resp = client.get("/health/dependencies")
        assert resp.status_code == 503
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["ok"] is False
        assert "Unknown engine" in body["dependencies"]["llm_provider"]["error"]

    def test_llm_provider_reachable_returns_200(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate a reachable provider via mock."""
        monkeypatch.setenv("JUDGE_ENGINE", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        class FakeResponse:
            status_code = 200

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, **kw):
                return FakeResponse()

        with patch("llm_judge.main.httpx.Client", return_value=FakeClient()):
            resp = client.get("/health/dependencies")

        assert resp.status_code == 200
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["ok"] is True
        assert body["dependencies"]["llm_provider"]["http_status"] == 200

    def test_llm_provider_unreachable_returns_503(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Simulate connection failure via mock."""
        monkeypatch.setenv("JUDGE_ENGINE", "openai")
        monkeypatch.setenv("LLM_API_KEY", "test-key")

        import httpx

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, **kw):
                raise httpx.ConnectError("Connection refused")

        with patch("llm_judge.main.httpx.Client", return_value=FakeClient()):
            resp = client.get("/health/dependencies")

        assert resp.status_code == 503
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["ok"] is False
        assert "Connection refused" in body["dependencies"]["llm_provider"]["error"]

    def test_response_includes_engine_field(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AC-3: Per-check detail includes engine name."""
        monkeypatch.setenv("JUDGE_ENGINE", "gemini")
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        resp = client.get("/health/dependencies")
        body = resp.json()
        assert body["dependencies"]["llm_provider"]["engine"] == "gemini"
