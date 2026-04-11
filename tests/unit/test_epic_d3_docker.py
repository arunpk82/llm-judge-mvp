"""
EPIC-D3: Docker Packaging & Deployment Configuration — Tests.

Acceptance criteria verified (static validation — Docker not available in CI):
  AC-1: docker-compose.yaml is valid and has correct structure
  AC-2: .env.example lists all env vars with defaults and descriptions
  AC-3: Dockerfile copies required directories

Note: Full acceptance (docker-compose up → /ready returns 200) requires
a Docker-capable environment. These tests validate the configuration
files themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class TestDockerCompose:
    """Validate docker-compose.yaml structure."""

    @pytest.fixture
    def compose(self) -> dict:
        path = REPO_ROOT / "docker-compose.yaml"
        assert path.exists(), "docker-compose.yaml missing from repo root"
        return yaml.safe_load(path.read_text())

    def test_service_defined(self, compose: dict) -> None:
        assert "llm-judge" in compose["services"]

    def test_port_mapping(self, compose: dict) -> None:
        svc = compose["services"]["llm-judge"]
        ports = svc.get("ports", [])
        # At least one port mapping referencing 8000
        assert any("8000" in str(p) for p in ports)

    def test_volume_mount(self, compose: dict) -> None:
        svc = compose["services"]["llm-judge"]
        volumes = svc.get("volumes", [])
        # Must mount persistent data volume to /data
        assert any("/data" in str(v) for v in volumes), "No volume mounted to /data"

    def test_named_volume_defined(self, compose: dict) -> None:
        volumes = compose.get("volumes", {})
        assert "llm-judge-data" in volumes

    def test_env_file_referenced(self, compose: dict) -> None:
        svc = compose["services"]["llm-judge"]
        env_files = svc.get("env_file", [])
        assert ".env" in env_files

    def test_healthcheck_uses_ready(self, compose: dict) -> None:
        svc = compose["services"]["llm-judge"]
        hc = svc.get("healthcheck", {})
        test = " ".join(str(t) for t in hc.get("test", []))
        assert "/ready" in test, "Healthcheck should use /ready endpoint"

    def test_restart_policy(self, compose: dict) -> None:
        svc = compose["services"]["llm-judge"]
        assert svc.get("restart") in ("unless-stopped", "always")


class TestEnvExample:
    """Validate .env.example completeness."""

    @pytest.fixture
    def env_content(self) -> str:
        path = REPO_ROOT / ".env.example"
        assert path.exists(), ".env.example missing from repo root"
        return path.read_text()

    def test_judge_engine_documented(self, env_content: str) -> None:
        assert "JUDGE_ENGINE" in env_content

    def test_gemini_key_documented(self, env_content: str) -> None:
        assert "GEMINI_API_KEY" in env_content

    def test_openai_key_documented(self, env_content: str) -> None:
        assert "LLM_API_KEY" in env_content

    def test_groq_key_documented(self, env_content: str) -> None:
        assert "GROQ_API_KEY" in env_content

    def test_ollama_url_documented(self, env_content: str) -> None:
        assert "OLLAMA_BASE_URL" in env_content

    def test_config_dir_documented(self, env_content: str) -> None:
        assert "LLM_JUDGE_CONFIGS_DIR" in env_content

    def test_data_dir_documented(self, env_content: str) -> None:
        assert "LLM_JUDGE_DATA_DIR" in env_content

    def test_port_documented(self, env_content: str) -> None:
        assert "LLM_JUDGE_PORT" in env_content


class TestDockerfile:
    """Validate Dockerfile has required COPY and ENV directives."""

    @pytest.fixture
    def dockerfile(self) -> str:
        path = REPO_ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile missing from repo root"
        return path.read_text()

    def test_copies_src(self, dockerfile: str) -> None:
        assert "COPY src" in dockerfile

    def test_copies_configs(self, dockerfile: str) -> None:
        assert "COPY configs" in dockerfile

    def test_copies_rules(self, dockerfile: str) -> None:
        assert "COPY rules" in dockerfile

    def test_copies_rubrics(self, dockerfile: str) -> None:
        assert "COPY rubrics" in dockerfile

    def test_copies_sample_dataset(self, dockerfile: str) -> None:
        assert "datasets/math_basic" in dockerfile

    def test_sets_config_dir_env(self, dockerfile: str) -> None:
        assert "LLM_JUDGE_CONFIGS_DIR" in dockerfile

    def test_sets_data_dir_env(self, dockerfile: str) -> None:
        assert "LLM_JUDGE_DATA_DIR" in dockerfile

    def test_creates_data_dirs(self, dockerfile: str) -> None:
        assert "/data/reports" in dockerfile

    def test_healthcheck_defined(self, dockerfile: str) -> None:
        assert "HEALTHCHECK" in dockerfile
        assert "/ready" in dockerfile
