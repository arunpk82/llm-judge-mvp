"""Mode-aware HMAC key resolution tests (CP-F2 closure).

Exercises :func:`llm_judge.control_plane.envelope._resolve_hmac_key`
across the four mode/key combinations and verifies that
``PlatformRunner.__init__`` invokes :func:`validate_configuration`.
"""

from __future__ import annotations

import pytest

from llm_judge.control_plane import configuration, envelope
from llm_judge.control_plane.runner import PlatformRunner
from llm_judge.control_plane.types import ConfigurationError


@pytest.fixture(autouse=True)
def _reset_configuration_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LLM_JUDGE_MODE", raising=False)
    monkeypatch.delenv("LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", raising=False)
    configuration._reset_for_tests()


def test_resolve_hmac_key_development_mode_default_used() -> None:
    key = envelope._resolve_hmac_key()
    assert key == envelope._DEFAULT_DEV_KEY.encode("utf-8")


def test_resolve_hmac_key_development_mode_env_var_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "dev-override")
    key = envelope._resolve_hmac_key()
    assert key == b"dev-override"


def test_resolve_hmac_key_production_mode_env_var_used(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    monkeypatch.setenv(
        "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "real-production-key"
    )
    configuration.validate_configuration()
    key = envelope._resolve_hmac_key()
    assert key == b"real-production-key"


def test_resolve_hmac_key_production_mode_no_env_var_raises_defense_in_depth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If validate_configuration is bypassed (e.g. an envelope is
    constructed without going through PlatformRunner), production-mode
    signing still refuses to fall back to the development key."""
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    with pytest.raises(ConfigurationError) as excinfo:
        envelope._resolve_hmac_key()
    assert "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY" in str(excinfo.value)
    assert "production" in str(excinfo.value)


def test_platform_runner_init_calls_validate_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-mode-no-key fails during PlatformRunner construction,
    not at first envelope or first request — the runtime gate."""
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    with pytest.raises(ConfigurationError):
        PlatformRunner()


def test_platform_runner_init_succeeds_in_development_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = PlatformRunner()
    assert runner is not None


def test_platform_runner_init_succeeds_in_production_with_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    monkeypatch.setenv(
        "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "real-production-key"
    )
    runner = PlatformRunner()
    assert runner is not None
