"""Unit tests for control_plane.configuration startup validation.

Covers CP-F2 (HMAC default-key fallback hardening) and CP-F11
(no startup-time configuration validation) closure mechanisms.
"""

from __future__ import annotations

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane import configuration
from llm_judge.control_plane.types import ConfigurationError


@pytest.fixture(autouse=True)
def _reset_configuration_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each test starts with a clean environment + cleared mode cache."""
    monkeypatch.delenv("LLM_JUDGE_MODE", raising=False)
    monkeypatch.delenv("LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", raising=False)
    configuration._reset_for_tests()


def test_validate_configuration_production_mode_no_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_configuration()
    assert "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY" in str(excinfo.value)
    assert "production" in str(excinfo.value)


def test_validate_configuration_production_mode_empty_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    monkeypatch.setenv("LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "   ")
    with pytest.raises(ConfigurationError):
        configuration.validate_configuration()


def test_validate_configuration_production_mode_key_set_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    monkeypatch.setenv(
        "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "real-production-key"
    )
    configuration.validate_configuration()
    assert configuration.get_mode() == "production"


def test_validate_configuration_development_mode_no_key_warns_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "development")
    with capture_logs() as logs:
        configuration.validate_configuration()
    assert any(
        entry.get("event") == "control_plane.config.default_hmac_key_in_use"
        and entry.get("log_level") == "warning"
        for entry in logs
    ), f"expected default_hmac_key_in_use warning; got {logs!r}"


def test_validate_configuration_development_mode_default_passes() -> None:
    configuration.validate_configuration()
    assert configuration.get_mode() == "development"


def test_validate_configuration_development_mode_key_set_no_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "dev-override")
    with capture_logs() as logs:
        configuration.validate_configuration()
    assert not any(
        entry.get("event") == "control_plane.config.default_hmac_key_in_use"
        for entry in logs
    )


def test_validate_configuration_unknown_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "staging")
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_configuration()
    assert "staging" in str(excinfo.value)
    assert "LLM_JUDGE_MODE" in str(excinfo.value)


def test_get_mode_returns_validated_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    monkeypatch.setenv(
        "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY", "real-production-key"
    )
    configuration.validate_configuration()
    assert configuration.get_mode() == "production"


def test_get_mode_before_validation_reads_env_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defense-in-depth: callers reaching get_mode() before
    validate_configuration() ran (e.g. envelope construction outside
    the standard PlatformRunner path) still get a checked mode."""
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    assert configuration.get_mode() == "production"


def test_get_mode_before_validation_unknown_mode_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_JUDGE_MODE", "staging")
    with pytest.raises(ConfigurationError):
        configuration.get_mode()
