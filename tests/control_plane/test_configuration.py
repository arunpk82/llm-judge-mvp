"""Unit tests for control_plane.configuration startup validation.

Covers CP-F2 (HMAC default-key fallback hardening), CP-F11
(no startup-time configuration validation), and CP-F4 (layer vocabulary
alignment via :func:`validate_layer_vocabulary`) closure mechanisms.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from llm_judge.control_plane import configuration, wrappers
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


# ---------------------------------------------------------------------------
# CP-F4: validate_layer_vocabulary
# ---------------------------------------------------------------------------


def test_validate_layer_vocabulary_aligned_passes() -> None:
    """At master HEAD, ``VALID_LAYERS ∪ STUB_LAYERS`` equals the
    declared argparse choice set; the validator returns silently."""
    configuration.validate_layer_vocabulary()


def test_validate_layer_vocabulary_argparse_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A change to the declared argparse choices that no longer matches
    ``VALID_LAYERS ∪ STUB_LAYERS`` raises ConfigurationError."""
    monkeypatch.setattr(
        configuration,
        "_ARGPARSE_LAYER_CHOICES",
        frozenset({"L1", "L2", "L3", "L4"}),  # dropped L5
    )
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_layer_vocabulary()
    assert "argparse choices" in str(excinfo.value)
    assert "L5" in str(excinfo.value)


def test_validate_layer_vocabulary_valid_layers_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wiring a new layer in ``VALID_LAYERS`` without updating
    ``_ARGPARSE_LAYER_CHOICES`` raises ConfigurationError."""
    monkeypatch.setattr(
        wrappers,
        "VALID_LAYERS",
        frozenset({"L1", "L2", "L3", "L4", "L6"}),  # added L6
    )
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_layer_vocabulary()
    assert "L6" in str(excinfo.value)


def test_validate_layer_vocabulary_stub_layers_drift_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing L5 from ``STUB_LAYERS`` without also removing it from
    the argparse choice set raises ConfigurationError."""
    monkeypatch.setattr(configuration, "STUB_LAYERS", frozenset())
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_layer_vocabulary()
    assert "L5" in str(excinfo.value)


def test_validate_configuration_calls_layer_vocabulary_validator_in_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``validate_configuration`` runs HMAC-mode validation first, then
    layer vocabulary validation. A vocabulary fault surfaces only after
    HMAC has been checked, and the cached mode is NOT set when vocabulary
    validation fails (fail-closed)."""
    monkeypatch.setattr(configuration, "STUB_LAYERS", frozenset())
    with pytest.raises(ConfigurationError):
        configuration.validate_configuration()
    # Mode cache must remain unset because validate_configuration aborted.
    assert configuration._resolved_mode is None


def test_validate_configuration_production_hmac_fault_skips_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If HMAC validation fails in production mode, vocabulary
    validation is never reached; both checks running unconditionally
    would lose the sequencing invariant."""
    monkeypatch.setenv("LLM_JUDGE_MODE", "production")
    # Break vocabulary too — but HMAC failure should fire first.
    monkeypatch.setattr(configuration, "STUB_LAYERS", frozenset())
    with pytest.raises(ConfigurationError) as excinfo:
        configuration.validate_configuration()
    assert "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY" in str(excinfo.value)


def test_argparse_choices_match_declared_set() -> None:
    """The ``--isolate-layer`` choice list in
    ``tools/run_batch_evaluation.py`` MUST equal the set declared in
    :data:`configuration._ARGPARSE_LAYER_CHOICES`. Catches drift in the
    tool file even when configuration.py is unchanged.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "run_batch_evaluation.py"
    spec = importlib.util.spec_from_file_location(
        "_l1pkt2_run_batch_evaluation", tool_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    parser = module._build_argparser()
    isolate_action = next(
        action for action in parser._actions if action.dest == "isolate_layer"
    )
    assert isolate_action.choices is not None
    assert frozenset(isolate_action.choices) == configuration._ARGPARSE_LAYER_CHOICES
