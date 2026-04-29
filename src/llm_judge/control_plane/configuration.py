"""Startup-time configuration validation for the platform.

Centralised entry point for configuration checks that should fire
**before** any wrapper or capability setup runs. The pattern is
"validate at PlatformRunner construction; fail closed on missing
production-required values; warn loudly on development defaults".

Currently absorbed checks:
  - HMAC key in production mode (closes CP-F2 and CP-F11): when
    ``LLM_JUDGE_MODE=production``, ``LLM_JUDGE_CONTROL_PLANE_HMAC_KEY``
    must be set; otherwise :class:`ConfigurationError` is raised.
  - Layer vocabulary alignment (closes CP-F4): the cascade-layer
    string set declared in
    :data:`llm_judge.control_plane.wrappers.VALID_LAYERS` plus
    :data:`STUB_LAYERS` must equal the ``--isolate-layer`` choice list
    accepted by ``tools/run_batch_evaluation.py``. ``L5`` is in the
    argparse choices and in :data:`STUB_LAYERS` but not yet in
    ``VALID_LAYERS`` — argparse keeps it stable across the
    level-by-level arc, ``invoke_cap7`` rejects it as unwired (see
    ``tools/run_batch_evaluation.py`` ``--isolate-layer`` help text).
    Capturing the gap in :data:`STUB_LAYERS` makes the asymmetry a
    documented architecture decision rather than silent drift.

Expected future extensions (declared here for governance continuity;
not yet implemented):
  - Artifact root validation: writability of ``runs_root`` /
    ``transient_root`` at startup rather than at first persistence.
  - Governance preflight reachability: ``rubrics/registry.yaml`` exists
    and parses before the first request.

Mode handling
~~~~~~~~~~~~~
Modes are read from ``LLM_JUDGE_MODE``. Recognised values are
``"development"`` (the default when the env var is unset or empty) and
``"production"``. Any other non-empty value raises
:class:`ConfigurationError` — fail-closed on typos rather than silently
defaulting.

After :func:`validate_configuration` runs, the resolved mode is cached
at module level so :func:`get_mode` (consumed by
:func:`llm_judge.control_plane.envelope._resolve_hmac_key` for
defense-in-depth) does not need to re-read the environment. Tests that
exercise mode transitions reset the cache via :func:`_reset_for_tests`.
"""

from __future__ import annotations

import os

import structlog

from llm_judge.control_plane.types import ConfigurationError

logger = structlog.get_logger()

_MODE_ENV_VAR = "LLM_JUDGE_MODE"
_HMAC_ENV_VAR = "LLM_JUDGE_CONTROL_PLANE_HMAC_KEY"
_DEFAULT_MODE = "development"
_VALID_MODES = frozenset({"development", "production"})

STUB_LAYERS: frozenset[str] = frozenset({"L5"})
"""Cascade-layer identifiers that are accepted at the argparse boundary
but not yet wired into ``invoke_cap7``. Listed here so layer-vocabulary
alignment treats the gap as documented architecture rather than drift."""

_ARGPARSE_LAYER_CHOICES: frozenset[str] = frozenset({"L1", "L2", "L3", "L4", "L5"})
"""The cascade-layer choices that ``tools/run_batch_evaluation.py``
``--isolate-layer`` accepts. Mirrored here so :func:`validate_layer_vocabulary`
can verify alignment without importing the script. A separate test
(``tests/control_plane/test_configuration.py``) loads the actual parser
and asserts the live choices equal this set, catching drift in the tool
file even when configuration.py is unchanged."""

_resolved_mode: str | None = None


def _read_mode() -> str:
    raw = os.environ.get(_MODE_ENV_VAR)
    if raw is None or not raw.strip():
        return _DEFAULT_MODE
    candidate = raw.strip().lower()
    if candidate not in _VALID_MODES:
        raise ConfigurationError(
            f"unknown {_MODE_ENV_VAR}={raw!r}; "
            f"valid values are {sorted(_VALID_MODES)}"
        )
    return candidate


def validate_layer_vocabulary() -> None:
    """Validate cascade-layer vocabulary alignment at startup.

    The ``--isolate-layer`` argparse choice list in
    ``tools/run_batch_evaluation.py`` MUST equal
    :data:`llm_judge.control_plane.wrappers.VALID_LAYERS` ∪
    :data:`STUB_LAYERS`. The split distinguishes wired layers (rejected
    by ``invoke_cap7`` only when truly unknown) from stub layers
    (rejected by ``invoke_cap7`` as not-yet-implemented). When a stub
    layer becomes wired, the developer moves it from :data:`STUB_LAYERS`
    into ``VALID_LAYERS``; this function verifies the split stays
    coherent so neither set silently drifts.

    Raises:
        ConfigurationError: ``VALID_LAYERS ∪ STUB_LAYERS`` does not
            equal the argparse choice set declared here.
    """
    from llm_judge.control_plane.wrappers import VALID_LAYERS

    expected = VALID_LAYERS | STUB_LAYERS
    if _ARGPARSE_LAYER_CHOICES != expected:
        raise ConfigurationError(
            f"layer vocabulary misalignment: argparse choices "
            f"{sorted(_ARGPARSE_LAYER_CHOICES)} != "
            f"VALID_LAYERS ∪ STUB_LAYERS {sorted(expected)}; "
            f"if wiring a stub, move it from STUB_LAYERS into "
            f"VALID_LAYERS; if adding a layer, update both "
            f"_ARGPARSE_LAYER_CHOICES and the choice list in "
            f"tools/run_batch_evaluation.py"
        )


def validate_configuration() -> None:
    """Validate platform configuration; raise on production gaps.

    Called once near the top of ``PlatformRunner.__init__``. After a
    successful return, :func:`get_mode` reflects the validated mode for
    the lifetime of the process (or until :func:`_reset_for_tests` runs).
    Validators run in sequence: HMAC mode first, then layer vocabulary.

    Raises:
        ConfigurationError: production mode without an HMAC key, an
            unknown ``LLM_JUDGE_MODE`` value, or layer vocabulary
            misalignment (see :func:`validate_layer_vocabulary`).
    """
    global _resolved_mode

    mode = _read_mode()
    hmac_key = os.environ.get(_HMAC_ENV_VAR)
    hmac_set = hmac_key is not None and hmac_key.strip() != ""

    if mode == "production" and not hmac_set:
        raise ConfigurationError(
            f"{_HMAC_ENV_VAR} must be set in production mode "
            f"(LLM_JUDGE_MODE=production); platform startup aborted"
        )

    if mode == "development" and not hmac_set:
        logger.warning(
            "control_plane.config.default_hmac_key_in_use",
            mode=mode,
            env_var=_HMAC_ENV_VAR,
            hint="set env var for non-development use",
        )

    validate_layer_vocabulary()

    _resolved_mode = mode


def get_mode() -> str:
    """Return the validated mode. If :func:`validate_configuration`
    has not yet run (e.g. an envelope is constructed outside the
    standard ``PlatformRunner`` path), read the environment directly
    and validate inline so callers always see a checked value."""
    if _resolved_mode is not None:
        return _resolved_mode
    return _read_mode()


def _reset_for_tests() -> None:
    """Clear the cached mode so each test reads a fresh environment.
    Test-only; not part of the public API."""
    global _resolved_mode
    _resolved_mode = None
