"""
Environment-independent path resolution (EPIC-D1).

Centralises all filesystem path construction so that every config read
and state write flows through two environment variables:

  LLM_JUDGE_CONFIGS_DIR  — root for configuration files
                           (fallback: ``configs``, relative to CWD)
  LLM_JUDGE_DATA_DIR     — root for ALL persistent data
                           (fallback: ``.`` i.e. project root / CWD)

Persistent-data sub-directories (resolved from data_root()):
  state_root()      → data_root() / "reports"
  baselines_root()  → data_root() / "baselines"
  datasets_root()   → data_root() / "datasets"

Design rationale
~~~~~~~~~~~~~~~~
* **Functions, not constants** — env vars are read at call time so that
  ``monkeypatch.setenv`` works in tests without import-order issues.
* **Single data_root()** — one Docker volume mount (``/data``) covers
  reports, baselines, and datasets.
* **Fallbacks match existing repo layout** — when no env vars are set,
  every path resolves identically to the pre-D1 hardcoded values.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# =====================================================================
# Root resolvers
# =====================================================================

def config_root() -> Path:
    """Root directory for configuration files.

    Reads ``LLM_JUDGE_CONFIGS_DIR``; falls back to ``configs``
    (relative to CWD — matches repo layout).
    """
    return Path(os.environ.get("LLM_JUDGE_CONFIGS_DIR", "configs"))


def data_root() -> Path:
    """Root directory for all persistent data.

    Reads ``LLM_JUDGE_DATA_DIR``; falls back to ``.`` (CWD), which
    preserves the existing repo layout where ``reports/``,
    ``baselines/``, ``datasets/`` are project-root siblings.
    """
    return Path(os.environ.get("LLM_JUDGE_DATA_DIR", "."))


# =====================================================================
# Sub-directory resolvers
# =====================================================================

def state_root() -> Path:
    """Root for state/report output: ``data_root() / "reports"``."""
    return data_root() / "reports"


def baselines_root() -> Path:
    """Root for baseline snapshots: ``data_root() / "baselines"``."""
    return data_root() / "baselines"


def datasets_root() -> Path:
    """Root for datasets: ``data_root() / "datasets"``."""
    return data_root() / "datasets"


# =====================================================================
# Helpers
# =====================================================================

def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if needed, return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def validate_paths() -> dict[str, Any]:
    """Validate that required directories exist and are accessible.

    Returns a structured report suitable for the ``/ready`` health
    endpoint (EPIC-D2).  Each check runs independently — early failures
    do **not** short-circuit later checks.
    """
    checks: dict[str, Any] = {}
    all_ok = True

    # --- config root must exist (read-only is fine) ---
    cr = config_root()
    config_ok = cr.is_dir()
    checks["config_root"] = {
        "path": str(cr.resolve()),
        "exists": config_ok,
        "ok": config_ok,
    }
    if not config_ok:
        all_ok = False

    # --- state root must be writable ---
    sr = state_root()
    state_exists = sr.is_dir()
    state_writable = False
    if state_exists:
        try:
            probe = sr / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            state_writable = True
        except OSError:
            pass
    state_ok = state_exists and state_writable
    checks["state_root"] = {
        "path": str(sr.resolve()),
        "exists": state_exists,
        "writable": state_writable,
        "ok": state_ok,
    }
    if not state_ok:
        all_ok = False

    # --- baselines dir (optional at startup — created on first baseline) ---
    br = baselines_root()
    checks["baselines_root"] = {
        "path": str(br.resolve()),
        "exists": br.is_dir(),
        "ok": True,  # not required at startup
    }

    # --- datasets dir should exist ---
    dr = datasets_root()
    datasets_ok = dr.is_dir()
    checks["datasets_root"] = {
        "path": str(dr.resolve()),
        "exists": datasets_ok,
        "ok": datasets_ok,
    }
    if not datasets_ok:
        all_ok = False

    checks["ok"] = all_ok
    return checks
