"""
EPIC-D1: Environment-Independent Configuration & State — Tests.

Acceptance criteria verified:
  AC-1: config_root() reads LLM_JUDGE_CONFIGS_DIR, falls back to Path('configs')
  AC-2: state_root() reads LLM_JUDGE_DATA_DIR / 'reports', falls back to 'reports'
  AC-3: All existing tests pass with no env vars set (verified by CI)
  AC-4: Tests pass with env vars pointing to temp directories
  AC-5: No hardcoded Path('configs/') or Path('reports/') remain in src/

Note: rules/lifecycle.py retains Path("rules/manifest.yaml") — this is
a project-root file outside configs/ and reports/, explicitly excluded
from D1 scope by the acceptance criteria.
"""
from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

# =====================================================================
# AC-1 & AC-2: Root resolvers respect env vars and fallbacks
# =====================================================================

class TestPathResolvers:
    """Test config_root / data_root / state_root / baselines_root / datasets_root."""

    def test_config_root_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_JUDGE_CONFIGS_DIR", raising=False)
        from llm_judge.paths import config_root
        assert config_root() == Path("configs")

    def test_config_root_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(tmp_path / "custom_cfg"))
        from llm_judge.paths import config_root
        assert config_root() == tmp_path / "custom_cfg"

    def test_data_root_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_JUDGE_DATA_DIR", raising=False)
        from llm_judge.paths import data_root
        assert data_root() == Path(".")

    def test_data_root_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path / "data"))
        from llm_judge.paths import data_root
        assert data_root() == tmp_path / "data"

    def test_state_root_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_JUDGE_DATA_DIR", raising=False)
        from llm_judge.paths import state_root
        assert state_root() == Path("reports")

    def test_state_root_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path))
        from llm_judge.paths import state_root
        assert state_root() == tmp_path / "reports"

    def test_baselines_root_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_JUDGE_DATA_DIR", raising=False)
        from llm_judge.paths import baselines_root
        assert baselines_root() == Path("baselines")

    def test_baselines_root_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path))
        from llm_judge.paths import baselines_root
        assert baselines_root() == tmp_path / "baselines"

    def test_datasets_root_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_JUDGE_DATA_DIR", raising=False)
        from llm_judge.paths import datasets_root
        assert datasets_root() == Path("datasets")

    def test_datasets_root_env_override(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path))
        from llm_judge.paths import datasets_root
        assert datasets_root() == tmp_path / "datasets"


# =====================================================================
# ensure_dir helper
# =====================================================================

class TestEnsureDir:
    def test_creates_nested_dirs(self, tmp_path: Path) -> None:
        from llm_judge.paths import ensure_dir
        target = tmp_path / "a" / "b" / "c"
        assert not target.exists()
        result = ensure_dir(target)
        assert target.is_dir()
        assert result == target

    def test_idempotent_on_existing(self, tmp_path: Path) -> None:
        from llm_judge.paths import ensure_dir
        target = tmp_path / "existing"
        target.mkdir()
        result = ensure_dir(target)
        assert result == target


# =====================================================================
# validate_paths — readiness check for D2
# =====================================================================

class TestValidatePaths:
    def _setup_valid_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Create a valid directory layout and set env vars."""
        cfg = tmp_path / "configs"
        cfg.mkdir()
        data = tmp_path / "data"
        data.mkdir()
        (data / "reports").mkdir()
        (data / "datasets").mkdir()

        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(data))
        return data

    def test_all_checks_pass(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self._setup_valid_env(tmp_path, monkeypatch)
        from llm_judge.paths import validate_paths
        result = validate_paths()

        assert result["ok"] is True
        assert result["config_root"]["ok"] is True
        assert result["state_root"]["ok"] is True
        assert result["state_root"]["writable"] is True
        assert result["datasets_root"]["ok"] is True

    def test_missing_config_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(tmp_path / "nonexistent"))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path))
        (tmp_path / "reports").mkdir()
        (tmp_path / "datasets").mkdir()
        from llm_judge.paths import validate_paths
        result = validate_paths()

        assert result["ok"] is False
        assert result["config_root"]["ok"] is False

    def test_missing_state_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = tmp_path / "configs"
        cfg.mkdir()
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))
        monkeypatch.setenv("LLM_JUDGE_DATA_DIR", str(tmp_path / "nonexistent"))
        from llm_judge.paths import validate_paths
        result = validate_paths()

        assert result["ok"] is False
        assert result["state_root"]["ok"] is False

    def test_baselines_not_required(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Baselines dir missing should NOT cause overall failure."""
        self._setup_valid_env(tmp_path, monkeypatch)
        # baselines dir intentionally not created
        from llm_judge.paths import validate_paths
        result = validate_paths()

        assert result["baselines_root"]["ok"] is True  # optional
        assert result["ok"] is True


# =====================================================================
# AC-4: Modules resolve paths correctly with env override
# =====================================================================

class TestModulePathIntegration:
    """Verify that downstream modules pick up env vars when freshly imported."""

    def test_engine_loads_from_config_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """rules/engine.py load_plan_for_rubric uses config_root()."""
        import yaml
        cfg = tmp_path / "configs"
        (cfg / "rules").mkdir(parents=True)
        (cfg / "rules" / "test_v1.yaml").write_text(
            yaml.dump({"rubric_id": "test", "version": "v1", "rules": []}),
            encoding="utf-8",
        )
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))

        # Force fresh import to pick up env var
        import llm_judge.paths
        importlib.reload(llm_judge.paths)
        import llm_judge.rules.engine as engine_mod
        importlib.reload(engine_mod)

        plan = engine_mod.load_plan_for_rubric("test", "v1")
        assert plan.rubric_id == "test"

    def test_prompts_dir_resolves_via_config_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """calibration/prompts.py PROMPTS_DIR uses config_root()."""
        cfg = tmp_path / "configs"
        monkeypatch.setenv("LLM_JUDGE_CONFIGS_DIR", str(cfg))

        import llm_judge.paths
        importlib.reload(llm_judge.paths)
        import llm_judge.calibration.prompts as prompts_mod
        importlib.reload(prompts_mod)

        assert prompts_mod.PROMPTS_DIR == cfg / "prompts"


# =====================================================================
# AC-5: No hardcoded paths remain in src/
# =====================================================================

class TestNoHardcodedPaths:
    """Grep-based verification that no hardcoded paths remain."""

    REPO_ROOT = Path(__file__).resolve().parent.parent.parent

    def test_no_hardcoded_configs_path(self) -> None:
        result = subprocess.run(
            ["grep", "-rn", 'Path("configs', str(self.REPO_ROOT / "src")],
            capture_output=True, text=True,
        )
        assert result.stdout == "", (
            f"Hardcoded Path('configs/...') found in src/:\n{result.stdout}"
        )

    def test_no_hardcoded_reports_path(self) -> None:
        result = subprocess.run(
            ["grep", "-rn", 'Path("reports', str(self.REPO_ROOT / "src")],
            capture_output=True, text=True,
        )
        assert result.stdout == "", (
            f"Hardcoded Path('reports/...') found in src/:\n{result.stdout}"
        )

    def test_no_hardcoded_baselines_path(self) -> None:
        result = subprocess.run(
            ["grep", "-rn", 'Path("baselines"', str(self.REPO_ROOT / "src")],
            capture_output=True, text=True,
        )
        assert result.stdout == "", (
            f"Hardcoded Path('baselines') found in src/:\n{result.stdout}"
        )

    def test_no_hardcoded_datasets_path(self) -> None:
        result = subprocess.run(
            ["grep", "-rn", 'Path("datasets"', str(self.REPO_ROOT / "src")],
            capture_output=True, text=True,
        )
        # paths.py itself references "datasets" as a fallback string — allow it
        lines = [
            line for line in result.stdout.strip().split("\n")
            if line and "paths.py" not in line
        ]
        assert lines == [], (
            "Hardcoded Path('datasets') found outside paths.py:\n" + "\n".join(lines)
        )
