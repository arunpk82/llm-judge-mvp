"""
Tests for pipeline_config.py — ADR-0026 implementation.

Coverage:
  - Load from valid YAML
  - Defaults when file missing
  - Unknown field rejection (the EPIC 7.4 misspelling failure mode)
  - Invalid values (bad l3_method, non-bool layer, non-numeric threshold)
  - Dependency constraint (l3_method=fact_counting + l3_enabled=false)
  - Singleton behavior (get_pipeline_config caches)
  - Config hash for manifest capture
  - to_manifest_dict produces flat keys
  - Empty YAML file returns defaults
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_judge.calibration.pipeline_config import (
    LayerConfig,
    PipelineConfigError,
    get_pipeline_config,
    load_pipeline_config,
    reset_pipeline_config,
)

# =====================================================================
# Fixtures
# =====================================================================


@pytest.fixture()
def valid_yaml(tmp_path: Path) -> Path:
    """Minimal valid config."""
    p = tmp_path / "pipeline.yaml"
    p.write_text(textwrap.dedent("""\
        layers:
          l1_enabled: true
          l2_enabled: true
          l3_enabled: true
          l4_enabled: false
        l3_method: minicheck_deberta
        thresholds:
          grounding_ratio: 0.80
          min_sentence_sim: 0.30
        sentence_splitter: spacy
        evidence_trail: true
    """))
    return p


@pytest.fixture()
def fact_counting_yaml(tmp_path: Path) -> Path:
    """Config with fact_counting method enabled."""
    p = tmp_path / "pipeline.yaml"
    p.write_text(textwrap.dedent("""\
        layers:
          l1_enabled: true
          l2_enabled: true
          l3_enabled: true
          l4_enabled: false
        l3_method: fact_counting
        thresholds:
          fact_counting_clear: 0.80
        sentence_splitter: spacy
    """))
    return p


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Reset the singleton before and after each test."""
    reset_pipeline_config()
    yield  # type: ignore[misc]
    reset_pipeline_config()


# =====================================================================
# Happy path
# =====================================================================


class TestLoadValid:
    """Loading a well-formed config produces correct PipelineConfig."""

    def test_loads_layer_flags(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.layers.l1_enabled is True
        assert cfg.layers.l2_enabled is True
        assert cfg.layers.l3_enabled is True
        assert cfg.layers.l4_enabled is False

    def test_loads_thresholds(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.thresholds.grounding_ratio == 0.80
        assert cfg.thresholds.min_sentence_sim == 0.30

    def test_loads_l3_method(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.l3_method == "minicheck_deberta"

    def test_loads_splitter(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.sentence_splitter == "spacy"

    def test_loads_evidence_trail(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.evidence_trail is True

    def test_fact_counting_method(self, fact_counting_yaml: Path) -> None:
        cfg = load_pipeline_config(fact_counting_yaml)
        assert cfg.l3_method == "fact_counting"
        assert cfg.thresholds.fact_counting_clear == 0.80

    def test_config_hash_populated(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert len(cfg.config_hash) == 64  # SHA-256 hex

    def test_source_path_populated(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        assert cfg.source_path == str(valid_yaml)


class TestDefaults:
    """Missing file or empty file returns sensible defaults."""

    def test_missing_file_returns_defaults(self, tmp_path: Path) -> None:
        cfg = load_pipeline_config(tmp_path / "nonexistent.yaml")
        assert cfg.layers.l1_enabled is True
        assert cfg.layers.l2_enabled is True
        assert cfg.layers.l3_enabled is True
        assert cfg.layers.l4_enabled is False
        assert cfg.l3_method == "minicheck_deberta"
        assert cfg.sentence_splitter == "spacy"

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.yaml"
        p.write_text("")
        cfg = load_pipeline_config(p)
        assert cfg.layers.l1_enabled is True
        assert cfg.l3_method == "minicheck_deberta"
        assert cfg.config_hash  # Hash of empty string

    def test_defaults_have_empty_config_hash(self, tmp_path: Path) -> None:
        cfg = load_pipeline_config(tmp_path / "missing.yaml")
        assert cfg.config_hash == ""


class TestThresholdDefaults:
    """Omitted thresholds get correct default values."""

    def test_omitted_thresholds_use_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "minimal.yaml"
        p.write_text("layers:\n  l1_enabled: true\n")
        cfg = load_pipeline_config(p)
        assert cfg.thresholds.grounding_ratio == 0.80
        assert cfg.thresholds.min_sentence_sim == 0.30
        assert cfg.thresholds.similarity == 0.60
        assert cfg.thresholds.max_ungrounded_claims == 2
        assert cfg.thresholds.minicheck == 0.50
        assert cfg.thresholds.nli_entailment == 0.70
        assert cfg.thresholds.fact_counting_clear == 0.80


# =====================================================================
# Validation — reject bad configs at startup
# =====================================================================


class TestUnknownFieldRejection:
    """Unknown fields are rejected — catches misspellings (EPIC 7.4 pattern)."""

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("layers:\n  l1_enabled: true\nunkown_key: value\n")
        with pytest.raises(PipelineConfigError, match="unknown keys.*unkown_key"):
            load_pipeline_config(p)

    def test_unknown_layer_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("layers:\n  l1_enaabled: true\n")  # misspelling
        with pytest.raises(PipelineConfigError, match="unknown keys.*l1_enaabled"):
            load_pipeline_config(p)

    def test_unknown_threshold_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("thresholds:\n  grouding_ratio: 0.5\n")  # misspelling
        with pytest.raises(PipelineConfigError, match="unknown keys.*grouding_ratio"):
            load_pipeline_config(p)

    def test_unknown_cache_key(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("graph_cache:\n  enbled: true\n")  # misspelling
        with pytest.raises(PipelineConfigError, match="unknown keys.*enbled"):
            load_pipeline_config(p)


class TestInvalidValues:
    """Invalid value types and out-of-range values are caught."""

    def test_non_bool_layer(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("layers:\n  l1_enabled: yes_please\n")
        with pytest.raises(PipelineConfigError, match="must be boolean"):
            load_pipeline_config(p)

    def test_invalid_l3_method(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("l3_method: deep_learning\n")
        with pytest.raises(PipelineConfigError, match="invalid.*deep_learning"):
            load_pipeline_config(p)

    def test_non_numeric_threshold(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("thresholds:\n  grounding_ratio: high\n")
        with pytest.raises(PipelineConfigError, match="must be numeric"):
            load_pipeline_config(p)

    def test_invalid_splitter(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("sentence_splitter: nltk\n")
        with pytest.raises(PipelineConfigError, match="invalid.*nltk"):
            load_pipeline_config(p)


class TestDependencyConstraints:
    """Cross-field dependency constraints are enforced."""

    def test_fact_counting_requires_l3_enabled(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text(textwrap.dedent("""\
            layers:
              l3_enabled: false
            l3_method: fact_counting
        """))
        with pytest.raises(PipelineConfigError, match="requires.*l3_enabled"):
            load_pipeline_config(p)

    def test_fact_counting_with_l3_enabled_passes(
        self, fact_counting_yaml: Path
    ) -> None:
        cfg = load_pipeline_config(fact_counting_yaml)
        assert cfg.l3_method == "fact_counting"


# =====================================================================
# Frozen (immutable)
# =====================================================================


class TestFrozen:
    """Config is immutable after creation."""

    def test_cannot_modify_layers(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        with pytest.raises(AttributeError):
            cfg.layers = LayerConfig(l1_enabled=False)  # type: ignore[misc]

    def test_cannot_modify_l3_method(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        with pytest.raises(AttributeError):
            cfg.l3_method = "fact_counting"  # type: ignore[misc]


# =====================================================================
# Singleton
# =====================================================================


class TestSingleton:
    """get_pipeline_config() returns the same instance on repeated calls."""

    def test_returns_same_instance(self, valid_yaml: Path) -> None:
        a = get_pipeline_config(valid_yaml)
        b = get_pipeline_config()
        assert a is b

    def test_force_reload_returns_new_instance(self, valid_yaml: Path) -> None:
        a = get_pipeline_config(valid_yaml)
        b = get_pipeline_config(valid_yaml, force_reload=True)
        assert a is not b
        assert a.config_hash == b.config_hash  # same file, same hash

    def test_reset_clears_cache(self, valid_yaml: Path) -> None:
        a = get_pipeline_config(valid_yaml)
        reset_pipeline_config()
        b = get_pipeline_config(valid_yaml)
        assert a is not b


# =====================================================================
# Manifest capture
# =====================================================================


class TestManifestCapture:
    """to_manifest_dict produces flat keys suitable for the diff engine."""

    def test_manifest_dict_has_all_flags(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        m = cfg.to_manifest_dict()
        # All diff-critical keys must be present and flat (not nested)
        assert "l1_enabled" in m
        assert "l2_enabled" in m
        assert "l3_enabled" in m
        assert "l4_enabled" in m
        assert "l3_method" in m
        assert "config_hash" in m
        assert "sentence_splitter" in m
        assert "threshold_grounding_ratio" in m
        assert "threshold_fact_counting_clear" in m
        assert "graph_cache_enabled" in m

    def test_manifest_dict_values_are_not_nested(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        m = cfg.to_manifest_dict()
        for key, val in m.items():
            assert not isinstance(val, dict), f"manifest key '{key}' is nested"
            assert not isinstance(val, list), f"manifest key '{key}' is a list"

    def test_manifest_dict_includes_config_hash(self, valid_yaml: Path) -> None:
        cfg = load_pipeline_config(valid_yaml)
        m = cfg.to_manifest_dict()
        assert m["config_hash"] == cfg.config_hash
        assert len(m["config_hash"]) == 64


# =====================================================================
# Production config file loads without error
# =====================================================================


class TestProductionConfig:
    """The actual production config file passes validation."""

    def test_production_config_loads(self) -> None:
        prod_path = Path("configs/pipeline/hallucination_pipeline.yaml")
        if not prod_path.exists():
            pytest.skip("Production config not found (expected in repo root)")
        cfg = load_pipeline_config(prod_path)
        assert cfg.layers.l1_enabled is True
        assert cfg.l3_method == "minicheck_deberta"
        assert cfg.config_hash  # non-empty
