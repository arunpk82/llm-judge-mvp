"""
Hallucination Pipeline Configuration — ADR-0026.

Single source of truth for pipeline behavioral switches.
Loaded once at startup, frozen, config hash captured in manifests.

Pattern: follows ``property_config.py`` — YAML load, strict
validation, frozen dataclass, singleton access via
``get_pipeline_config()``.

Design decisions (ADR-0026):
  - One config file answers "what is the pipeline doing?"
  - Startup validator rejects unknown fields (catches misspellings)
  - Dependency constraints enforced (l2 requires fact-table path)
  - Read-once semantics — no hot reload, change requires restart
  - Config hash available for manifest capture

See also: ADR-0006 (config-driven pipeline), ADR-0027 (L3 method),
          ADR-0028 (CI tests flagged-off paths).
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_judge.paths import config_root

logger = logging.getLogger(__name__)

# Default config path: configs/pipeline/hallucination_pipeline.yaml
_DEFAULT_CONFIG_PATH = config_root() / "pipeline" / "hallucination_pipeline.yaml"

# =====================================================================
# Valid values — used by the startup validator
# =====================================================================

VALID_L3_METHODS = frozenset({"minicheck_deberta", "fact_counting"})

# Known top-level sections in the YAML
_VALID_TOP_LEVEL_KEYS = frozenset({
    "layers",
    "thresholds",
    "l3_method",
    "graph_cache",
    "sentence_splitter",
    "evidence_trail",
})

# Known keys within the 'layers' section
_VALID_LAYER_KEYS = frozenset({
    "l1_enabled",
    "l2_enabled",
    "l3_enabled",
    "l4_enabled",
})

# Known keys within the 'thresholds' section
_VALID_THRESHOLD_KEYS = frozenset({
    "grounding_ratio",
    "min_sentence_sim",
    "similarity",
    "max_ungrounded_claims",
    "minicheck",
    "nli_entailment",
    "fact_counting_clear",
})

# Known keys within 'graph_cache'
_VALID_CACHE_KEYS = frozenset({
    "enabled",
    "ttl_hours",
    "directory",
    "pre_seed_path",
})

VALID_SPLITTERS = frozenset({"spacy", "regex"})


# =====================================================================
# Config dataclasses — frozen (immutable after load)
# =====================================================================


@dataclass(frozen=True)
class LayerConfig:
    """Enable/disable flags for each pipeline layer."""

    l1_enabled: bool = True
    l2_enabled: bool = True
    l3_enabled: bool = True
    l4_enabled: bool = False  # Not yet validated at sentence level


@dataclass(frozen=True)
class ThresholdConfig:
    """Numeric thresholds for pipeline decisions."""

    grounding_ratio: float = 0.80
    min_sentence_sim: float = 0.30
    similarity: float = 0.60
    max_ungrounded_claims: int = 2
    minicheck: float = 0.50
    nli_entailment: float = 0.70
    fact_counting_clear: float = 0.80  # Exp 43: ratio >= 0.8 = auto-clear


@dataclass(frozen=True)
class GraphCacheConfig:
    """Graph cache settings for L2 knowledge graph ensemble."""

    enabled: bool = True
    ttl_hours: int = 168  # 7 days
    directory: str = ".cache/hallucination_graphs"
    pre_seed_path: str = ""  # Path to Exp 31 fact tables for pre-seeding


@dataclass(frozen=True)
class PipelineConfig:
    """
    Complete pipeline configuration — the behavioral contract.

    Frozen after creation. Config hash available for manifest capture.
    Access via ``get_pipeline_config()`` singleton.
    """

    layers: LayerConfig = LayerConfig()
    thresholds: ThresholdConfig = ThresholdConfig()
    l3_method: str = "minicheck_deberta"  # ADR-0027: default until fact-counting ported
    graph_cache: GraphCacheConfig = GraphCacheConfig()
    sentence_splitter: str = "spacy"  # ADR-0014
    evidence_trail: bool = True
    _config_hash: str = ""
    _source_path: str = ""

    @property
    def config_hash(self) -> str:
        """SHA-256 of the source config file. Empty if built from defaults."""
        return self._config_hash

    @property
    def source_path(self) -> str:
        """Path to the config file this was loaded from."""
        return self._source_path

    def to_manifest_dict(self) -> dict[str, Any]:
        """Flat dict for manifest.json capture — every flag visible."""
        return {
            "config_hash": self._config_hash,
            "config_source": self._source_path,
            "l1_enabled": self.layers.l1_enabled,
            "l2_enabled": self.layers.l2_enabled,
            "l3_enabled": self.layers.l3_enabled,
            "l4_enabled": self.layers.l4_enabled,
            "l3_method": self.l3_method,
            "sentence_splitter": self.sentence_splitter,
            "evidence_trail": self.evidence_trail,
            "threshold_grounding_ratio": self.thresholds.grounding_ratio,
            "threshold_min_sentence_sim": self.thresholds.min_sentence_sim,
            "threshold_similarity": self.thresholds.similarity,
            "threshold_max_ungrounded_claims": self.thresholds.max_ungrounded_claims,
            "threshold_fact_counting_clear": self.thresholds.fact_counting_clear,
            "graph_cache_enabled": self.graph_cache.enabled,
            "graph_cache_ttl_hours": self.graph_cache.ttl_hours,
        }


# =====================================================================
# Validation — strict, fail-fast at startup
# =====================================================================


class PipelineConfigError(ValueError):
    """Raised when pipeline config is invalid."""


def _reject_unknown_keys(
    section_name: str,
    raw: dict[str, Any],
    valid: frozenset[str],
) -> None:
    """Reject unknown keys in a config section. Catches misspellings."""
    unknown = set(raw.keys()) - valid
    if unknown:
        raise PipelineConfigError(
            f"[{section_name}] unknown keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(valid)}"
        )


def _validate_raw(raw: dict[str, Any]) -> None:
    """Validate the raw YAML dict. Raises PipelineConfigError on problems."""
    if not isinstance(raw, dict):
        raise PipelineConfigError(
            "Pipeline config must be a YAML mapping, got "
            f"{type(raw).__name__}"
        )

    # Top-level keys
    _reject_unknown_keys("top-level", raw, _VALID_TOP_LEVEL_KEYS)

    # Layers section
    layers_raw = raw.get("layers", {})
    if layers_raw and isinstance(layers_raw, dict):
        _reject_unknown_keys("layers", layers_raw, _VALID_LAYER_KEYS)
        for key, val in layers_raw.items():
            if not isinstance(val, bool):
                raise PipelineConfigError(
                    f"[layers.{key}] must be boolean, got {type(val).__name__}: {val}"
                )

    # Thresholds section
    thresholds_raw = raw.get("thresholds", {})
    if thresholds_raw and isinstance(thresholds_raw, dict):
        _reject_unknown_keys("thresholds", thresholds_raw, _VALID_THRESHOLD_KEYS)
        for key, val in thresholds_raw.items():
            if not isinstance(val, (int, float)):
                raise PipelineConfigError(
                    f"[thresholds.{key}] must be numeric, got {type(val).__name__}: {val}"
                )

    # L3 method
    l3_method = raw.get("l3_method")
    if l3_method is not None and l3_method not in VALID_L3_METHODS:
        raise PipelineConfigError(
            f"[l3_method] invalid: '{l3_method}'. "
            f"Must be one of {sorted(VALID_L3_METHODS)}"
        )

    # Graph cache section
    cache_raw = raw.get("graph_cache", {})
    if cache_raw and isinstance(cache_raw, dict):
        _reject_unknown_keys("graph_cache", cache_raw, _VALID_CACHE_KEYS)

    # Sentence splitter
    splitter = raw.get("sentence_splitter")
    if splitter is not None and splitter not in VALID_SPLITTERS:
        raise PipelineConfigError(
            f"[sentence_splitter] invalid: '{splitter}'. "
            f"Must be one of {sorted(VALID_SPLITTERS)}"
        )

    # Dependency constraint: l3_method=fact_counting requires l3_enabled
    layers = raw.get("layers", {})
    if (
        raw.get("l3_method") == "fact_counting"
        and isinstance(layers, dict)
        and layers.get("l3_enabled") is False
    ):
        raise PipelineConfigError(
            "[dependency] l3_method='fact_counting' requires layers.l3_enabled=true"
        )


# =====================================================================
# Loader — YAML file → validated PipelineConfig
# =====================================================================


def load_pipeline_config(
    path: Path | None = None,
) -> PipelineConfig:
    """
    Load and validate pipeline config from YAML.

    If *path* is None, uses the default location under config_root().
    If the file doesn't exist, returns defaults (all layers enabled,
    minicheck_deberta method, spaCy splitter).

    Raises PipelineConfigError on invalid configuration.
    """
    resolve_path = path or _DEFAULT_CONFIG_PATH

    if not resolve_path.exists():
        logger.info(
            "pipeline_config.defaults",
            extra={"reason": "file_not_found", "path": str(resolve_path)},
        )
        return PipelineConfig()

    raw_text = resolve_path.read_text(encoding="utf-8")
    config_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    raw = yaml.safe_load(raw_text)
    if raw is None:
        # Empty file — use defaults
        return PipelineConfig(
            _config_hash=config_hash, _source_path=str(resolve_path)
        )

    _validate_raw(raw)

    # Build sub-configs
    layers_raw = raw.get("layers", {}) or {}
    layers = LayerConfig(
        l1_enabled=layers_raw.get("l1_enabled", True),
        l2_enabled=layers_raw.get("l2_enabled", True),
        l3_enabled=layers_raw.get("l3_enabled", True),
        l4_enabled=layers_raw.get("l4_enabled", False),
    )

    thresholds_raw = raw.get("thresholds", {}) or {}
    thresholds = ThresholdConfig(
        grounding_ratio=float(thresholds_raw.get("grounding_ratio", 0.80)),
        min_sentence_sim=float(thresholds_raw.get("min_sentence_sim", 0.30)),
        similarity=float(thresholds_raw.get("similarity", 0.60)),
        max_ungrounded_claims=int(thresholds_raw.get("max_ungrounded_claims", 2)),
        minicheck=float(thresholds_raw.get("minicheck", 0.50)),
        nli_entailment=float(thresholds_raw.get("nli_entailment", 0.70)),
        fact_counting_clear=float(thresholds_raw.get("fact_counting_clear", 0.80)),
    )

    cache_raw = raw.get("graph_cache", {}) or {}
    graph_cache = GraphCacheConfig(
        enabled=bool(cache_raw.get("enabled", True)),
        ttl_hours=int(cache_raw.get("ttl_hours", 168)),
        directory=str(cache_raw.get("directory", ".cache/hallucination_graphs")),
        pre_seed_path=str(cache_raw.get("pre_seed_path", "")),
    )

    config = PipelineConfig(
        layers=layers,
        thresholds=thresholds,
        l3_method=str(raw.get("l3_method", "minicheck_deberta")),
        graph_cache=graph_cache,
        sentence_splitter=str(raw.get("sentence_splitter", "spacy")),
        evidence_trail=bool(raw.get("evidence_trail", True)),
        _config_hash=config_hash,
        _source_path=str(resolve_path),
    )

    logger.info(
        "pipeline_config.loaded",
        extra={
            "path": str(resolve_path),
            "hash": config_hash[:16],
            "l1": layers.l1_enabled,
            "l2": layers.l2_enabled,
            "l3": layers.l3_enabled,
            "l4": layers.l4_enabled,
            "l3_method": config.l3_method,
            "splitter": config.sentence_splitter,
        },
    )

    return config


# =====================================================================
# Singleton — load once, reuse everywhere
# =====================================================================

_cached_config: PipelineConfig | None = None


def get_pipeline_config(
    path: Path | None = None,
    *,
    force_reload: bool = False,
) -> PipelineConfig:
    """
    Get the pipeline config singleton.

    First call loads and validates from YAML (or defaults).
    Subsequent calls return the cached instance.

    Args:
        path: Override config file path (first call only).
        force_reload: Force reload from disk (testing only).
    """
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = load_pipeline_config(path)
    return _cached_config


def reset_pipeline_config() -> None:
    """Reset the cached config. For testing only."""
    global _cached_config
    _cached_config = None
