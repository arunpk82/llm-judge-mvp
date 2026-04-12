"""
Property Configuration Registry (EPIC 7.4).

Loads property_config.yaml and provides the PropertyRegistry for the
integrated evaluation pipeline. Each property has three independent
configuration dimensions: execution (enabled/disabled), gate mode
(informational/auto-gated/human-gated), and visibility (per persona).

The PropertyRegistry is the control plane — it determines what runs,
what gates, and what's visible. Without it, properties are hardcoded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_judge.paths import config_root

logger = logging.getLogger(__name__)

PROPERTY_CONFIG_PATH = config_root() / "properties" / "property_config.yaml"

VALID_GATE_MODES = frozenset({"informational", "auto-gated", "human-gated"})
VALID_CATEGORIES = frozenset(
    {
        "faithfulness",
        "semantic_quality",
        "safety",
        "task_fidelity",
        "robustness",
        "performance",
    }
)
VALID_VISIBILITY = frozenset({"show", "hide"})
VALID_PERSONAS = frozenset(
    {
        "ml_engineer",
        "product_owner",
        "qa_lead",
        "executive",
    }
)


@dataclass(frozen=True)
class PropertyConfig:
    """Configuration for a single evaluation property."""

    name: str
    id: str
    category: str
    enabled: bool
    gate_mode: str
    threshold: float | int | None = None
    visibility: dict[str, str] = field(default_factory=dict)

    def is_visible_to(self, persona: str) -> bool:
        """Check if this property is visible to a given persona."""
        return self.visibility.get(persona, "hide") == "show"


@dataclass
class DetectionCoverage:
    """Platform Detection Coverage Metric."""

    total: int
    enabled: int
    gated: int
    informational: int
    disabled: int

    @property
    def enabled_pct(self) -> float:
        return (self.enabled / self.total * 100) if self.total > 0 else 0.0

    @property
    def gated_pct(self) -> float:
        return (self.gated / self.total * 100) if self.total > 0 else 0.0

    def summary(self) -> str:
        return (
            f"{self.total} properties defined. "
            f"{self.enabled} enabled, {self.gated} gated, "
            f"{self.informational} informational, {self.disabled} disabled. "
            f"Coverage: {self.enabled_pct:.0f}% enabled, {self.gated_pct:.0f}% gated."
        )


class PropertyRegistry:
    """
    Registry of all evaluation properties with their configuration.

    Loaded once at pipeline startup. Provides lookup by name, category,
    and filtering by enabled/gate_mode/visibility.
    """

    def __init__(self, properties: dict[str, PropertyConfig]) -> None:
        self._properties = properties

    @property
    def all_properties(self) -> dict[str, PropertyConfig]:
        return dict(self._properties)

    def get(self, name: str) -> PropertyConfig | None:
        """Get a property by name."""
        return self._properties.get(name)

    def is_enabled(self, name: str) -> bool:
        """Check if a property is enabled."""
        prop = self._properties.get(name)
        return prop.enabled if prop else False

    def get_enabled(self) -> dict[str, PropertyConfig]:
        """Get all enabled properties."""
        return {k: v for k, v in self._properties.items() if v.enabled}

    def get_by_category(self, category: str) -> dict[str, PropertyConfig]:
        """Get all properties in a category."""
        return {k: v for k, v in self._properties.items() if v.category == category}

    def get_enabled_by_category(self, category: str) -> dict[str, PropertyConfig]:
        """Get enabled properties in a category."""
        return {
            k: v
            for k, v in self._properties.items()
            if v.category == category and v.enabled
        }

    def get_gated(self) -> dict[str, PropertyConfig]:
        """Get all auto-gated or human-gated properties."""
        return {
            k: v
            for k, v in self._properties.items()
            if v.enabled and v.gate_mode in ("auto-gated", "human-gated")
        }

    def detection_coverage(self) -> DetectionCoverage:
        """Compute the Platform Detection Coverage Metric."""
        total = len(self._properties)
        enabled = sum(1 for p in self._properties.values() if p.enabled)
        gated = sum(
            1
            for p in self._properties.values()
            if p.enabled and p.gate_mode in ("auto-gated", "human-gated")
        )
        informational = sum(
            1
            for p in self._properties.values()
            if p.enabled and p.gate_mode == "informational"
        )
        disabled = total - enabled
        return DetectionCoverage(
            total=total,
            enabled=enabled,
            gated=gated,
            informational=informational,
            disabled=disabled,
        )

    def filter_for_persona(
        self,
        results: dict[str, Any],
        persona: str,
    ) -> dict[str, Any]:
        """Filter property results for a specific persona's visibility."""
        filtered: dict[str, Any] = {}
        for prop_name, result in results.items():
            prop = self._properties.get(prop_name)
            if prop and prop.is_visible_to(persona):
                filtered[prop_name] = result
        return filtered


def _validate_property(name: str, raw: dict[str, Any]) -> None:
    """Validate a single property configuration. Raises ValueError on invalid."""
    gate_mode = raw.get("gate_mode", "")
    if gate_mode not in VALID_GATE_MODES:
        raise ValueError(
            f"Property '{name}': invalid gate_mode '{gate_mode}'. "
            f"Must be one of {sorted(VALID_GATE_MODES)}"
        )

    category = raw.get("category", "")
    if category not in VALID_CATEGORIES:
        raise ValueError(
            f"Property '{name}': invalid category '{category}'. "
            f"Must be one of {sorted(VALID_CATEGORIES)}"
        )

    visibility = raw.get("visibility", {})
    if not isinstance(visibility, dict):
        raise ValueError(f"Property '{name}': visibility must be a mapping")

    for persona, vis in visibility.items():
        if persona not in VALID_PERSONAS:
            raise ValueError(
                f"Property '{name}': unknown persona '{persona}' in visibility. "
                f"Must be one of {sorted(VALID_PERSONAS)}"
            )
        if vis not in VALID_VISIBILITY:
            raise ValueError(
                f"Property '{name}': invalid visibility '{vis}' for persona '{persona}'. "
                f"Must be 'show' or 'hide'"
            )

    # Reject unknown top-level fields (catches misspellings like "gatemode")
    known_fields = {"id", "category", "enabled", "gate_mode", "threshold", "visibility"}
    unknown = set(raw.keys()) - known_fields
    if unknown:
        raise ValueError(
            f"Property '{name}': unknown fields {sorted(unknown)}. "
            f"Known fields: {sorted(known_fields)}"
        )


def load_property_config(
    path: Path = PROPERTY_CONFIG_PATH,
) -> PropertyRegistry:
    """
    Load property configuration from YAML.

    Validates schema strictly — no silent defaults for mandatory fields.
    Raises FileNotFoundError if config missing.
    Raises ValueError on invalid configuration.
    """
    if not path.exists():
        raise FileNotFoundError(f"Property config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Property config must be a YAML mapping: {path}")

    properties: dict[str, PropertyConfig] = {}

    for name, prop_raw in raw.get("properties", {}).items():
        if not isinstance(prop_raw, dict):
            raise ValueError(f"Property '{name}' must be a mapping")

        _validate_property(name, prop_raw)

        visibility = prop_raw.get("visibility", {})

        properties[str(name)] = PropertyConfig(
            name=str(name),
            id=str(prop_raw.get("id", name)),
            category=str(prop_raw["category"]),
            enabled=bool(prop_raw.get("enabled", False)),
            gate_mode=str(prop_raw["gate_mode"]),
            threshold=prop_raw.get("threshold"),
            visibility={str(k): str(v) for k, v in visibility.items()},
        )

    logger.info(
        "property_config.loaded",
        extra={
            "total": len(properties),
            "enabled": sum(1 for p in properties.values() if p.enabled),
            "path": str(path),
        },
    )

    return PropertyRegistry(properties)
