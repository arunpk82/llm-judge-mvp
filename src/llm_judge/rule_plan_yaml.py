"""Pydantic models for rule-plan and rubric-registry YAML files.

Scope of each model:
  - ``RulePlanRule``, ``RulePlanConfig``: the rule-plan YAML at
    ``configs/rules/<rubric_id>/<version>.yaml``. Carries the list
    of rules the engine executes for that rubric. Matches the
    actual on-disk schema (``id`` + ``enabled`` + ``params``); the
    previous ``RuleConfig`` / ``RubricConfig`` pair required a
    ``severity`` field that no real rule plan carries — deleted in
    CP-1d Commit 2b.
  - ``RubricRegistryConfig``: the index at ``rubrics/registry.yaml``.
    Carries the ``latest:`` version pointer map plus the optional
    ``rubrics:`` block that declares per-version metrics-schema
    contracts.

The governance-field schema for a rubric *definition* file
(``rubrics/<rubric_id>/<version>.yaml``) lives on
``llm_judge.rubrics.lifecycle.RubricLifecycleEntry`` so that
lifecycle logic and rubric parsing share one source of truth.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class RulePlanRule(BaseModel):
    """One rule entry inside a rule-plan YAML.

    Matches the actual on-disk shape observed across
    ``configs/rules/chat_quality/v1.yaml`` and
    ``configs/rules/math_basic/v1.yaml``: every rule has ``id``,
    an ``enabled`` flag (defaulting True), and an optional
    ``params`` dict.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)


class RulePlanConfig(BaseModel):
    """Rule-plan YAML at ``configs/rules/<rubric_id>/<version>.yaml``."""

    model_config = ConfigDict(extra="forbid")

    rubric_id: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    rules: list[RulePlanRule] = Field(default_factory=list)


class RubricMetricsSchema(BaseModel):
    """Per-version metrics-schema declaration from the registry."""

    model_config = ConfigDict(extra="forbid")

    required: list[str] = Field(default_factory=list)
    optional: list[str] = Field(default_factory=list)


class RubricVersionEntry(BaseModel):
    """One entry under ``rubrics.<rubric_id>.versions.<version>``."""

    model_config = ConfigDict(extra="allow")

    metrics_schema: Optional[RubricMetricsSchema] = None


class RubricRegistrySection(BaseModel):
    """One entry under ``rubrics.<rubric_id>``."""

    model_config = ConfigDict(extra="allow")

    versions: dict[str, RubricVersionEntry] = Field(default_factory=dict)


class RubricRegistryConfig(BaseModel):
    """Schema for the rubric registry/index at
    ``rubrics/registry.yaml``.

    Example:

      latest:
        chat_quality: v1
        math_basic: v1
      rubrics:
        chat_quality:
          versions:
            v1:
              metrics_schema:
                required: [f1_fail, cohen_kappa]
                optional: []
    """

    model_config = ConfigDict(extra="allow")

    latest: dict[str, str] = Field(default_factory=dict)
    rubrics: Optional[dict[str, RubricRegistrySection]] = None

    def metrics_schema_for(
        self, rubric_id: str, version: str
    ) -> Optional[RubricMetricsSchema]:
        """Return the metrics schema for a specific rubric version, or
        ``None`` if the registry declares no schema for it."""
        if self.rubrics is None:
            return None
        section = self.rubrics.get(rubric_id)
        if section is None:
            return None
        entry = section.versions.get(version)
        if entry is None:
            return None
        return entry.metrics_schema


# Permissive passthrough for the governance-validated rubric
# definition file. The authoritative Pydantic shape for governance
# fields is RubricLifecycleEntry in llm_judge.rubrics.lifecycle.
# Declared here for completeness so rubric_yaml can be a single
# reference for "every YAML shape in the rubric stack."
def rubric_definition_model() -> type[BaseModel]:
    """Late-import shim to avoid a circular import at module load.

    ``rubric_store`` calls this to get the Pydantic class that
    validates a rubric definition file's governance fields.
    """
    from llm_judge.rubrics.lifecycle import RubricLifecycleEntry

    return RubricLifecycleEntry


# Re-export for type-hint convenience (structural only; the actual
# class object is resolved lazily by rubric_definition_model()).
RubricDefinition: Any = rubric_definition_model
