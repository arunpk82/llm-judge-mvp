"""Metrics schema governance — enforces rubric-declared schemas.

Hoisted from ``eval/run.py`` in CP-1c-b.2. Centralizing this lets
both the rigor path (eval/run.py) and any future caller share one
enforcement surface, and forecloses the prior early-return opt-out
that allowed schema drift to creep back in via newly added rubrics.

Architectural distinction — rigor path vs demonstration path
-------------------------------------------------------------

There are two evaluation paths in this repo:

* **Rigor path** — ``eval/run.py`` runs reproducible benchmark
  evaluations. ``compute_metrics`` produces the full metrics
  surface (``f1_fail``, ``cohen_kappa``, ``accuracy``, etc.) that
  rubrics declare in ``rubrics/registry.yaml``. This is the path
  enforced here.

* **Demonstration path** — Control Plane CAP-5
  (``record_evaluation_manifest``) writes manifests with only
  ``risk_score`` and ``grounding_ratio`` (degenerate metrics
  copied from CAP-7's verdict). ``chat_quality@v1``'s declared
  ``f1_fail``/``cohen_kappa`` are NOT computed on this path.

This split is intentional. Computing the full metrics surface in
CAP-5 would require running an offline benchmark for every
single-case demonstration request, which is not the point of the
demonstration path. Closing this gap (computing the missing
metrics in CAP-5) is a future packet.

Schema enforcement applies only to the rigor path. Anyone reading
a Control Plane manifest must understand its metrics surface is
intentionally smaller.
"""

from __future__ import annotations

from typing import Any

from llm_judge.rubric_store import get_rubric


class MetricsSchemaViolationError(Exception):
    """Raised when computed metrics miss a required schema field."""


def enforce_metrics_schema(
    *,
    rubric_ref: str,
    metrics: dict[str, Any],
) -> None:
    """Enforce that computed metrics include every required field.

    No early return — mandatory enforcement. A rubric without any
    declared required metrics is treated as a governance failure
    upstream (see ``check_rubrics_governed`` in
    ``rubrics/lifecycle.py``); this function trusts that contract
    and raises here if required keys are missing from ``metrics``.

    Raises :class:`MetricsSchemaViolationError` when one or more
    required keys are absent from ``metrics``. The message names
    the missing keys and the rubric reference so the operator can
    correct the offending run without grepping.
    """
    rubric = get_rubric(rubric_ref)
    required = list(rubric.metrics_required)

    missing = [k for k in required if k not in metrics]
    if missing:
        raise MetricsSchemaViolationError(
            f"Metrics schema violation for rubric "
            f"{rubric.rubric_id}@{rubric.version}: missing "
            f"required keys {missing}. Present keys="
            f"{sorted(metrics.keys())}"
        )
