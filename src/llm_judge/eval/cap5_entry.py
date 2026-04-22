"""CAP-5 Artifact Governance — canonical entry point (CP-1 Turn 2).

This module exposes ONE function, ``record_evaluation_manifest``.
Internally it orchestrates the three governance writes that
``eval/run.py:413-448`` does inline today:

  1. Write ``manifest.json`` to a single-eval run directory.
  2. Append a governed ``RunRegistryEntry`` via
     ``eval.registry.append_run_registry_entry``.
  3. Append an ``eval_run`` ``GovernanceEvent`` via
     ``eval.event_registry.append_event``.

The inline calls in ``eval/run.py`` are NOT migrated to this entry
in this packet (D4). A later CAP-5 completion packet does that
migration; CP-1 only adds the new entry additively.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.control_plane.envelope import ProvenanceEnvelope
from llm_judge.eval.event_registry import append_event
from llm_judge.eval.io import write_json
from llm_judge.eval.registry import append_run_registry_entry
from llm_judge.paths import state_root

logger = logging.getLogger(__name__)

SINGLE_EVAL_ROOT_DEFAULT = state_root() / "single_eval"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def record_evaluation_manifest(
    envelope: ProvenanceEnvelope,
    verdict: dict[str, Any],
    integrity: dict[str, Any],
    *,
    rubric_id: str = "chat_quality",
    rubric_version: str = "v1",
    judge_engine: str = "control-plane-v1",
    runs_root: Path | None = None,
) -> str:
    """Canonical CAP-5 entry for a single-evaluation manifest.

    Returns the manifest_id (equal to ``envelope.request_id``). The
    manifest lives at ``<runs_root>/<manifest_id>/manifest.json``;
    governed registry + event records are appended alongside.
    """
    if not envelope.dataset_registry_id:
        raise ValueError(
            "cap5_entry: envelope.dataset_registry_id is required; "
            "CAP-1 must have stamped the envelope before CAP-5."
        )
    if not envelope.input_hash:
        raise ValueError(
            "cap5_entry: envelope.input_hash is required; CAP-1 must "
            "have stamped the envelope before CAP-5."
        )

    manifest_id = envelope.request_id
    root = (runs_root or SINGLE_EVAL_ROOT_DEFAULT).resolve()
    run_dir = root / manifest_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "artifact_type": "single_eval_manifest",
        "manifest_id": manifest_id,
        "created_at_utc": _utc_now_iso(),
        "envelope": envelope.model_dump(mode="json"),
        "verdict": verdict,
        "integrity": integrity,
        "git_sha": envelope.platform_version,
    }
    write_json(run_dir / "manifest.json", manifest)

    metrics: dict[str, Any] = {}
    for key in ("risk_score", "grounding_ratio"):
        val = verdict.get(key)
        if isinstance(val, (int, float)):
            metrics[key] = float(val)

    append_run_registry_entry(
        run_dir=run_dir,
        manifest=manifest,
        metrics=metrics,
        cases_total=1,
        cases_evaluated=1,
        sampled=False,
        dataset_id=envelope.dataset_registry_id,
        dataset_version=rubric_version,
        rubric_id=rubric_id,
        judge_engine=os.environ.get("JUDGE_ENGINE", judge_engine),
        dataset_hash=envelope.input_hash,
    )

    append_event(
        event_type="eval_run",
        source="control_plane/runner.py",
        actor=envelope.caller_id or "internal",
        related_ids={
            "run_id": manifest_id,
            "request_id": envelope.request_id,
            "dataset_id": envelope.dataset_registry_id,
            "rubric_id": rubric_id,
        },
        payload={
            "cases_total": 1,
            "cases_evaluated": 1,
            "sampled": False,
            "dataset_hash": envelope.input_hash,
            "integrity_complete": bool(integrity.get("complete")),
            "missing_capabilities": list(
                integrity.get("missing_capabilities") or []
            ),
            "capability_chain": list(envelope.capability_chain),
        },
    )

    logger.info(
        "cap5.manifest_recorded",
        extra={"manifest_id": manifest_id, "run_dir": str(run_dir)},
    )
    return manifest_id
