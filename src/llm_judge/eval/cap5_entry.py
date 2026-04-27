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

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from llm_judge.control_plane.envelope import ProvenanceEnvelope
from llm_judge.control_plane.observability import timed
from llm_judge.eval.event_registry import append_event
from llm_judge.eval.io import write_json
from llm_judge.eval.registry import append_run_registry_entry
from llm_judge.paths import state_root

logger = structlog.get_logger()

SINGLE_EVAL_ROOT_DEFAULT = state_root() / "single_eval"


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


MANIFEST_SCHEMA_VERSION = 2
DEGRADED_DATASET_ID = "unregistered"
DEGRADED_DATASET_HASH = "sha256:unknown"


@timed(
    "cap5_manifest_composition",
    sub_capability_id="manifest_composition",
    capability_id="CAP-5",
)
def _cap5_manifest_composition(
    envelope: ProvenanceEnvelope,
    verdict: dict[str, Any],
    integrity: dict[str, Any],
    manifest_id: str,
) -> dict[str, Any]:
    """CAP-5 Manifest composition: assemble the canonical
    single-eval manifest dict from envelope + verdict + integrity."""
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_type": "single_eval_manifest",
        "manifest_id": manifest_id,
        "created_at_utc": _utc_now_iso(),
        "envelope": envelope.model_dump(mode="json"),
        "verdict": verdict,
        "integrity": integrity,
        "envelope_integrity": [r.model_dump() for r in envelope.integrity],
        "git_sha": envelope.platform_version,
    }


@timed(
    "cap5_persistence",
    sub_capability_id="persistence",
    capability_id="CAP-5",
)
def _cap5_persistence(
    envelope: ProvenanceEnvelope,
    run_dir: Path,
    manifest: dict[str, Any],
) -> None:
    """CAP-5 Persistence: write the manifest.json artifact to disk."""
    del envelope  # only here so @timed can extract request_id
    write_json(run_dir / "manifest.json", manifest)


@timed(
    "cap5_lineage_linking",
    sub_capability_id="lineage_linking",
    capability_id="CAP-5",
)
def _cap5_lineage_linking(
    envelope: ProvenanceEnvelope,
    run_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    integrity: dict[str, Any],
    rubric_id: str,
    rubric_version: str,
    judge_engine: str,
    dataset_id: str,
    dataset_hash: str,
    manifest_id: str,
) -> None:
    """CAP-5 Lineage linking: append the run registry entry and the
    eval_run governance event so this manifest is reachable from
    cross-run governance queries."""
    append_run_registry_entry(
        run_dir=run_dir,
        manifest=manifest,
        metrics=metrics,
        cases_total=1,
        cases_evaluated=1,
        sampled=False,
        dataset_id=dataset_id,
        dataset_version=rubric_version,
        rubric_id=rubric_id,
        judge_engine=os.environ.get("JUDGE_ENGINE", judge_engine),
        dataset_hash=dataset_hash,
    )
    append_event(
        event_type="eval_run",
        source="control_plane/runner.py",
        actor=envelope.caller_id or "internal",
        related_ids={
            "run_id": manifest_id,
            "request_id": envelope.request_id,
            "dataset_id": dataset_id,
            "rubric_id": rubric_id,
        },
        payload={
            "cases_total": 1,
            "cases_evaluated": 1,
            "sampled": False,
            "dataset_hash": dataset_hash,
            "integrity_complete": bool(integrity.get("complete")),
            "missing_capabilities": list(
                integrity.get("missing_capabilities") or []
            ),
            "capability_chain": list(envelope.capability_chain),
        },
    )


def record_evaluation_manifest(
    envelope: ProvenanceEnvelope,
    verdict: dict[str, Any],
    integrity: dict[str, Any],
    *,
    rubric_id: str,
    rubric_version: str,
    judge_engine: str = "control-plane-v1",
    runs_root: Path | None = None,
) -> str:
    """Canonical CAP-5 entry for a single-evaluation manifest.

    CP-1b (horizontal CAP-5): this function writes a manifest regardless
    of whether upstream capabilities succeeded. The envelope may lack
    ``dataset_registry_id`` / ``input_hash`` (CAP-1 failed); sentinel
    values are substituted so governance writes always complete. The
    per-capability integrity record on the envelope explains what
    actually happened — that *is* the audit trail.

    CP-3 sub-capability decomposition: this entry sequences three
    portal sub-capabilities (Manifest composition, Persistence,
    Lineage linking). Envelope reception runs on the wrapper side
    (invoke_cap5); Query interface is read-side and never engages on
    the write path.

    Returns the manifest_id (equal to ``envelope.request_id``). The
    manifest lives at ``<runs_root>/<manifest_id>/manifest.json``;
    governed registry + event records are appended alongside.
    """
    manifest_id = envelope.request_id
    root = (runs_root or SINGLE_EVAL_ROOT_DEFAULT).resolve()
    run_dir = root / manifest_id
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = envelope.dataset_registry_id or DEGRADED_DATASET_ID
    dataset_hash = envelope.input_hash or DEGRADED_DATASET_HASH

    manifest = _cap5_manifest_composition(
        envelope, verdict, integrity, manifest_id
    )
    _cap5_persistence(envelope, run_dir, manifest)

    metrics: dict[str, Any] = {}
    for key in ("risk_score", "grounding_ratio"):
        val = verdict.get(key)
        if isinstance(val, (int, float)):
            metrics[key] = float(val)

    _cap5_lineage_linking(
        envelope,
        run_dir,
        manifest,
        metrics,
        integrity,
        rubric_id,
        rubric_version,
        judge_engine,
        dataset_id,
        dataset_hash,
        manifest_id,
    )

    logger.info(
        "cap5.manifest_recorded",
        manifest_id=manifest_id,
        run_dir=str(run_dir),
    )
    return manifest_id
