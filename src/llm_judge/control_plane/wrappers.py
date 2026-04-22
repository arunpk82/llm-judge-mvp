"""Capability wrappers: invoke_cap{1,2,5,7}.

Each wrapper performs a loud pre-check on the envelope's upstream
stamps (raises ``MissingProvenanceError`` naming the absent stamp),
calls the capability's real entry, then returns a newly-stamped
envelope plus the capability's output. Wrappers do NOT catch
exceptions from capability code — sibling-failure tolerance is the
Runner's concern.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import yaml

from llm_judge.calibration.hallucination import check_hallucination
from llm_judge.control_plane.envelope import ProvenanceEnvelope
from llm_judge.control_plane.types import (
    MissingProvenanceError,
    SingleEvaluationRequest,
)
from llm_judge.datasets.models import DatasetMetadata
from llm_judge.datasets.registry import DatasetRegistry, ResolvedDataset
from llm_judge.eval.cap5_entry import record_evaluation_manifest
from llm_judge.paths import state_root
from llm_judge.rules.engine import load_plan_for_rubric, run_rules
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)

# v1 simplification: SingleEvaluationRequest does not carry a rubric.
# The Control Plane binds to a documented default; future request
# shapes may specify an explicit rubric_id.
DEFAULT_RUBRIC_ID = "chat_quality"
DEFAULT_RUBRIC_VERSION = "v1"

TRANSIENT_DATASETS_ROOT = state_root() / "transient_datasets"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def invoke_cap1(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    *,
    transient_root: Path | None = None,
) -> tuple[ProvenanceEnvelope, ResolvedDataset]:
    """Construct a one-row transient dataset, pass it through CAP-1's
    real registration path, stamp dataset_registry_id + input_hash.
    """
    root = (transient_root or TRANSIENT_DATASETS_ROOT).resolve()
    dataset_id = f"transient_{uuid.uuid4().hex[:12]}"
    ds_dir = root / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    data_path = ds_dir / "dataset.jsonl"
    row: dict[str, Any] = {
        "case_id": f"{envelope.request_id}_row_0",
        "conversation": [{"role": "user", "content": request.source}],
        "candidate_answer": request.response,
        "rubric_id": DEFAULT_RUBRIC_ID,
    }
    data_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    content_hash = _sha256_file(data_path)

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        version=DEFAULT_RUBRIC_VERSION,
        data_file="dataset.jsonl",
        owner="control-plane",
        task_type="single_eval",
        content_hash=content_hash,
    )
    (ds_dir / "dataset.yaml").write_text(
        yaml.safe_dump(metadata.model_dump(), sort_keys=True),
        encoding="utf-8",
    )

    registry = DatasetRegistry(root_dir=root)
    resolved = registry.resolve(
        dataset_id=dataset_id, version=DEFAULT_RUBRIC_VERSION
    )

    stamped = envelope.stamped(
        capability="CAP-1",
        dataset_registry_id=dataset_id,
        input_hash=content_hash,
    )
    return stamped, resolved


def invoke_cap2(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    dataset_handle: ResolvedDataset,
) -> tuple[ProvenanceEnvelope, list[str]]:
    """Load the rubric's rule plan, run rules against a RuleContext
    built from the request, stamp rule_set_version + rules_fired.
    """
    if not envelope.dataset_registry_id:
        raise MissingProvenanceError(
            "invoke_cap2: envelope.dataset_registry_id is absent; "
            "CAP-1 must run before CAP-2."
        )
    del dataset_handle  # CAP-2 uses the request directly; kept for symmetry

    plan = load_plan_for_rubric(DEFAULT_RUBRIC_ID, DEFAULT_RUBRIC_VERSION)
    req = PredictRequest(
        conversation=[Message(role="user", content=request.source)],
        candidate_answer=request.response,
        rubric_id=DEFAULT_RUBRIC_ID,
    )
    ctx = RuleContext(request=req)

    result = run_rules(ctx, plan)
    flags = list(getattr(result, "flags", []) or [])
    rules_fired: list[str] = []
    for flag in flags:
        fid = getattr(flag, "id", None)
        if isinstance(fid, str) and fid:
            rules_fired.append(fid)

    stamped = envelope.stamped(
        capability="CAP-2",
        rule_set_version=plan.version,
        rules_fired=rules_fired,
    )
    return stamped, rules_fired


def invoke_cap7(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    dataset_handle: ResolvedDataset,
) -> tuple[ProvenanceEnvelope, dict[str, Any]]:
    """Run the hallucination pipeline on (response, source), return a
    serialisable verdict dict. prompt_version stamp deferred (D3)."""
    if not envelope.dataset_registry_id:
        raise MissingProvenanceError(
            "invoke_cap7: envelope.dataset_registry_id is absent; "
            "CAP-1 must run before CAP-7."
        )
    del dataset_handle

    result = check_hallucination(
        response=request.response,
        context=request.source,
        source_context=request.source,
        case_id=envelope.request_id,
        # Keep CP-1 cheap and deterministic: L1 rules only, no embeddings,
        # no model loads, no API calls. A full-layer invocation belongs
        # to a later packet.
        skip_embeddings=True,
        l1_enabled=True,
        l2_enabled=False,
        l3_enabled=False,
        l4_enabled=False,
    )

    verdict: dict[str, Any] = {
        "risk_score": result.risk_score,
        "grounding_ratio": result.grounding_ratio,
        "min_sentence_sim": result.min_sentence_sim,
        "ungrounded_claims": result.ungrounded_claims,
        "unverifiable_citations": result.unverifiable_citations,
        "gate1_decision": result.gate1_decision,
        "gate2_decision": result.gate2_decision,
        "flags": list(result.flags),
        "layer_stats": dict(result.layer_stats),
    }

    stamped = envelope.stamped(capability="CAP-7")
    return stamped, verdict


def invoke_cap5(
    envelope: ProvenanceEnvelope,
    verdict: dict[str, Any],
    integrity: dict[str, Any],
    *,
    runs_root: Path | None = None,
) -> tuple[ProvenanceEnvelope, str]:
    """Write the governed manifest via record_evaluation_manifest, then
    append CAP-5 to the envelope chain."""
    chain = set(envelope.capability_chain)
    if "CAP-1" not in chain:
        raise MissingProvenanceError(
            "invoke_cap5: CAP-1 is absent from capability_chain; "
            "CAP-1 must run before CAP-5."
        )
    if not (chain & {"CAP-2", "CAP-7"}):
        raise MissingProvenanceError(
            "invoke_cap5: neither CAP-2 nor CAP-7 present in "
            "capability_chain; at least one sibling must run."
        )

    manifest_id = record_evaluation_manifest(
        envelope=envelope,
        verdict=verdict,
        integrity=integrity,
        rubric_id=DEFAULT_RUBRIC_ID,
        rubric_version=DEFAULT_RUBRIC_VERSION,
        runs_root=runs_root,
    )
    stamped = envelope.stamped(capability="CAP-5")
    return stamped, manifest_id
