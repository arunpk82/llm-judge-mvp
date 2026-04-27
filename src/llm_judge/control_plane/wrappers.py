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
import uuid
from pathlib import Path
from typing import Any

import structlog
import yaml

from llm_judge.calibration.hallucination import check_hallucination
from llm_judge.control_plane.envelope import ProvenanceEnvelope
from llm_judge.control_plane.observability import (
    emit_event,
    emit_sub_capability_skipped,
    timed,
)
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

logger = structlog.get_logger()

TRANSIENT_DATASETS_ROOT = state_root() / "transient_datasets"


def _resolve_effective_version(rubric_id: str, rubric_version: str) -> str:
    """Resolve the ``"latest"`` sentinel against the rubric registry.

    Pass-through for explicit versions like ``"v1"``. CP-1c-b.1 only
    plumbs the binding; full governance preflight is CP-1c-b.2 scope.
    """
    if rubric_version != "latest":
        return rubric_version
    from llm_judge.rubric_store import _resolve_version

    return _resolve_version(rubric_id)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


@timed("cap1_reception", sub_capability_id="reception", capability_id="CAP-1")
def _cap1_reception(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    root: Path,
) -> tuple[str, Path, Path]:
    """CAP-1 Reception: allocate transient dataset_id, create the run
    directory, and write the one-row JSONL payload that downstream
    sub-capabilities operate on. Returns (dataset_id, ds_dir, data_path)."""
    dataset_id = f"transient_{uuid.uuid4().hex[:12]}"
    ds_dir = root / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)

    data_path = ds_dir / "dataset.jsonl"
    row: dict[str, Any] = {
        "case_id": f"{envelope.request_id}_row_0",
        "conversation": [{"role": "user", "content": request.source}],
        "candidate_answer": request.response,
        "rubric_id": request.rubric_id,
    }
    data_path.write_text(
        json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return dataset_id, ds_dir, data_path


@timed("cap1_hashing", sub_capability_id="hashing", capability_id="CAP-1")
def _cap1_hashing(envelope: ProvenanceEnvelope, data_path: Path) -> str:
    """CAP-1 Hashing: compute the content-addressed digest of the
    transient dataset file. Returns the sha256 string."""
    del envelope  # only here so @timed can extract request_id
    return _sha256_file(data_path)


@timed("cap1_validation", sub_capability_id="validation", capability_id="CAP-1")
def _cap1_validation(
    envelope: ProvenanceEnvelope,
    dataset_id: str,
    ds_dir: Path,
    content_hash: str,
) -> DatasetMetadata:
    """CAP-1 Validation: build + persist a DatasetMetadata record
    for the transient dataset. Returns the validated metadata."""
    del envelope  # only here so @timed can extract request_id
    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        version="v1",
        data_file="dataset.jsonl",
        owner="control-plane",
        task_type="single_eval",
        content_hash=content_hash,
    )
    (ds_dir / "dataset.yaml").write_text(
        yaml.safe_dump(metadata.model_dump(), sort_keys=True),
        encoding="utf-8",
    )
    return metadata


@timed("cap1_registration", sub_capability_id="registration", capability_id="CAP-1")
def _cap1_registration(
    envelope: ProvenanceEnvelope,
    dataset_id: str,
    root: Path,
) -> ResolvedDataset:
    """CAP-1 Registration: register the transient dataset with the
    DatasetRegistry and resolve the handle for downstream callers."""
    del envelope  # only here so @timed can extract request_id
    registry = DatasetRegistry(root_dir=root)
    return registry.resolve(
        dataset_id=dataset_id, version="v1"
    )


@timed(
    "cap1_lineage_tracking",
    sub_capability_id="lineage_tracking",
    capability_id="CAP-1",
)
def _cap1_lineage_tracking(
    envelope: ProvenanceEnvelope,
    dataset_id: str,
    content_hash: str,
) -> ProvenanceEnvelope:
    """CAP-1 Lineage tracking: stamp the envelope with
    dataset_registry_id + input_hash so downstream caps can verify
    upstream provenance."""
    return envelope.stamped(
        capability="CAP-1",
        dataset_registry_id=dataset_id,
        input_hash=content_hash,
    )


def invoke_cap1(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    *,
    transient_root: Path | None = None,
) -> tuple[ProvenanceEnvelope, ResolvedDataset]:
    """Construct a one-row transient dataset, pass it through CAP-1's
    real registration path, stamp dataset_registry_id + input_hash.

    Decomposed into 5 instrumented sub-capabilities (Reception,
    Hashing, Validation, Registration, Lineage tracking). The 6th
    portal sub-capability — Discovery — is not engaged in the
    single-eval flow and is reported via ``sub_capability_skipped``
    so batch aggregation can render an honest 0/N engagement count.
    Behaviour and outputs are identical to the pre-instrumentation
    implementation.
    """
    root = (transient_root or TRANSIENT_DATASETS_ROOT).resolve()
    dataset_id, ds_dir, data_path = _cap1_reception(envelope, request, root)
    content_hash = _cap1_hashing(envelope, data_path)
    _cap1_validation(envelope, dataset_id, ds_dir, content_hash)
    resolved = _cap1_registration(envelope, dataset_id, root)
    stamped = _cap1_lineage_tracking(envelope, dataset_id, content_hash)

    # Discovery is registry-side lookup the platform exposes for ad-hoc
    # consumers; the single-eval Runner constructs a transient dataset
    # rather than querying the registry, so this sub-cap never engages.
    emit_sub_capability_skipped(
        capability_id="CAP-1",
        sub_capability_id="discovery",
        request_id=envelope.request_id,
        reason="single_eval_does_not_query_registry",
    )

    return stamped, resolved


@timed("cap2_rule_loading", sub_capability_id="rule_loading", capability_id="CAP-2")
def _cap2_rule_loading(
    envelope: ProvenanceEnvelope,
    rubric_id: str,
    rubric_version: str,
) -> Any:
    """CAP-2 Rule loading: resolve the rubric's rule plan."""
    del envelope  # only here so @timed can extract request_id
    return load_plan_for_rubric(rubric_id, rubric_version)


@timed(
    "cap2_input_matching",
    sub_capability_id="input_matching",
    capability_id="CAP-2",
)
def _cap2_input_matching(
    envelope: ProvenanceEnvelope,
    plan: Any,
    request: SingleEvaluationRequest,
) -> Any:
    """CAP-2 Input matching: run the rule plan against a RuleContext
    built from the evaluation request. Returns the rule-engine result."""
    del envelope  # only here so @timed can extract request_id
    req = PredictRequest(
        conversation=[Message(role="user", content=request.source)],
        candidate_answer=request.response,
        rubric_id=request.rubric_id,
    )
    ctx = RuleContext(request=req)
    return run_rules(ctx, plan)


@timed(
    "cap2_evidence_capture",
    sub_capability_id="evidence_capture",
    capability_id="CAP-2",
)
def _cap2_evidence_capture(
    envelope: ProvenanceEnvelope, result: Any
) -> list[str]:
    """CAP-2 Evidence capture: extract rule-id evidence from the
    engine result. Returns the list of fired rule ids."""
    del envelope  # only here so @timed can extract request_id
    flags = list(getattr(result, "flags", []) or [])
    rules_fired: list[str] = []
    for flag in flags:
        fid = getattr(flag, "id", None)
        if isinstance(fid, str) and fid:
            rules_fired.append(fid)
    return rules_fired


@timed(
    "cap2_result_emission",
    sub_capability_id="result_emission",
    capability_id="CAP-2",
)
def _cap2_result_emission(
    envelope: ProvenanceEnvelope,
    plan_version: str,
    rules_fired: list[str],
) -> ProvenanceEnvelope:
    """CAP-2 Result emission: stamp the envelope with rule_set_version
    and rules_fired so downstream caps observe the rule outcomes."""
    return envelope.stamped(
        capability="CAP-2",
        rule_set_version=plan_version,
        rules_fired=rules_fired,
    )


def _emit_pattern_compilation_soft(envelope: ProvenanceEnvelope) -> None:
    """Emit fire-rate signals for the SOFT pattern_compilation sub-cap.

    Pattern compilation happens inside load_plan_for_rubric
    (rules/engine.py); it cannot be timed separately without
    refactoring the engine, which is out of scope for CP-3 (D1: do
    not expand scope). We emit started+completed events with
    ``duration_ms=0.0`` to keep the fire-rate aggregation honest:
    the boundary exists logically, even though this layer cannot
    observe its duration.
    """
    emit_event(
        "sub_capability_started",
        capability_id="CAP-2",
        sub_capability_id="pattern_compilation",
        request_id=envelope.request_id,
    )
    emit_event(
        "sub_capability_completed",
        capability_id="CAP-2",
        sub_capability_id="pattern_compilation",
        request_id=envelope.request_id,
        duration_ms=0.0,
        status="success",
    )


def invoke_cap2(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    dataset_handle: ResolvedDataset,
) -> tuple[ProvenanceEnvelope, list[str]]:
    """Load the rubric's rule plan, run rules against a RuleContext
    built from the request, stamp rule_set_version + rules_fired.

    Decomposed into the 5 portal sub-capabilities (Rule loading,
    Pattern compilation, Input matching, Evidence capture, Result
    emission). Pattern compilation is SOFT — it lives inside
    ``load_plan_for_rubric`` and emits fire-rate signals only.
    Behaviour and outputs are identical to the pre-instrumentation
    implementation.
    """
    if not envelope.dataset_registry_id:
        raise MissingProvenanceError(
            "invoke_cap2: envelope.dataset_registry_id is absent; "
            "CAP-1 must run before CAP-2."
        )
    del dataset_handle  # CAP-2 uses the request directly; kept for symmetry

    effective_version = _resolve_effective_version(
        request.rubric_id, request.rubric_version
    )
    plan = _cap2_rule_loading(envelope, request.rubric_id, effective_version)
    _emit_pattern_compilation_soft(envelope)
    result = _cap2_input_matching(envelope, plan, request)
    rules_fired = _cap2_evidence_capture(envelope, result)
    stamped = _cap2_result_emission(envelope, plan.version, rules_fired)
    return stamped, rules_fired


VALID_LAYERS = frozenset({"L1", "L2", "L3", "L4"})
DEFAULT_LAYERS: tuple[str, ...] = ("L1",)


def invoke_cap7(
    envelope: ProvenanceEnvelope,
    request: SingleEvaluationRequest,
    dataset_handle: ResolvedDataset,
    *,
    layers: list[str] | None = None,
) -> tuple[ProvenanceEnvelope, dict[str, Any]]:
    """Run the hallucination pipeline on (response, source).

    ``layers`` selects which pipeline layers are enabled. Default is
    ``["L1"]`` — callers must explicitly request higher layers
    (CP-1b A1). Unknown layers raise ``ValueError`` from this
    wrapper (input validation, not a capability failure — so it's
    appropriate to raise here before calling the capability).

    Returns a serialisable verdict dict. prompt_version stamp
    deferred (D3).
    """
    if not envelope.dataset_registry_id:
        raise MissingProvenanceError(
            "invoke_cap7: envelope.dataset_registry_id is absent; "
            "CAP-1 must run before CAP-7."
        )
    del dataset_handle

    requested = tuple(layers) if layers else DEFAULT_LAYERS
    unknown = set(requested) - VALID_LAYERS
    if unknown:
        raise ValueError(
            f"invoke_cap7: unknown layers {sorted(unknown)}; "
            f"valid: {sorted(VALID_LAYERS)}"
        )
    active = {layer: (layer in requested) for layer in VALID_LAYERS}

    result = check_hallucination(
        response=request.response,
        context=request.source,
        source_context=request.source,
        case_id=envelope.request_id,
        # Embeddings are only needed when L2/L3 are active.
        skip_embeddings=not (active["L2"] or active["L3"]),
        l1_enabled=active["L1"],
        l2_enabled=active["L2"],
        l3_enabled=active["L3"],
        l4_enabled=active["L4"],
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
        "layers_requested": list(requested),
    }

    stamped = envelope.stamped(capability="CAP-7")
    return stamped, verdict


@timed(
    "cap5_envelope_reception",
    sub_capability_id="envelope_reception",
    capability_id="CAP-5",
)
def _cap5_envelope_reception(envelope: ProvenanceEnvelope) -> None:
    """CAP-5 Envelope reception: enforce that at least one upstream
    capability outcome has been recorded before writing a manifest.

    CP-1b (horizontal CAP-5) loosened the prior strict pre-check: a
    Total-degradation run (CAP-1 plus both siblings failed) is a
    legitimate manifest to write; the integrity trail is the audit.
    """
    if not envelope.integrity:
        raise MissingProvenanceError(
            "invoke_cap5: envelope.integrity is empty; at least one "
            "upstream capability outcome must be recorded before "
            "writing a manifest."
        )


def invoke_cap5(
    envelope: ProvenanceEnvelope,
    verdict: dict[str, Any],
    integrity: dict[str, Any],
    *,
    rubric_id: str,
    rubric_version: str,
    runs_root: Path | None = None,
) -> tuple[ProvenanceEnvelope, str]:
    """Write the governed manifest via record_evaluation_manifest, then
    append CAP-5 to the envelope chain.

    Decomposed into 4 instrumented sub-capabilities (Envelope
    reception, Manifest composition, Persistence, Lineage linking).
    The 5th portal sub-capability — Query interface — is not engaged
    on the write path; readers query elsewhere (run-registry walks,
    manifest fetchers). Behaviour and outputs are unchanged.
    """
    _cap5_envelope_reception(envelope)

    effective_version = _resolve_effective_version(rubric_id, rubric_version)
    manifest_id = record_evaluation_manifest(
        envelope=envelope,
        verdict=verdict,
        integrity=integrity,
        rubric_id=rubric_id,
        rubric_version=effective_version,
        runs_root=runs_root,
    )

    # Query interface is the read-side surface (run registry queries,
    # manifest lookups). The write path never engages it; emit the
    # skipped signal so batch aggregation reports honest 0/N
    # engagement rather than a missing event.
    emit_sub_capability_skipped(
        capability_id="CAP-5",
        sub_capability_id="query_interface",
        request_id=envelope.request_id,
        reason="manifest_write_does_not_query",
    )

    stamped = envelope.stamped(capability="CAP-5")
    return stamped, manifest_id
