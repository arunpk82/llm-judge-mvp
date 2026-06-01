"""Integration test: Exp 31 stage 2 ensemble parity on RAGTruth-50.

Sources of truth for the assertions below:

- experiments/exp31_stage2_ensemble.py — the experimental L2
  implementation whose contract this test locks. Aggregation logic
  at lines 860-912 (aggregate_verdicts); SentenceResult dataclass at
  lines 945-958.
- experiments/exp31_stage2_ensemble_results.json — numerical
  baseline at master HEAD 9b81f6f. verdict_counts = {grounded: 61,
  flagged: 96, unknown: 126}; safety_violations = 0;
  hallucination_catches = 11; elapsed_s = 2.1.
- docs/adr/0020-l2-flag-wins-ensemble-aggregation.md — the flag-wins
  ensemble contract. ADR-0020:79 cites the combined L1+L2 form:
  "78 cleared at 100% precision, 11/16 hallucinations caught, 0
  safety violations."
- docs/adr/0031-l2-cache-miss-contract.md — cache behavior on miss.
  The fixture below preseeds the graph cache from exp31 fact tables
  so this test asserts pure ensemble behavior, not cache-miss
  semantics.
- src/llm_judge/calibration/hallucination_graphs.py — the in-tree
  production port of the Exp 31 ensemble.
- src/llm_judge/calibration/hallucination.py:696-743 — the L2
  cascade call site. Note: the ``unknown`` verdict has no handler
  (row 7c of the 2026-04-21 delta table). Sentences silently fall
  through. This test therefore asserts the 5 Exp 31 ``unknown``
  cases by ABSENCE of an L1/L2/L2_flagged sentence_result entry,
  not by presence of an ``L2_unknown`` record.
- tests/smoke/test_l1_parity.py — template for parametrised
  per-sentence parity assertions (the L1 analogue, PR #170).

Why this test exists: Exp 31 is the experimental L2 implementation.
Production L2 is the port. Nothing on master verifies the port
reproduces Exp 31's RAGTruth-50 numbers end-to-end. Unit tests
cover builders and traversals on toy fixtures;
tools/run_ragtruth50.py measures but does not gate. This is the CI
gate for the L2 ensemble contract.

Failure protocol: if any assertion fails, rerun
``HALLUCINATION_LAYERS=l1,l2 poetry run python tools/run_ragtruth50.py
benchmark`` on master HEAD. Paste the measured numbers into the
failure triage. DO NOT tune tolerances or edit expectations here.
A failure is a real finding: production has drifted from the Exp 31
contract. Expectations change only after a new ADR formally revises
the contract and documents the new baseline.

Gating: this is the parity floor for L2. Phase 2 work
(Exp 35 five-graph integration, aggregator changes, builder
rewrites) must keep this test green OR ship a replacement ADR.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# The 11 RAGTruth-50 hallucinations that Exp 31 CAUGHT (verdict=flagged).
# Extracted from exp31_stage2_ensemble_results.json by filtering
# gt_label == "hallucinated" AND verdict == "flagged".
# Production must emit a sentence_result with resolved_by == "L2_flagged"
# for each.
EXP31_CAUGHT: list[tuple[str, int]] = [
    ("ragtruth_28", 0),
    ("ragtruth_28", 1),
    ("ragtruth_29", 1),
    ("ragtruth_44", 0),
    ("ragtruth_46", 4),
    ("ragtruth_68", 0),
    ("ragtruth_122", 0),
    ("ragtruth_124", 2),
    ("ragtruth_141", 2),
    ("ragtruth_180", 0),
    ("ragtruth_184", 2),
]

# The 5 RAGTruth-50 hallucinations that Exp 31 MISSED (verdict=unknown).
# Production's L2 cascade has no ``unknown`` handler. With L3/L4
# disabled, these sentences silently fall through — no L1/L2/L2_flagged
# sentence_result entry. Assertion is by absence.
EXP31_MISSED: list[tuple[str, int]] = [
    ("ragtruth_58", 4),
    ("ragtruth_121", 2),
    ("ragtruth_123", 1),
    ("ragtruth_146", 2),
    ("ragtruth_185", 3),
]


@pytest.fixture(scope="module")
def l1_l2_benchmark_run(tmp_path_factory):
    """Run the full RAGTruth-50 benchmark with L1+L2 enabled, L3+L4
    disabled. Module-scoped because the run takes ~2-3 minutes.

    The graph cache singleton reads its directory from the pipeline
    config at first call. We keep the default .cache/hallucination_graphs
    location and seed it with Exp 31 fact tables — preseed is
    idempotent and writes files keyed by SHA-256 source hash, so it
    does not corrupt anything if the cache is already populated.
    """
    from dataclasses import replace

    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
    from llm_judge.benchmarks.runner import run_benchmark
    from llm_judge.calibration.graph_cache import (
        GraphCache,
        preseed_from_exp31,
        reset_graph_cache,
    )
    from llm_judge.calibration.pipeline_config import (
        LayerConfig,
        get_pipeline_config,
    )

    exp31_path = REPO_ROOT / "experiments" / "exp31_multipass_fact_tables.json"
    source_file = (
        REPO_ROOT / "datasets" / "benchmarks" / "ragtruth" / "source_info.jsonl"
    )
    response_file = (
        REPO_ROOT / "datasets" / "benchmarks" / "ragtruth" / "response.jsonl"
    )
    benchmark_file = (
        REPO_ROOT
        / "datasets"
        / "benchmarks"
        / "ragtruth"
        / "ragtruth_50_benchmark.json"
    )
    for p in (exp31_path, source_file, response_file, benchmark_file):
        assert p.exists(), f"required input missing: {p}"

    with open(benchmark_file) as f:
        benchmark = json.load(f)
    valid_ids = {str(rid) for rid in benchmark["response_ids"]}

    sources: dict[str, str] = {}
    with open(source_file) as f:
        for line in f:
            item = json.loads(line)
            sources[str(item["source_id"])] = item.get("source_info", "")

    source_texts: dict[str, str] = {}
    with open(response_file) as f:
        for line in f:
            item = json.loads(line)
            rid = str(item["id"])
            if rid in valid_ids:
                sid = str(item["source_id"])
                source_texts[f"ragtruth_{rid}"] = sources.get(sid, "")

    config = get_pipeline_config()
    cache_dir = REPO_ROOT / config.graph_cache.directory
    cache_dir.mkdir(parents=True, exist_ok=True)
    seed_cache = GraphCache(cache_dir, ttl_hours=config.graph_cache.ttl_hours)
    preseed_from_exp31(seed_cache, exp31_path, source_texts)
    # Force the runtime singleton to pick up a fresh view of this dir.
    reset_graph_cache()

    config = replace(
        config,
        layers=LayerConfig(
            l1_enabled=True,
            l2_enabled=True,
            l3_enabled=False,
            l4_enabled=False,
        ),
    )

    adapter = RAGTruthAdapter()
    adapter.set_benchmark_filter(benchmark_file)

    checkpoint_dir = tmp_path_factory.mktemp("l2_exp31_checkpoint")

    result = run_benchmark(
        adapter,
        split="test",
        max_cases=None,
        skip_embeddings=True,
        with_llm=False,
        gate2_routing="pass",
        config=config,
        checkpoint_every=50,
        checkpoint_dir=str(checkpoint_dir),
    )
    return result


@pytest.fixture(scope="module")
def sentence_layer_index(l1_l2_benchmark_run):
    """Map (case_id, sentence_idx) -> set of resolved_by layers.

    Used by the parametrised per-sentence assertions. A sentence with
    no entry in any sentence_results list appears as an empty set on
    .get(..., set())."""
    index: dict[tuple[str, int], set[str]] = defaultdict(set)
    for rr in l1_l2_benchmark_run.response_level_results:
        case_id = rr["case_id"]
        hal = rr.get("hallucination") or {}
        for sr in hal.get("sentence_results") or []:
            idx = sr.get("sentence_idx", -1)
            layer = sr.get("resolved_by", "")
            if layer:
                index[(case_id, idx)].add(layer)
    return dict(index)


@pytest.mark.integration
class TestExp31EnsembleContract:
    """Aggregate parity against Exp 31 stage 2 baseline."""

    def test_l2_grounded_count(self, l1_l2_benchmark_run):
        """L2 grounded count matches Exp 31 baseline (61) within ±1."""
        by_layer = l1_l2_benchmark_run.sentence_level_metrics.get("by_layer", {})
        l2_total = by_layer.get("L2", {}).get("total", 0)
        assert 60 <= l2_total <= 62, (
            f"L2 grounded drift: got={l2_total}, expected=61±1. "
            f"Exp 31 baseline: verdict_counts.grounded=61 in "
            f"exp31_stage2_ensemble_results.json. Tolerance ±1 absorbs "
            f"spaCy/model nondeterminism. Drift outside this range means "
            f"production L2 no longer matches the Exp 31 ensemble contract "
            f"(ADR-0020)."
        )

    def test_hallucination_catches(self, l1_l2_benchmark_run):
        """Exactly 11 of 16 hallucinations are L2_flagged (exact, no tolerance)."""
        by_layer = l1_l2_benchmark_run.sentence_level_metrics.get("by_layer", {})
        catches = by_layer.get("L2_flagged", {}).get("flagged_hallucinated", 0)
        assert catches == 11, (
            f"L2 hallucination catch drift: got={catches}, expected=11 exact. "
            f"Exp 31 baseline: hallucination_catches=11 in "
            f"exp31_stage2_ensemble_results.json. ADR-0020:79 cites 11/16 "
            f"as the combined L1+L2 recall contract."
        )

    def test_no_safety_violations(self, l1_l2_benchmark_run):
        """No deterministic layer clears a hallucinated sentence (exact, 0)."""
        by_layer = l1_l2_benchmark_run.sentence_level_metrics.get("by_layer", {})
        l2_violations = by_layer.get("L2", {}).get("cleared_hallucinated", 0)
        l1_violations = by_layer.get("L1", {}).get("cleared_hallucinated", 0)
        assert l1_violations == 0 and l2_violations == 0, (
            f"Safety violation: L1 cleared {l1_violations} hallucination(s), "
            f"L2 cleared {l2_violations} hallucination(s). Expected 0 for "
            f"both. Any non-zero value means a deterministic layer is "
            f"approving content that RAGTruth labels hallucinated — a "
            f"precision breach in the cascade floor."
        )

    def test_combined_l1_l2_clearance(self, l1_l2_benchmark_run):
        """L1 + L2 combined grounded count ~78 per ADR-0020:79 (±2)."""
        by_layer = l1_l2_benchmark_run.sentence_level_metrics.get("by_layer", {})
        l1_total = by_layer.get("L1", {}).get("total", 0)
        l2_total = by_layer.get("L2", {}).get("total", 0)
        combined = l1_total + l2_total
        assert 76 <= combined <= 80, (
            f"Combined L1+L2 clearance drift: got={combined} "
            f"(L1={l1_total}, L2={l2_total}), expected=78±2 per ADR-0020:79. "
            f"Tolerance ±2 absorbs L1/L2 overlap variance. Drift outside "
            f"this range means combined deterministic clearance has shifted "
            f"and warrants a new measurement run plus an ADR update."
        )


@pytest.mark.integration
@pytest.mark.parametrize(
    "case_id,sentence_idx",
    EXP31_CAUGHT,
    ids=[f"{cid}_S{idx}" for cid, idx in EXP31_CAUGHT],
)
def test_exp31_caught_is_l2_flagged(
    case_id: str,
    sentence_idx: int,
    sentence_layer_index: dict[tuple[str, int], set[str]],
) -> None:
    """Each of the 11 Exp 31-caught hallucinations must be L2_flagged."""
    layers = sentence_layer_index.get((case_id, sentence_idx), set())
    assert "L2_flagged" in layers, (
        f"L2 parity break: Exp 31 caught {case_id} S{sentence_idx} "
        f"(verdict=flagged), but production resolved_by set = "
        f"{sorted(layers) or '(none)'}. Expected a sentence_result with "
        f"resolved_by='L2_flagged'. Missing L2_flagged means the L2 "
        f"ensemble has drifted — either the flag-wins aggregator "
        f"(ADR-0020), a graph builder/traversal, or spaCy parse behavior "
        f"has changed."
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "case_id,sentence_idx",
    EXP31_MISSED,
    ids=[f"{cid}_S{idx}" for cid, idx in EXP31_MISSED],
)
def test_exp31_missed_has_no_deterministic_resolution(
    case_id: str,
    sentence_idx: int,
    sentence_layer_index: dict[tuple[str, int], set[str]],
) -> None:
    """Each of the 5 Exp 31-missed hallucinations must NOT be resolved
    by L1, L2, or L2_flagged. Assertion is by absence because
    production has no ``unknown`` handler (drift row 7c); L3/L4 are
    disabled by this fixture, so no later layer can resolve either."""
    layers = sentence_layer_index.get((case_id, sentence_idx), set())
    deterministic = layers & {"L1", "L2", "L2_flagged"}
    assert not deterministic, (
        f"L2 parity break: Exp 31 returned 'unknown' for {case_id} "
        f"S{sentence_idx}, but production resolved it via "
        f"{sorted(deterministic)}. Each path is a distinct finding: "
        f"resolved_by='L2' is a SAFETY VIOLATION (L2 grounded a "
        f"hallucinated sentence); resolved_by='L1' means L1 cleared a "
        f"hallucination (also a safety violation); resolved_by='L2_flagged' "
        f"is an UNEXPECTED GAIN (real precision improvement — ship a new "
        f"ADR before editing this expectation)."
    )
