# LLM-Judge Platform — Architecture Portal

**Generated:** 2026-05-05 (refreshed after L1-Pkt-B verified on master)
**Source state:** master HEAD (`b3eaf56 L1-Pkt-B`)
**Generator:** Claude Code, executing architect-chat brief v2.1
**Inputs used:** code-review-graph knowledge graph (3,299 nodes / 30,453 edges, indexed 2026-05-05T05:24:36), `.code-review-graph/wiki/` (15 auto-clustered pages), `docs/adr/architecture-map-8-layer-specification.md` (Proposed), source files, git log (last 30 days).

> Interactive visualization: open `.code-review-graph/graph.html` in a browser. The portal complements that view with the 8-layer human abstraction the auto-clustering does not capture.

---

## 1. Overview

LLM-Judge is a hallucination/quality evaluation platform organized as an 8-layer architecture per `docs/adr/architecture-map-8-layer-specification.md` (Proposed). The platform's executable surface is a **PlatformRunner** (`src/llm_judge/control_plane/runner.py`) that orchestrates a registry-driven capability sequence — **CAP-1 → [CAP-2, CAP-7] → CAP-5** — over a Pydantic-validated request, accumulating outputs and HMAC-integrity records onto a `ProvenanceEnvelope`. As of L1-Pkt-B (`b3eaf56`), the sequence is iterated from `CAPABILITY_REGISTRY` (a frozen tuple of `CapabilitySpec` entries with per-capability timeouts) and every invocation flows through a guardrail substrate.

The 8-layer model is a human abstraction over the package structure. Of the **11 capabilities** the architecture map enumerates (CAP-1 through CAP-11 minus gaps), **4 are formally registered and runtime-orchestrated** (CAP-1, CAP-2, CAP-7, CAP-5 — all in `CAPABILITY_REGISTRY`) and **CAP-10 is referenced as a tuple-constant only**. The rest (CAP-3, CAP-4, CAP-6, CAP-8, CAP-9, CAP-11) are specified but unimplemented or implemented only as components without capability-id stamping. Last-30-day activity is dominated by Layer 1 Phase 2 hardening (CP-F1 through CP-F14, L1-Pkt-1/2/A/B) and Phase 1 verification work in `experiments/`.

Auto-clustering (Leiden) produced 14 communities that **do not** map to either the 8 layers or the capability set; the wiki at `.code-review-graph/wiki/` is included as raw input but the portal does not adopt its grouping.

---

## 2. 8-Layer architecture

### Layer 1 · Control

- **One-line:** Single entry point and capability ordering, with formal registry + guardrail substrate.
- **Best-fit packages:** `src/llm_judge/control_plane/` (13 modules + `guardrails/` subpackage of 3 modules, 3,068 LOC total). Clean alignment — this is the only layer whose package boundary is unambiguous.
- **Capabilities owned:** none (Layer 1 is orchestration components, not a capability host).
- **Capability components stamped here:** CAP-1, CAP-2, CAP-5, CAP-7 wrappers (`wrappers.py`), with runtime metadata in `capability_registry.py` (`CAPABILITY_REGISTRY`) and per-invocation timeout enforcement in `guardrails/timeout.py` (`TimeoutGuardrail`).
- **Status:** **Operational, Phase 2 Categories 1+2 closed.** `tests/control_plane/` has 33 test files, the most-imported production package (91 test imports). L1-Pkt-A (CP-F1, CP-F3) and L1-Pkt-B (CP-F8 partial, CP-F9, CP-F12) shipped within last 30 days. Test count site-wide: 1071 → 1092 (+21 from L1-Pkt-B). Coverage 83.67% per merge commit. Remaining Phase 2 categories: L1-Pkt-C (Cat 3 exception taxonomy refactor — CP-F10/F20/F22) and L1-Pkt-D (Cat 4 event vocabulary + governance preflight — CP-F7/F13).
- **Key data contracts:** `ProvenanceEnvelope`, `CapabilityIntegrityRecord`, `SingleEvaluationRequest`, `SingleEvaluationResult`, `BenchmarkReference`, `Integrity`, `BatchCase`, `BatchInputFile`, `CaseResult`, `BatchResult`, `BatchAggregation`, **`CapabilitySpec`** (envelope.py, types.py, batch_input.py, batch_runner.py, batch_aggregation.py, **capability_registry.py**), plus `GuardrailContext` / `GuardrailDecision` in `guardrails/substrate.py`.
- **Recent significant changes:**
  - `b3eaf56` **L1-Pkt-B** — substrate + timeout + capability registry (closes CP-F9, CP-F12; partial CP-F8)
  - `85c7e5c` L1-Pkt-A — CP-F1 + CP-F3 closure (parallel-entry elimination, field-ownership runtime gate)
  - `994eaba` L1-Pkt-2 — CP-F4 + CP-F5 + CP-F14 cleanup
  - `9c145d4` L1-Pkt-1 — HMAC mode + startup validation (CP-F2 + CP-F11)
  - `8e57430` Phase 1 — platform-as-harness verification + sub-cap instrumentation

### Layer 2 · Governance Foundation (Data Plane)

- **One-line:** Define what's allowed; establish the truth everything else operates against.
- **Best-fit packages:** `src/llm_judge/datasets/` (6 modules — CAP-1 home), parts of `src/llm_judge/rubric_store.py` and `src/llm_judge/rubrics/` (note: ADR explicitly flags rubric governance as wrongly placed in Layer 7).
- **Capabilities owned:** CAP-1 (Dataset Governance, present), CAP-3 (Rule Governance, **absent from code**), CAP-4 (Baseline Governance, **absent from code**).
- **Status:** **Partial.** CAP-1 operational with sub-capability instrumentation (Reception, Hashing, Validation, Registration, Lineage tracking — all stamped in `wrappers.py`). CAP-3 and CAP-4 are not present as `capability_id`-stamped wrappers — baseline mechanics live in `eval/baseline.py` but without capability registration.
- **Key data contracts:** `DatasetMetadata`, `MessageRow`, `EvalCaseRow` (`datasets/models.py`); `BenchmarkReference` (lives in `control_plane/types.py` per Pre-flight 6 recon to avoid registry→types circular import).
- **Recent significant changes:** `85c7e5c` added benchmark provenance fields to CAP-1's allowlist; `e2abcf5` adapter registry factory closures (CP-3 finding).

### Layer 3 · Evaluation Engine (Data Plane)

- **One-line:** Run evaluation work; apply governance to inputs and produce verdicts.
- **Best-fit packages:** `src/llm_judge/rules/` (CAP-2 — 19 py files), `src/llm_judge/calibration/` (CAP-7 hallucination cascade — 14 py files). Imperfect: CAP-7's L1 substring matching lives in `calibration/hallucination.py`, while the cascade design encompasses L1–L5 layers, only some of which are wired (per ADR-listed F-numbers).
- **Capabilities owned:** CAP-2 (Rule Engine, present), CAP-7 (Evaluation cascade, present).
- **Status:** **Operational for L1; partial for L2–L5.** CAP-7 sub-capability instrumentation present at `calibration/hallucination.py:652+` for L1. Per ADR §Layer 3, F17 (L2 silent no-op), F16 (L3 fact_counting flag flip never landed), F18 (L4 unvalidated), F12 (L5 unwired), F13 (DEFAULT_LAYERS=("L1",)), F14 (misnamed gate check). CAP-2 has sub-capability tests (`tests/control_plane/test_cap2_sub_capabilities.py`).
- **Key data contracts:** `Flag`, `RuleContext`, `RuleResult` (`rules/types.py`); rubric metric schema via `rule_plan_yaml.py`.
- **Recent significant changes:** `8e57430` calibration L1 sub-cap instrumentation + L1 completion integration tests.

### Layer 4 · Calibration & Trust (Data Plane)

- **One-line:** Verify the evaluator; meta-evaluation feedback loop.
- **Best-fit packages:** `src/llm_judge/calibration/` partial overlap (note: `calibration/` also hosts CAP-7 work — package and layer don't separate cleanly). Specifically `calibration/adjudication.py`, `calibration/bias.py`, `calibration/feedback.py`.
- **Capabilities owned:** CAP-10 (Calibration & Adjudication).
- **Status:** **Stub / Partial.** CAP-10 appears only as a constant in `control_plane/batch_aggregation.py` (`HORIZONTAL_CAPABILITIES: tuple[str, ...] = ("CAP-6", "CAP-10")`); no `capability_id="CAP-10"` stamping wrapper exists. Per ADR: adjudication.py exists but unwired (F12), feedback loop to Layer 3 not established, bias detection not running across full property set.
- **Key data contracts:** none stamped under CAP-10's allowlist (no entry in `field_ownership.py`).
- **Recent significant changes:** none in last 30 days specific to CAP-10 instrumentation.

### Layer 5 · Evidence & Observability (Data Plane)

- **One-line:** Record what happened; detect drift; surface platform health.
- **Best-fit packages:** `src/llm_judge/eval/cap5_entry.py` (canonical CAP-5 entry, 4 sub-capability wrappers), `src/llm_judge/eval/registry.py`, `src/llm_judge/eval/event_registry.py`, `src/llm_judge/eval/drift.py`, `src/llm_judge/eval/diff.py`, `src/llm_judge/control_plane/observability.py`, `src/llm_judge/control_plane/event_bus.py`. Layer is **split across packages**.
- **Capabilities owned:** CAP-5 (Artifact Governance, present), CAP-6 (Drift Monitoring, **constant only**), CAP-9 (Platform Health, **absent from code**).
- **Status:** **Operational for CAP-5; CAP-6 minimal; CAP-9 absent.** CAP-5 has sub-capability stamping for Manifest composition / Persistence / Lineage linking (`eval/cap5_entry.py:56,91,106`) and is the only capability NOT wrapped in fail-fast (per Layer 1 design D5: CAP-5 propagates). CAP-6 is referenced as a HORIZONTAL_CAPABILITIES constant; no wrappers with `capability_id="CAP-6"`. CAP-9 has no implementation. EventBus (`control_plane/event_bus.py`) and observability primitives (`control_plane/observability.py`) provide the foundation; `eval/event_registry.py` persists events to journal.
- **Key data contracts:** manifest JSON via `eval/cap5_entry.py`; `CapabilityIntegrityRecord` accumulates onto `ProvenanceEnvelope`.
- **Recent significant changes:** `f936383` CP-2 observability foundation (event_bus + observability + envelope hooks); `f184df5` CP-3 batch sub-cap instrumentation; `2693c6f` CAP-5 sub-capabilities tests.

### Layer 6 · Operator Surface (Data Plane)

- **One-line:** Translate platform state into human-actionable form.
- **Best-fit packages:** `Makefile` (28 phony targets), `tools/` (30 churn entries last 30 days — outside `src/`), `experiments/render_layer_verification_report.py`, `tools/_batch_html_report.py`, `tools/_batch_terminal.py`. **No `src/llm_judge/` package owns Layer 6.**
- **Capabilities owned:** CAP-8 (Developer Experience, **components present, no capability stamping**), CAP-11 (Reporting, **not yet filed**).
- **Status:** **Operational without capability registration.** Make targets exist (see §8). Rendering and report tooling exist. Per ADR: CAP-11 not filed; reporting today distributed across `tools/`, `experiments/`, partial CAP-9 with no capability ownership.
- **Key data contracts:** none stamped under CAP-8/CAP-11 allowlist.
- **Recent significant changes:** `8e57430` Phase 1 added `experiments/render_layer_verification_report.py` (12 tests in `tests/calibration/test_layer_verification_renderer.py`); `f184df5` added `tools/_batch_html_report.py` + `tools/_batch_terminal.py`.

### Layer 7 · Management

- **One-line:** Cross-cutting governance fabric (rubric governance, schema enforcement, ADR ledger).
- **Best-fit packages:** `src/llm_judge/rubric_store.py`, `src/llm_judge/rubrics/lifecycle.py`, `src/llm_judge/governance.py`, `src/llm_judge/rule_plan_yaml.py`, `docs/adr/`. Pydantic schema enforcement is woven through Pydantic models on all data contracts.
- **Capabilities owned:** none (Management is governance fabric, not a capability host).
- **Status:** **Operational with known architectural debt.** Per ADR: rubric governance currently lives in Layer 7 but per industry pattern belongs in Layer 2; migration logged as architectural debt. Rubric lifecycle has 4 test files in `tests/rubrics/`. F11 (governance preflight doesn't validate prompt artifact existence), F16 (ADR-0027 fact_counting default flag flip never landed).
- **Key data contracts:** `RubricAuditEvent`, `RubricLifecycleEntry`, `RubricMetricsSchema`, `RubricVersionEntry`, `RubricRegistrySection`, `RubricRegistryConfig`, `RulePlanRule`, `RulePlanConfig`.
- **Recent significant changes:** `e333b8a` CP-1c-b.2 binding enforcement (closes rubric governance arc); `2693c6f` CP-1c-b.1 rubric binding plumbing; `ccad817` CP-1c-a rubric governance core.

### Layer 8 · Presentation

- **One-line:** Top-level platform framing — demonstrations, walkthroughs.
- **Best-fit packages:** `Makefile` targets `demo`, `demo-batch*`, `verify-l1`; `tools/demo_platform.py`; `README.md`; `docs/`. **Outside `src/llm_judge/`.**
- **Capabilities owned:** none (Presentation is platform framing, not a capability host).
- **Status:** **Light footprint, operational.** Per ADR: walkthroughs predate ADR-0029 cascade renaming (deferred rewrite), top-level dashboards absent.
- **Key data contracts:** none.
- **Recent significant changes:** `f184df5` README + batch demo example; `f936383` README updates + demo_platform.py.

---

## 3. Capability inventory

| Capability | Layer | Registered? | Timeout | Implementation locus | Status | Field-ownership entries | Notes |
|------------|-------|-------------|---------|---------------------|--------|-------------------------|-------|
| CAP-1 · Dataset Governance | 2 | ✅ pos 0 | 5.0 s | `control_plane/wrappers.py` (5 sub-cap wrappers) + `datasets/benchmark_registry.py` | Operational | `dataset_registry_id`, `input_hash`, `benchmark_id`, `benchmark_version`, `benchmark_content_hash`, `benchmark_registration_timestamp` | Reception, Hashing, Validation, Registration, Lineage tracking sub-caps stamped. |
| CAP-2 · Rule Engine | 3 | ✅ pos 1 | 5.0 s | `rules/engine.py` invoked via runner registry iteration | Operational | `rule_set_version`, `rules_fired` | Sub-cap tests at `tests/control_plane/test_cap2_sub_capabilities.py`. |
| CAP-3 · Rule Governance | 2 | ❌ | — | **Absent from code.** | Not implemented | n/a | ADR-specified; `rules/lifecycle.py` exists but lacks `capability_id="CAP-3"` stamping. |
| CAP-4 · Baseline Governance | 2 | ❌ | — | `eval/baseline.py` (no capability stamping) | Partial | n/a | Make targets exist; not wrapped as a `@timed` capability. |
| CAP-5 · Artifact Governance | 5 | ✅ pos 3 | 5.0 s | `eval/cap5_entry.py` (3 sub-cap wrappers) | Operational | (empty frozenset — chain stamping only) | Manifest composition / Persistence / Lineage linking. NOT wrapped in fail-fast (D5 design). |
| CAP-6 · Drift Monitoring | 5 | ❌ | — | `eval/drift.py`, `eval/diff.py` (no capability stamping) | Stub | n/a | Referenced only as `HORIZONTAL_CAPABILITIES` tuple constant. |
| CAP-7 · Evaluation cascade | 3 | ✅ pos 2 | 30.0 s | `calibration/hallucination.py` (6 sub-cap entries) | Partial | (empty frozenset — chain stamping only) | L1 instrumentation operational; L2/L3/L4/L5 cascade gaps F12/F13/F16/F17/F18 per ADR. 30s timeout per L1-Pkt-B due to LLM warm-up volatility. |
| CAP-8 · Developer Experience | 6 | ❌ | — | `Makefile` (28 phony targets), `tools/` | Operational (no capability stamping) | n/a | No `capability_id="CAP-8"` references in code. |
| CAP-9 · Platform Health | 5 | ❌ | — | **Absent from code.** | Not implemented | n/a | ADR notes "pending EPIC-9.x". |
| CAP-10 · Calibration & Adjudication | 4 | ❌ | — | `calibration/adjudication.py`, `calibration/bias.py`, `calibration/feedback.py` (no capability stamping) | Stub | n/a | Referenced only as `HORIZONTAL_CAPABILITIES` tuple constant. F12: adjudication unwired from CAP-7 cascade. |
| CAP-11 · Reporting | 6 | ❌ | — | **Not yet filed.** Components in `experiments/render_layer_verification_report.py`, `tools/_batch_html_report.py`. | Not implemented | n/a | ADR notes "committed, not yet filed". |

**Capability registry (`CAPABILITY_REGISTRY`)** — landed on master in `b3eaf56` (L1-Pkt-B). Source: `src/llm_judge/control_plane/capability_registry.py`. Frozen `tuple[CapabilitySpec, ...]` with 4 entries; `CapabilitySpec` is a frozen Pydantic model with `capability_id: str`, `sequence_position: int`, `timeout_seconds: float`. Lookup helper `get_spec(capability_id)` raises `KeyError` on unknown id. Of 11 ADR-specified capabilities, **4 are formally registered** (CAP-1, CAP-2, CAP-7, CAP-5 — exactly the runtime orchestration sequence); the remaining 7 are not registered.

**Capability dependencies:** runtime sequence consumed from `CAPABILITY_REGISTRY` by `control_plane/runner.py:263` (Option A dispatch-by-id pattern):
- CAP-1 (wrapped, fail-fast: CAP-1 failure → CAP-2 + CAP-7 skipped, CAP-5 still writes manifest per D5)
- CAP-2 and CAP-7 (sibling phase, failures tolerated)
- CAP-5 (propagates; not wrapped per design D5)
- Drive-by fix in L1-Pkt-B: `batch_aggregation.VERTICAL_CAPABILITIES` now derived from `CAPABILITY_REGISTRY` (was a parallel hard-coded tuple — the second hard-coded site that the gap-absence test caught).

**Per-invocation guardrails (post-L1-Pkt-B):** every capability call passes through `guardrails/substrate.py:guardrail_context()` which fires pre-call + post-call hooks (`guardrail.pre_call`, `guardrail.post_call` structlog events). `TimeoutGuardrail` (`guardrails/timeout.py`) implements **Option β**: post-completion denial. Capability runs to completion regardless of timeout; if elapsed exceeds `timeout_seconds`, `GuardrailDeniedError` is raised post-completion with structured context and `guardrail.timeout_exceeded` is emitted. Resource-interruption gap recorded as **CP-F23 candidate** for v1.5.

---

## 4. Cross-layer dependencies

### Per-package outbound imports (AP-Q2)

| From package | Imports from |
|--------------|-------------|
| `benchmarks/` | `calibration`, `properties`, `schemas` |
| `calibration/` | `control_plane`, `judge_base`, `paths`, `schemas` |
| `control_plane/` | `calibration`, `datasets`, `eval`, `paths`, `rules`, `schemas` |
| `datasets/` | `control_plane`, `paths` |
| `eval/` | `control_plane`, `datasets`, `governance`, `paths`, `rubric_store`, `runtime`, `schemas` |
| `properties/` | `judge_base`, `schemas` |
| `retrieval/` | (no cross-package imports) |
| `rubrics/` | (no cross-package imports) |
| `rules/` | `control_plane`, `paths`, `rule_plan_yaml`, `schemas` |

### Most-frequent cross-package edges (top imports by call count)

| Import | Count |
|---|---|
| `from llm_judge.schemas import Message, PredictRequest` | 11 |
| `from llm_judge.calibration.hallucination import (...)` | 11 |
| `from llm_judge.properties import get_embedding_provider` | 10 |
| `from llm_judge.benchmarks.ragtruth import RAGTruthAdapter` | 10 |
| `from llm_judge.paths import state_root` | 8 |
| `from llm_judge.judge_base import JudgeEngine` | 8 |
| `from llm_judge.benchmarks import (...)` | 8 |

### Layer-boundary observations

The intended cross-layer relationships per ADR (§Cross-layer relationships) and what the import graph shows:

| Intended | Realized in code? |
|----------|-------------------|
| Layer 1 → all Data Plane | ✓ `control_plane/` imports from `calibration` (CAP-7), `datasets` (CAP-1), `eval` (CAP-5), `rules` (CAP-2). |
| Layer 2 → Layer 3 | ✓ `datasets` does **not** import from `rules` or `calibration` directly (governance flows downstream via Layer 1 envelope). |
| Layer 3 → Layer 5 | Indirect — CAP-7 produces verdicts consumed by CAP-5 via `ProvenanceEnvelope` rather than direct imports. |
| Layer 4 → Layer 3 (feedback loop) | ✗ ADR notes this is currently absent. |
| Layer 5 → Layer 6 | Out of `src/`, lives in `tools/` and `experiments/`. |

### Cross-community edges (graph-detected)

The auto-clustering detected only **one cross-community edge cluster**: 16 CALLS edges from `tests/calibration/test_layer_verification_renderer.py` → `experiments/render_layer_verification_report.py`. This is intra-Layer-6 from the ADR's perspective (verification report tooling tested by calibration tests) and matches the recent `8e57430` Phase 1 work.

---

## 5. Layer 1 detail (Phase 2 hardening reflection)

### Module inventory (`src/llm_judge/control_plane/` + `guardrails/` subpackage)

| Module | LOC | Purpose |
|--------|-----|---------|
| `runner.py` | 489 | `PlatformRunner` — iterates `CAPABILITY_REGISTRY` via Option A dispatch-by-id (refactored in L1-Pkt-B) |
| `wrappers.py` | 495 | Capability invocation wrappers; sub-cap `@timed` decorators |
| `envelope.py` | 275 | `ProvenanceEnvelope` + `CapabilityIntegrityRecord` + HMAC stamping |
| `batch_runner.py` | 269 | Batch wrapper around Runner |
| `guardrails/substrate.py` | 259 | `GuardrailContext`, `GuardrailDecision`, `guardrail_context()` (CP-F8 partial) |
| `observability.py` | 250 | `Timer`, `emit_event`, sub-capability instrumentation primitives |
| `batch_aggregation.py` | 220 | `BatchAggregation`; `VERTICAL_CAPABILITIES` derived from `CAPABILITY_REGISTRY` |
| `types.py` | 212 | Request/result Pydantic models + 7 exception classes (added `GuardrailDeniedError` in L1-Pkt-B) |
| `configuration.py` | 174 | Mode-aware config validation (production HMAC required, etc.) |
| `event_bus.py` | 110 | EventBus pub/sub primitive |
| `batch_input.py` | 102 | YAML/JSON batch input file parsing + schema |
| `guardrails/timeout.py` | 94 | `TimeoutGuardrail` — Option β post-completion denial (CP-F12 closure) |
| `capability_registry.py` | 87 | `CAPABILITY_REGISTRY` + `CapabilitySpec` + `get_spec()` (CP-F9 closure) |
| `guardrails/__init__.py` | 49 | Guardrails subpackage public surface |
| `field_ownership.py` | 41 | `FIELD_OWNERSHIP` allowlist (CP-F3 closure) |
| `__init__.py` | 24 | Public re-exports |
| **Total** | **3,148** | 16 modules (12 in `control_plane/` + 3 in `guardrails/` + 1 registry) |

LOC delta from L1-Pkt-B: +569 lines (registry 87 + guardrails 402 + runner +57 + types +25 — minus minor batch_aggregation refactor). Note: 16-module total is correct; `control_plane/` has 13 direct modules (12 + capability_registry) plus the `guardrails/` subpackage with 3 modules.

### Capability registry contents (AP-Q6)

From `src/llm_judge/control_plane/capability_registry.py` (added in `b3eaf56` L1-Pkt-B, closing CP-F9):

| Position | Capability | Timeout (s) | P99 observed (50-case verify-l1) |
|----------|-----------|-------------|----------------------------------|
| 0 | CAP-1 | 5.0 | 32.9 ms |
| 1 | CAP-2 | 5.0 | 44.5 ms |
| 2 | CAP-7 | 30.0 | 3925 ms (case-0 LLM warm-up volatility headroom) |
| 3 | CAP-5 | 5.0 | 26.5 ms |

The registry is `tuple[CapabilitySpec, ...]` — frozen at import. `CapabilitySpec` is a frozen Pydantic model with three fields (`capability_id`, `sequence_position`, `timeout_seconds`). `sequence_position` is redundant with the tuple index but retained as an explicit field so a spec viewed in isolation records its chain position. `get_spec(capability_id)` performs a linear scan (negligible cost; small fixed registry) and raises `KeyError` on unknown id.

Per the module docstring: registry is **strict-from-day-one** — adding a fifth capability requires a registry entry plus a new dispatch arm in `runner.py`. Callable references are intentionally NOT in the registry: capabilities have heterogeneous signatures (CAP-1 envelope/payload, CAP-2 with dataset_handle, CAP-7 with layers, CAP-5 with rubric_id/runs_root) plus conditional dispatch on CAP-1 failure / CAP-7→CAP-5 aggregation; uniform-signature refactor (Option C from Layer 1 chat checkpoint) deferred.

Timeouts are operational starting points from a single `make verify-l1` 50-case run on master @ `85c7e5c`; sample is small (test runs only). Tune post-deployment as production latency data accumulates.

**Drive-by fix:** `batch_aggregation.VERTICAL_CAPABILITIES` was a parallel hard-coded tuple — L1-Pkt-B refactored it to `tuple(spec.capability_id for spec in CAPABILITY_REGISTRY)`. The gap-absence test caught the second site.

### Field ownership map (AP-Q7)

From `src/llm_judge/control_plane/field_ownership.py`:

| Capability | Owned envelope fields |
|------------|----------------------|
| CAP-1 | `dataset_registry_id`, `input_hash`, `benchmark_id`, `benchmark_version`, `benchmark_content_hash`, `benchmark_registration_timestamp` |
| CAP-2 | `rule_set_version`, `rules_fired` |
| CAP-5 | (empty frozenset — chain stamping only) |
| CAP-7 | (empty frozenset — chain stamping only) |

The empty frozensets are intentional ("chain stamping only" — capability records integrity but doesn't add typed fields), per the docstring.

`ProvenanceEnvelope.stamped(capability=..., **fields)` raises `FieldOwnershipViolationError` if any kwarg is absent from `FIELD_OWNERSHIP[capability]`. This is the runtime gate that converts End-State property A3.3 from convention to enforcement.

### Exception taxonomy (AP-Q8)

From `src/llm_judge/control_plane/types.py` and `batch_input.py`:

| Exception | Base | Purpose |
|-----------|------|---------|
| `MissingProvenanceError` | `Exception` | A wrapper's pre-check finds a required upstream stamp absent. |
| `ConfigurationError` | `Exception` | Required configuration missing/invalid for current mode (production without HMAC, etc.). |
| `GuardrailDeniedError` | `ConfigurationError` | **(L1-Pkt-B)** Guardrail substrate denied a capability invocation. Reserved namespace pattern for future subclasses (`RateLimitDeniedError`, `CircuitBreakerOpenError`, `KillSwitchActiveError`). Structural fix at L1-Pkt-C. |
| `BenchmarkFileNotFoundError` | `ValueError` | Benchmark JSON definition file not on disk (CP-F1 closure). |
| `BenchmarkContentCollisionError` | `ValueError` | Sidecar registration record's content hash mismatches SHA-256 of benchmark file (CP-F1 closure). |
| `FieldOwnershipViolationError` | `ValueError` | Capability stamps a field outside its `FIELD_OWNERSHIP` allowlist (CP-F3 closure). |
| `RubricNotInRegistryError` | `ValueError` | `rubric_store._resolve_version` asked for a rubric_id absent from `rubrics/registry.yaml`'s `latest:` map. |
| `BatchInputSchemaError` | `ValueError` | YAML/JSON batch input file fails Pydantic validation. |

All `ValueError`-based exceptions subclass for backward compatibility with callers using `except ValueError:` around adapter setup. The merge commit notes CP-F20 cluster expansion was accepted as part of L1-Pkt-B (Decision 8) — `GuardrailDeniedError` reserves the guardrail-denial namespace for the substrate.

### Phase 2 hardening status

L1 packets that have landed on master (`git log --grep="L1-Pkt"`):
- L1-Pkt-1 `9c145d4` (CP-F2 + CP-F11): HMAC mode + startup validation
- L1-Pkt-2 `994eaba` (CP-F4 + CP-F5 + CP-F14): combined Layer 1 cleanup
- L1-Pkt-A `85c7e5c` (CP-F1 + CP-F3): parallel-entry elimination + field-ownership runtime gate
- **L1-Pkt-B `b3eaf56` (CP-F9, CP-F12, partial CP-F8): substrate + timeout + capability registry**

Phase 2 progress per merge commit: **Category 1 complete (L1-Pkt-A), Category 2 complete (L1-Pkt-B)**. Remaining: L1-Pkt-C (Cat 3 exception taxonomy refactor + cleanup, CP-F10/F20/F22) and L1-Pkt-D (Cat 4 event vocabulary + governance preflight, CP-F7/F13).

**Mechanism class transitions delivered by L1-Pkt-B:**

| Decision | Before | After |
|----------|--------|-------|
| D1.1 (operational guardrails substrate) | missing | runtime gate |
| D1.2 (per-request timeout enforcement) | missing | runtime gate (post-completion denial; resource interruption deferred) |
| D2.1 (formal capability registry) | convention | runtime gate |
| D2.2 (orchestrator consumes registry) | convention | runtime gate |

### Test coverage (Layer 1)

| Test file | Tests |
|-----------|-------|
| `tests/control_plane/test_batch_aggregation.py` | sub-cap fire counts, success rate, percentiles, malformed events |
| `tests/control_plane/test_batch_input.py` | YAML/JSON load, round-trip, schema errors |
| `tests/control_plane/test_batch_runner_integration.py` | happy path, mixed outcomes, exception propagation, lifecycle ordering |
| `tests/control_plane/test_binding_enforcement.py` | rubric binding (math_basic, chat_quality regression) |
| `tests/control_plane/test_cap1_benchmark_provenance.py` | benchmark fields presence/absence, field-ownership compliance |
| `tests/control_plane/test_cap1_sub_capabilities.py` | CAP-1 sub-cap event firing + skip reasons |
| `tests/control_plane/test_cap2_sub_capabilities.py` | CAP-2 sub-cap instrumentation |
| `tests/control_plane/test_cap5_sub_capabilities.py` | CAP-5 sub-cap instrumentation |
| `tests/control_plane/test_capability_registry.py` | **(L1-Pkt-B)** registry membership, ordering, timeout values, `get_spec` lookup + KeyError |
| `tests/control_plane/test_capability_registry_gap_absence.py` | **(L1-Pkt-B)** no hard-coded sequence literal; multi-cap callers iterate registry |
| `tests/control_plane/test_configuration.py` | startup validation in production/dev modes |
| `tests/control_plane/test_envelope.py` | envelope stamping, integrity records |
| `tests/control_plane/test_envelope_hmac.py` | HMAC integrity preservation |
| `tests/control_plane/test_event_bus.py` | pub/sub semantics |
| `tests/control_plane/test_event_contracts.py` | event-payload schema |
| `tests/control_plane/test_field_ownership.py` | per-capability allowlist enforcement |
| `tests/control_plane/test_field_ownership_gap_absence.py` | absence assertion (no fields owned by unwrapped capability) |
| `tests/control_plane/test_no_parallel_orchestration.py` | CP-F1: parallel-entry elimination |
| `tests/control_plane/test_observability.py` | Timer, emit_event |
| `tests/control_plane/test_rubric_not_in_registry_error.py` | RubricNotInRegistryError exception path |
| `tests/control_plane/test_runner_bypass_rejection.py` | bypass attempts fail loudly |
| `tests/control_plane/test_runner_degradation.py` | degradation paths through capability failures |
| `tests/control_plane/test_runner_math_basic.py` | math_basic rubric integration |
| `tests/control_plane/test_runner_single_eval.py` | single-eval happy path |
| `tests/control_plane/test_sibling_failure.py` | CAP-2/CAP-7 sibling failure tolerance |
| `tests/control_plane/test_single_evaluation_request_benchmark_reference.py` | request shape with `benchmark_reference` |
| `tests/control_plane/test_types_request.py` | Pydantic validation on request types |
| `tests/control_plane/test_wrappers_version_resolution.py` | rubric version resolution in wrappers |
| `tests/control_plane/test_guardrails_substrate.py` | **(L1-Pkt-B)** substrate context manager + pre/post hook firing |
| `tests/control_plane/test_guardrails_substrate_gap_absence.py` | **(L1-Pkt-B)** mock `DenyAlwaysGuardrail` for CAP-1 actually denies; sibling-skip propagated; CAP-5 still ran (D5 contract preserved) |
| `tests/control_plane/test_guardrails_runner_integration.py` | **(L1-Pkt-B)** substrate hook firing across full CAP-1→CAP-2→CAP-7→CAP-5 sequence |
| `tests/control_plane/test_timeout_guardrail.py` | **(L1-Pkt-B)** Option β post-completion denial; `guardrail.timeout_exceeded` event emission |
| `tests/control_plane/test_timeout_guardrail_gap_absence.py` | **(L1-Pkt-B)** Option β semantics locked: CAP-1 timeout=0.1s, mock `invoke_cap1` sleeps 0.3s → denial fires AND wall-clock elapsed ≥ 0.25s (no interruption) |

**33 test files**, importing from `llm_judge.control_plane` **91 times** across the test suite — **the most-tested production package**. Site-wide test count grew 1071 → 1092 (+21 from L1-Pkt-B's 7 new files); coverage 83.67%.

---

## 6. External dependencies

### LLM provider APIs (AP-Q16, case-insensitive)

| Provider | Used in | Layer (best-fit) |
|----------|---------|------------------|
| **Gemini** | `llm_judge/main.py`, `llm_judge/llm_judge.py` (`GeminiAdapter`), `llm_judge/runtime.py`, `llm_judge/integrated_judge.py`, `llm_judge/benchmarks/funnel_analysis.py`, `llm_judge/benchmarks/generate_ground_truth.py`, `llm_judge/benchmarks/kg_extraction_experiment.py`, `llm_judge/benchmarks/slm_science_gate.py`, `llm_judge/benchmarks/nli_gemini_*.py` | 3 (CAP-7 L4) + various benchmark scripts |
| **OpenAI** | `llm_judge/main.py`, `llm_judge/runtime.py`, `llm_judge/llm_judge.py` (`OpenAIAdapter`), `llm_judge/llm_correctness.py` | 3 |
| **Groq** (via OpenAI-compatible API) | `llm_judge/llm_judge.py` (base_url switch in `OpenAIAdapter` factory), `llm_judge/main.py` | 3 |
| **Anthropic** | (none — 0 grep hits) | — |
| **Ollama** | (referenced in docstrings, served via OpenAI-compatible adapter) | 3 |

Env var keys observed: `GEMINI_API_KEY`, `OPENAI_API_KEY`, `LLM_API_KEY`, `LLM_API_BASE_URL`, `JUDGE_ENGINE`, `GATE2_ENGINE`, `GEMINI_MODEL`.

### Local model dependencies (AP-Q17)

| Pattern | Used in | Layer |
|---------|---------|-------|
| `SentenceTransformer` (sentence-transformers) | `properties/__init__.py` (`SentenceTransformerProvider`) | 3 (embedding provider) |
| `AutoTokenizer.from_pretrained` + `AutoModelForSeq2SeqLM` (MiniCheck) | `calibration/hallucination.py` | 3 (CAP-7) |
| `AutoModelForSequenceClassification` (NLI fallback, e.g. DeBERTa) | `calibration/hallucination.py`, multiple `benchmarks/*_science_gate.py` | 3 (CAP-7) |
| `AutoModelForCausalLM` (Qwen2.5-3B) | `benchmarks/slm_science_gate.py`, `benchmarks/slm_graphrag_science_gate.py` | benchmark scripts |
| `AutoModelForQuestionAnswering` (QG/QA pipeline) | `benchmarks/qgqa_science_gate.py` | benchmark scripts |
| `AutoModelForSeq2SeqLM` (QG model) | `benchmarks/qgqa_science_gate.py` | benchmark scripts |

The `benchmarks/` package contains many `*_science_gate.py` experiments; these are research/benchmarking scripts rather than production capability code.

### Test environment isolation

The `make test` target sets `TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 GEMINI_API_KEY=` to prevent network/model fetch during tests.

---

## 7. Known incompleteness

### Source TODOs (AP-Q18)

`grep -rn "TODO\|FIXME\|XXX" src/`:

| Location | Marker |
|----------|--------|
| `src/llm_judge/benchmarks/generate_ground_truth.py:795` | `# TODO: implement batch Gemini scoring` |

**That is the entire list across `src/`.** One TODO marker in the production source tree — exceptional cleanliness, suggesting either rigorous code-review hygiene or that incompleteness is tracked elsewhere (commit messages, ADRs, the F-number gap-list in the architecture map).

### F-number gaps (from ADR architecture map)

Tracked outside source comments — listed here for portal completeness:

| F-number | Layer | Description |
|----------|-------|-------------|
| F4 | 3 | CAP-2 `pattern_compilation` 0.0ms misrepresentation |
| F11 | 7 | Governance preflight doesn't validate prompt artifact existence |
| F12 | 4 | L5 `adjudication.py` exists but unwired |
| F13 | 3 | `DEFAULT_LAYERS=("L1",)` — cascade runs L1-only by default |
| F14 | 3 | `_l3_minilm_gate_check` is misnamed |
| F16 | 3/7 | ADR-0027 `fact_counting` default flip never landed |
| F17 | 3 | L2 silent no-op (no caller passes `fact_tables=`) |
| F18 | 3 | L4 prompt never experimentally validated |

### CP-F findings closed in last 30 days (master)

| Finding | Closed in | Notes |
|---------|-----------|-------|
| CP-F1 (parallel-entry elimination) | L1-Pkt-A `85c7e5c` | |
| CP-F2 (HMAC mode) | L1-Pkt-1 `9c145d4` | |
| CP-F3 (field-ownership runtime gate) | L1-Pkt-A `85c7e5c` | |
| CP-F4, CP-F5, CP-F14 | L1-Pkt-2 `994eaba` | |
| CP-F8 (operational guardrails substrate) | L1-Pkt-B `b3eaf56` | **Partial** — substrate complete; rate limits / circuit breakers / kill switch deferred to future packets that plug into this substrate. |
| CP-F9 (capability registry / hard-coded sequence) | L1-Pkt-B `b3eaf56` | Closed; orchestrator now iterates `CAPABILITY_REGISTRY`. Drive-by fix: `batch_aggregation.VERTICAL_CAPABILITIES` derived from registry. |
| CP-F11 (startup validation) | L1-Pkt-1 `9c145d4` | |
| CP-F12 (per-request timeout enforcement) | L1-Pkt-B `b3eaf56` | **Partial in resource sense, full in observability sense** — Option β post-completion denial (no resource interruption). Capability runs to completion regardless of timeout; denial event surfaces post-completion with structured context. Resource-interruption gap recorded as **CP-F23 candidate** for v1.5. |
| CP-F20 cluster (exception namespace) | partial in L1-Pkt-B | `GuardrailDeniedError(ConfigurationError)` namespace reserved (Decision 8); structural fix at L1-Pkt-C. |

---

## 8. User-facing surface

`Makefile` phony targets, grouped by audience:

### Demonstrations
- `make demo` — single-case chat_quality walkthrough
- `make demo-batch` — full RAGTruth-50 batch run (~7–10 min)
- `make demo-batch-quick` — first 10 RAGTruth-50 cases (~1–2 min)
- `make demo-batch-benchmark` — benchmark-driven batch
- `make demo-batch-file` — file-driven batch
- `make verify-l1` — L1-isolated batch + verification report

### Daily eval workflow
- `make eval` — composed: `pr-gate` + `baseline-dry-run` + `registry-list`
- `make pr-gate` — PR gate evaluation against `configs/runspecs/pr_gate.yaml`
- `make diff` — diff latest PR-gate run vs baseline
- `make baseline-validate` — validate baseline integrity
- `make baseline-dry-run` — policy-gated promotion check (no writes)
- `make baseline-promote` — actual promotion (writes snapshot + updates `latest.json`)

### Hygiene / preflight
- `make preflight` — composed: `lint` + `typecheck` + `test` + `rules-validate` + `baseline-validate` + `pr-gate` + `baseline-dry-run` + `registry-list` + `drift-check`
- `make lint` — `ruff check . --fix`
- `make typecheck` — `mypy .`
- `make test` — `pytest -q` with offline transformers/HF + cleared GEMINI_API_KEY
- `make install` — `poetry install --no-interaction --no-root`
- `make clean` — remove generated PR-gate run artifacts

### Run registry / observability
- `make registry-list` — list recent runs
- `make registry-show RUN_ID=...` — show one run entry
- `make registry-trend` — metric trend across recent runs
- `make drift-check` — drift detection
- `make rules-list`, `rules-validate`, `rules-export` — rule lifecycle ops

### Git workflow
- `make git-start`, `git-ship`, `git-merge` — branch lifecycle helpers (`git-ship` depends on `preflight`)

### Docker
- `make docker-build`, `docker-deploy`, `docker-status`, `docker-validate`, `docker-down`, `docker-reset`

### RAGTruth-50 calibration
- `make preseed` — seed graph cache from Exp 31 fact tables
- `make benchmark` — runs `demo-batch-quick`
- `make funnel` — print funnel from last benchmark run
- `make calibrate` — composed: `preseed` + `demo-batch-quick` + `funnel`

### Mode environment variables
- `LLM_JUDGE_MODE` — `development` (default) | `production`
- `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` — required in `production` mode

---

## 9. v1.0 status

*Architect chat editorial — synthesized from §1-§8 portal data + v1.0 scope document (`docs/decisions/v1.0-scope-document.md`).*

### What v1.0 ships

The end-of-week v1.0 demo is grounded in three operational capabilities and the orchestration substrate that governs them:

**Substrate (Layer 1) — operational and hardened.** Phase 1 + Phase 2 Category 1 closed: HMAC integrity (CP-F2), startup validation (CP-F11), parallel-entry elimination (CP-F1), field-ownership runtime gate (CP-F3), exception taxonomy cleanup (CP-F4, CP-F5, CP-F14). Reproducibility narrative for the demo builds on this hardening — the audit trail (§5) records every capability invocation with provenance.

**Capabilities (Layer 2 + 3 + 5) — operational for the demo path.**
- CAP-1 (Dataset Governance): operational with sub-cap instrumentation. Benchmark-as-governed-artifact closed via CP-F1.
- CAP-2 (Rule Engine): operational with sub-cap tests. Produces deterministic rule verdicts.
- CAP-7 (Hallucination Detection): L1 instrumentation operational. **L2-L5 layers have F-numbered gaps** (F13 default cascade L1-only, F17 L2 silent no-op, F12 L5 unwired) — this is v1.0's critical path remaining work for Tuesday-Wednesday.
- CAP-5 (Artifact Governance): operational. Manifests + lineage records + integrity trail.

**Demo surface (Layer 6 + 8) — operational, expanded this week.** 28 make targets exist (§8). For v1.0, the demo flow combines `make demo`, `make demo-batch-quick`, and `make verify-l1` plus the new post-batch report module (Wednesday deliverable).

### What v1.0 critical path requires this week

Sequenced against the v1.0 scope document's day-by-day plan:

| Day | Critical-path work | Portal-data anchor |
|---|---|---|
| **Mon** | L1-Pkt-A confirmation (already merged at `85c7e5c`); 28-metric reproducibility audit by QA chat | §5 confirms L1-Pkt-A on master; §3 confirms CAP-7 cascade has F-numbered gaps |
| **Tue** | Wire CAP-7 L4 (Gemini API) + L5 (adjudication); verify CAP-2 + CAP-7 match experimental baseline | §3 capability inventory shows CAP-7 partial (L1 only, L2-L5 gaps); §6 confirms Gemini integration exists in `main.py`/`runtime.py` (not yet wired into CAP-7 cascade) |
| **Wed** | Build post-batch report module computing 28 property metrics from `BatchResult` | §3 + §10 surface 5 of 11 capabilities operational; report module aggregates metrics from existing capability outputs |
| **Thu** | Build demo CLI/notebook surface; curate 3-5 demo examples | §8 shows existing `make demo*` targets as foundation; new CLI extends rather than replaces |
| **Fri** | QA dry-run + demo runbook + final ship | QA chat owns final go/no-go |

### What v1.0 explicitly defers to v1.1+

Portal data confirms what's NOT v1.0-blocking:

**Architectural quality work paused this week:**
- L1-Pkt-B (capability registry CP-F9; substrate timeout CP-F8/CP-F12) — branch exists (`layer-1/l1-pkt-b-substrate-timeout-registry`), not on master. Resumes post-v1.0.
- L1-Pkt-C (exception taxonomy refactor — CP-F10, CP-F20, CP-F22) — not started.
- L1-Pkt-D (event vocabulary + governance preflight — CP-F7, CP-F13, CP-F18) — not started.

**Capability gaps deferred:**
- CAP-3 (Rule Governance): absent from code. Future packet.
- CAP-4 (Baseline Governance): partial (`eval/baseline.py` without capability stamping). Future packet.
- CAP-6 (Drift Monitoring): stub only. Requires time-series data not available at v1.0.
- CAP-9 (Platform Health): absent from code. Future capability.
- CAP-10 (Calibration & Adjudication): stub only. F12 (adjudication unwired) is not v1.0-blocking — L5 wiring for CAP-7 specifically is on Tuesday's critical path; broader CAP-10 capability is deferred.
- CAP-11 (Reporting): not yet filed. Components exist in `tools/`, `experiments/`. Post-v1.0 work.

**Architecture map refinements deferred:**
- Rubric governance Layer 7 → Layer 2 migration (architectural debt, not v1.0-blocking).
- Cross-layer feedback loops (Layer 4 → Layer 3 absent per ADR).
- Per-layer formal capability ownership (Layers 6, 7, 8 currently operational without capability registration).

### v1.0 trustworthiness — what makes the demo defensible

Per the v1.0 scope document's expanded definition of "trustworthy" (three layers):

**Layer 1 (orchestration trustworthiness):** Achieved via Phase 1 + Phase 2 Category 1 closures on master. Audit trail records every CAP invocation with provenance (§5 evidence: 26 test files, HMAC integrity preservation tests, field-ownership gap-absence tests).

**Layer 2 (capability trustworthiness):** Tuesday's verification work — CAP-2 + CAP-7 results matching experimental baseline. QA chat owns the protocol; Layer 1 chat verifies substantive correctness.

**Layer 3 (helper function trustworthiness):** Reproducibility configuration for the 28 property check functions preserved as library per L1-Pkt-A Option (c). External LLM calls (Gemini, OpenAI) need temperature=0 + caching for demo consistency. Local model calls (MiniCheck, NLI fallback, SentenceTransformer) need explicit seeding.

The reproducibility-and-integrity narrative for the executive demo: same input → same output, with audit trail visible. Not a metrics dashboard; a concrete demonstration that the platform's outputs are deterministic and accountable.

### Open questions surfacing from portal data

These are not v1.0-blocking but worth flagging for post-v1.0 strategic planning:

1. **Capability registry placement.** CP-F9 closure (capability_registry.py introduction) is on the L1-Pkt-B branch but not on master. Should the registry concept land before strategic capability expansion (CAP-3, CAP-4, CAP-6, CAP-9, CAP-11)? Likely yes — the registry is the canonical source of capability identity for future expansion.

2. **CAP-7 cascade completion.** L4 + L5 wired this week for v1.0. F13/F17/F18 represent deeper cascade gaps; do these become priorities post-v1.0 or do current L4 + L5 wiring satisfy the demonstration need?

3. **Layer 6 / CAP-11 spawn timing.** Reporting capability is "committed not yet filed." Components exist in `tools/` and `experiments/`. Post-v1.0 demo feedback may make CAP-11 a high priority if executives or QA folks request reporting features.

4. **Architectural debt: rubric governance Layer 7 → Layer 2.** Documented in ADR but not scheduled. Resolves itself in proportion to how much rubric work happens; if v1.0 demo doesn't surface rubric authoring needs, this stays deferred.

These questions are for the post-v1.0 strategic planning cycle, not this week's work.

---

## 10. Surprises and gaps

The following items deviate from the brief's expectations or are otherwise worth flagging:

### Resolved since prior portal generation

1. **~~`capability_registry.py` is missing on master~~ — RESOLVED.** Prior portal generation (against master HEAD `85c7e5c`) flagged this as the hard surprise. L1-Pkt-B (`b3eaf56`) merged subsequently and is now on master. Registry contents reported in §3 and §5; orchestrator iterates it via Option A dispatch-by-id; gap-absence test prevents future hard-coded sequence regressions.

### Capability presence vs ADR specification

2. **Of 11 capabilities specified in the architecture map, 4 are formally registered + runtime-orchestrated** (CAP-1, CAP-2, CAP-7, CAP-5 in `CAPABILITY_REGISTRY`), **1 is referenced as a tuple-constant only** (CAP-10 in `HORIZONTAL_CAPABILITIES`), and **CAP-6 likewise appears only as a constant** (`HORIZONTAL_CAPABILITIES`). CAP-3, CAP-4, CAP-8, CAP-9, CAP-11 are absent from `src/` as capability-registered units. This is consistent with ADR-0007 (Deploy L1-L2-L3, defer L4-L5) but worth surfacing — the layer model's capability inventory is mostly aspirational. CAP reference frequencies in src (post L1-Pkt-B): CAP-1 (53), CAP-7 (49), CAP-5 (47), CAP-2 (45), CAP-6 (2), CAP-10 (2).

3. **CAP-6 and CAP-10 referenced only as a tuple constant**: `control_plane/batch_aggregation.py:29` declares `HORIZONTAL_CAPABILITIES = ("CAP-6", "CAP-10")` but no wrappers stamp `capability_id="CAP-6"` or `capability_id="CAP-10"` anywhere. Their "presence" in code is a string literal, not an instrumented capability.

### Layer / package alignment

4. **Layer 1 is the only layer with clean package alignment** (`control_plane/`). All other layers map to multiple packages, or share packages, or depend on `Makefile`+`tools/`+`experiments/` outside `src/`:
   - Layer 3 (CAP-7) lives in `calibration/` alongside Layer 4 (CAP-10) work
   - Layer 5 (CAP-5) lives in `eval/` with parts in `control_plane/observability.py` + `event_bus.py`
   - Layer 6 lives entirely in `Makefile`, `tools/`, `experiments/` — no `src/llm_judge/` package
   - Layer 7 lives partly in `rubrics/`, partly in top-level `governance.py`/`rubric_store.py`
   - Layer 8 lives in `Makefile` + `README.md` + `tools/demo_platform.py`

5. **Rubric governance lives in Layer 7 per ADR but architectural debt notes it should be in Layer 2.** This is documented in the ADR (§Layer 7 architectural inconsistency).

### Source content surprises

6. **Single TODO marker across the entire `src/` tree.** `grep -rn "TODO\|FIXME\|XXX" src/` returns one hit (`benchmarks/generate_ground_truth.py:795`). Either incompleteness is tracked outside source (ADR F-numbers, commit messages, packet briefs) or hygiene is unusually rigorous. The F-number list in the architecture map confirms incompleteness is real — it's just not annotated in code.

7. **Brief asked about `src/llm_judge/types.py`** for AP-Q8 exception taxonomy. **No such file exists.** Top-level types live in module-specific files (`schemas.py`, `dataset_models.py`, etc.); the file the brief was thinking of is `src/llm_judge/control_plane/types.py`. Exception taxonomy reported in §5 from that file plus `batch_input.py`.

### Auto-clustering does not match capability layers (expected, not flagged as surprise)

8. The 14 communities the Leiden algorithm produced (`unit-returns` 1034 nodes, `benchmarks-load` 980, `experiments-sentence` 378, `control-plane-runner` 316, etc.) cluster mostly by directory rather than by capability or layer. The largest community (`unit-returns`, 1034 nodes) has cohesion 0.19; the most cohesive community (`rules-plan`) has 2 nodes and cohesion 0.13. **Expected per brief; mentioned for completeness.**

### ADR document state

9. **`docs/adr/layer-1-external-contract-v3.3.1.md` exists alongside `layer-1-external-contract-v3.3.md`.** v3.3.1 looks newer but isn't tracked in git status. Architect chat may want to clarify which is canonical for §5 in future portal regenerations.

10. **Several ADR documents have `:Zone.Identifier` siblings** (Windows-origin metadata files): `adr-platform-8-layer-architecture.md:Zone.Identifier`, `architecture-map-8-layer-specification.md:Zone.Identifier`, `brief-template-v1.3.md:Zone.Identifier`, etc. These are noise from Windows file transfer. Worth a `.gitignore` entry.

### Cross-community edges

11. **The graph detected exactly one cross-community edge cluster** (16 CALLS edges, `tests/calibration/test_layer_verification_renderer.py` → `experiments/render_layer_verification_report.py`). This is the Phase 1 verification renderer being tested — appropriate, not architecturally suspicious, but the only inter-community coupling the algorithm flagged. It does suggest the auto-clustering's communities are unusually self-contained (or that the algorithm couldn't find structure in the rest of the codebase).

### Activity heatmap (last 30 days)

12. **Top-touched directories**: `experiments/` (310 churn entries, dominated by Phase 1 verification work), `src/llm_judge/benchmarks/` (74), `tests/unit/` (69), `tests/control_plane/` (45), `src/llm_judge/control_plane/` (38), `docs/adr/` (37), `src/llm_judge/calibration/` (31), `tools/` (30). Layer 1 hardening is the dominant theme; no recent activity on CAP-3, CAP-4, CAP-9, CAP-11 implementation.

---

## Generation provenance

- **Tool versions/state (refreshed):** code-review-graph index timestamp `2026-05-05T05:24:36`; 322 files, 3,299 nodes, 30,453 edges; incremental rebuild diffed against `85c7e5c` after `git pull` brought master to `b3eaf56`. Embeddings: 0 (semantic search uses keyword fallback).
- **Inputs read:**
  - `docs/adr/architecture-map-8-layer-specification.md` (layer specification)
  - `src/llm_judge/control_plane/field_ownership.py` (full)
  - `src/llm_judge/control_plane/types.py` (full)
  - `.code-review-graph/wiki/index.md`, `.code-review-graph/wiki/control-plane-runner.md`
  - `Makefile` (full)
  - Architecture overview JSON via jq filter
  - `git log --since="30 days ago" --name-only --oneline`
  - `grep -rn "CAP-[0-9]" src/`
  - `grep -ri "gemini\|openai\|anthropic" src/`
  - `grep -ri "from_pretrained\|SentenceTransformer" src/`
  - `grep -rn "TODO\|FIXME\|XXX" src/`
  - Per-package import graph via `grep -hr "^from llm_judge\."`
- **Queries that returned data:** AP-Q1, Q2, Q3, Q4 (resolved post-refresh via `capability_registry.py`), Q5 (partial via wiki), Q6 (resolved post-refresh — registry now on master), Q7, Q8, Q9 (partial), Q10 (partial), Q11, Q12, Q13, Q14, Q15 (test-import counts), Q16, Q17, Q18.
- **Queries that fell back / partial:** AP-Q4 was registry-fallback at original generation; AP-Q6 was reported missing. **Both resolved in this refresh** after L1-Pkt-B verified on master.
- **Refresh delta (this regeneration):** §1 overview, §2 Layer 1 entry, §3 capability inventory (added registered/timeout columns), §5 module inventory + §5 capability registry (AP-Q6) + §5 exception taxonomy (added `GuardrailDeniedError`) + §5 Phase 2 status + §5 test-coverage table (added 7 L1-Pkt-B tests), §7 CP-F findings (added CP-F8/F9/F12 closures + CP-F20/F23 notes), §10 surprise #1 marked resolved + §10 surprise #2 capability counts updated. **§9 v1.0 status preserved verbatim per refresh instruction; see refresh report for §9 contradictions to flag.**
- **Suggested next-step queries** (architect chat may consider):
  - Quantify CAP-7 cascade actual code paths: trace `calibration/hallucination.py` callers/callees to confirm which layers (L1–L5) are reachable from `runner.py`.
  - Per-capability test-coverage histogram: which CAP-N has thinnest test coverage relative to lines of capability code?
  - Identify any direct capability invocations bypassing wrappers (graph query: callers of CAP-stamped functions that are not in `control_plane/`).
  - Confirm whether `:Zone.Identifier` files in `docs/adr/` should be `.gitignore`d.
  - Decide canonical version for `layer-1-external-contract-v3.3*.md` files.
