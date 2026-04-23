# Packet CP-1 — Turn 1 Reconnaissance

Date: 2026-04-22
Branch: feat/packet-cp-1-control-plane
Commit: ce9db6a170130ece055a5d8b943df881b05ad780
Test baseline: **Unable to reproduce the claimed 756.** Under the
pyproject.toml default `addopts` (`--cov=src/llm_judge --cov-report=
term-missing --cov-fail-under=80`), `pytest --collect-only -q` exits
with an error — `pytest-cov` plugin is not installed in this
environment. With `-o addopts=""` and `PYTHONPATH=src`, collection
yields **278 tests collected, 40 errors** (all errors are
`ModuleNotFoundError` on third-party deps such as `pydantic` — the
project dependencies are not installed on this machine, not a code
defect). Ground truth for this branch on this host: no runnable
test baseline; environment is not provisioned. The architect should
confirm the 756 claim against a freshly provisioned environment
before Turn 2 treats any test count as an invariant.

Read-only reconnaissance: no capability modules modified, no code
written outside `docs/design/`.

## 1. CAP-5 entry point

**Verdict: PARTIALLY PRESENT — no single entry; CAP-5 is composed
from three functions called inline inside `eval/run.py`.**

Candidates identified by grepping `def write_/register_/persist_/
append_` and `manifest|artifact`:

1. **`eval/registry.py:85 append_run_registry_entry`** — the
   best-shaped governed entry.
   - Writes an immutable `RunRegistryEntry` (dataclass with
     `schema_version = "1.0"`, registry.py:12) as one JSONL line to
     `state/run_registry.jsonl` (registry.py:85–129).
   - Carries governance keys: `run_id, dataset_id, dataset_version,
     rubric_id, judge_engine, cases_total, cases_evaluated, sampled,
     dataset_hash, run_dir, metrics, git_sha` (registry.py:16–44).
   - Append-only, auditable.

2. **`eval/event_registry.py:121 append_event`** — the governed
   cross-capability event log.
   - Writes a `GovernanceEvent` (schema_version "1.0",
     event_registry.py:37) to `state/event_registry.jsonl`.
   - Enforces allow-listed event types `eval_run,
     baseline_promotion, rule_change, dataset_registration,
     drift_alert, drift_response` (event_registry.py:41–50;
     `ValueError` at line 135–139 on unknown type).
   - Retry-on-fail with exponential backoff (event_registry.py:86–
     118).

3. **`eval/io.py:49 build_manifest`** — helper that returns a manifest
   dict; does NOT persist it (io.py:49–63). The actual write is
   `eval/run.py:460` via `write_json`. Not a CAP-5 entry function.

4. **`eval/io.py:39 write_json` / `io.py:43 write_jsonl`** — untyped,
   ungoverned primitives. Not a governance entry (no schema, no
   registration).

Recommendation for Turn 2: treat `append_run_registry_entry` as the
**primary CAP-5 entry** and `append_event(event_type="eval_run", …)`
as a **companion event-stamp** that must be paired with it. This
mirrors the orchestration already present in `eval/run.py:413–448`.

**Composition finding the architect should see:** there is no
single function today that "the Runner calls to persist an
evaluation artifact." The three writes (validation_report.json,
judgments/metrics + registry entry, event) are threaded through
`eval/run.py`. A Runner written in Turn 2 must either (a) invoke
all three in sequence, duplicating the orchestration, or (b) the
architect decides to extract a CAP-5 entry shim. Turn 1 does not
propose which.

## 2. Capability entry signatures

### CAP-1 Dataset Governance

- **Entry:** `src/llm_judge/datasets/registry.py:56` —
  `DatasetRegistry.resolve`
- **Signature:** `resolve(*, dataset_id: str, version: str) ->
  ResolvedDataset` (registry.py:56)
- **Returns:** `ResolvedDataset(metadata: DatasetMetadata,
  dataset_dir: Path)` — frozen dataclass at registry.py:24–31; has
  `.data_path` property. `DatasetMetadata` schema at
  `datasets/models.py:8–30` (fields: `dataset_id, version,
  schema_version, data_file, owner, license, task_type,
  content_hash`).
- **Raises:**
  - `FileNotFoundError` — missing `dataset.yaml` (registry.py:60);
    missing data file (registry.py:75).
  - `ValueError` — dataset_id mismatch (registry.py:65–68); version
    mismatch (registry.py:69–72); content-hash mismatch
    (registry.py:81–88); parse failure (registry.py:117–123);
    integrity-check failure (registry.py:128–139).
  - `ImportError` caught silently (registry.py:153–159) — validator
    optional.
- **Side effects:** reads `dataset.yaml` and the data file; logs
  warnings on security findings (registry.py:143–152); no network;
  no file writes.

### CAP-2 Rule Engine

- **Entries (three related):**
  - `src/llm_judge/rules/engine.py:208` — `load_plan_for_rubric(
    rubric_id: str, version: str) -> RulePlan` (config loader).
  - `src/llm_judge/rules/engine.py:144` — `RuleEngine(rules).run(
    ctx: RuleContext) -> list[Any]` (engine core).
  - `src/llm_judge/rules/engine.py:261` — `run_rules(ctx:
    RuleContext, plan: RulePlan) -> RuleResult`-like (convenience
    wrapper; this is the most natural CAP-2 entry for a Runner).
- **Signature:** `run_rules(ctx: RuleContext, plan: RulePlan) ->
  Any` where returned object has a `.flags: list[Flag]` attribute
  (engine.py:261–281).
- **Returns:** Typically `RuleResult(flags=list[Flag])`
  (rules/types.py:19–21); falls back to a dynamic `_RR` wrapper
  (engine.py:276–280). `Flag` shape: `id, severity: "weak"|"strong",
  details: dict, evidence: list[str]` (types.py:11–16).
- **Raises:** **Swallows all rule-apply exceptions silently**
  (engine.py:161–164, 167–170 — "deterministic judge must never
  crash due to unknown rules"). `load_plan_for_rubric` can raise
  `FileNotFoundError` if the config YAML is missing; parse errors
  propagate. This is a governance concern worth the architect's
  attention: a bypass in rule code is non-loud.
- **Side effects:** mutates `ctx.flags` by appending normalized
  string flags (engine.py:198–203); `load_plan_for_rubric` reads
  `configs/rules/<rubric_id>_<version>.yaml` (engine.py:215) and
  queries `rules.lifecycle.get_deprecated_enforced_rules()`
  (engine.py:221–223); emits a log event `rule.excluded.deprecated`
  (engine.py:239–243). No network.

### CAP-7 Evaluation

- **Entry:** `src/llm_judge/calibration/hallucination.py:548` —
  `check_hallucination`.
- **Signature:** `check_hallucination(*, response: str, context:
  str, source_context: str = "", case_id: str = "unknown",
  grounding_threshold: float = 0.8, min_sentence_threshold: float =
  0.3, similarity_threshold: float = 0.6, max_ungrounded_claims:
  int = 2, skip_embeddings: bool = False, gate2_routing: str =
  "none", layered: bool = True, fact_tables: dict | None = None,
  knowledge_graphs: dict | None = None, l1_enabled: bool = True,
  l2_enabled: bool = True, l3_enabled: bool = True, l4_enabled:
  bool = True, config: PipelineConfig | None = None) ->
  HallucinationResult` (hallucination.py:548–600).
- **Returns:** `HallucinationResult` dataclass (hallucination.py:
  132–146): `case_id, risk_score, grounding_ratio, min_sentence_sim,
  ungrounded_claims, unverifiable_citations, gate1_decision,
  gate2_decision, flags: list[str], layer_stats: dict[str,
  int|float], sentence_results: list[SentenceLayerResult]`.
- **Raises:** no declared exceptions at the top-level signature.
  Internal layers can raise (e.g. spaCy load failure is caught and
  logged at hallucination.py:69–79; MiniCheck/DeBERTa loads may
  raise at runtime). L4 Gemini calls can raise network errors.
- **Side effects:** may load spaCy / MiniCheck / DeBERTa NLI models
  (GB-scale, first-call lazy); may hit the Gemini API for L4
  (hallucination.py:449); may read the graph cache (`.cache/
  hallucination_graphs/`); logs warnings. No registry writes.

### CAP-5 Artifact Governance (candidate)

- **Entry (primary):** `src/llm_judge/eval/registry.py:85` —
  `append_run_registry_entry`.
- **Signature:** `append_run_registry_entry(*, registry_path: Path
  = DEFAULT_REGISTRY_PATH, run_dir: Path, manifest: dict, metrics:
  dict, cases_total: int, cases_evaluated: int, sampled: bool,
  dataset_id: str, dataset_version: str, rubric_id: str,
  judge_engine: str, dataset_hash: str) -> None` (registry.py:85–
  99).
- **Returns:** `None`.
- **Raises:** I/O errors on the append (no retry, unlike
  `event_registry._append_with_retry`). Does not validate
  `run_dir` exists.
- **Side effects:** creates parent directory
  (`state/run_registry.jsonl`'s folder) if absent (registry.py:
  127); opens the file in append mode and writes one JSON line
  (registry.py:128–129). No network.
- **Companion entry (event stamp):** `src/llm_judge/eval/
  event_registry.py:121` — `append_event(*, event_type: str,
  source: str, actor: str = "system", related_ids: dict[str, str]
  | None = None, payload: dict[str, Any] | None = None,
  registry_path: Path = DEFAULT_EVENT_REGISTRY_PATH) ->
  GovernanceEvent | None`. Raises `ValueError` on unknown
  `event_type` (event_registry.py:135–139); returns `None` if all
  retries exhausted (event_registry.py:154–159).

## 3. SingleEvaluationRequest proposed shape

Packet-suggested minimum: `response, source, request_id?, caller_id?`.

Reconciliation with Phase 2 shows this 4-field set is sufficient
for CAP-7 but **not** for CAP-1, CAP-2, or CAP-5. Per-capability
requirements:

- **CAP-1 Dataset Governance** consumes `dataset_id: str, version:
  str` (registry.py:56). It does NOT consume a `response`/`source`
  pair. For a single ad-hoc instance (not a registered dataset),
  CAP-1's normal entry cannot meaningfully engage. See §4 for
  composition consequences.
- **CAP-2 Rule Engine** consumes a `RuleContext` built from a
  `PredictRequest` (which itself needs `conversation: list[Message],
  candidate_answer: str, rubric_id: str`) plus a `RulePlan` derived
  from `(rubric_id, version)` (engine.py:208, types.py:24–67). So
  CAP-2 needs `rubric_id: str` and `rubric_version: str` in scope,
  plus enough information to synthesize a `PredictRequest`.
- **CAP-7 Evaluation** consumes `response: str` and `context: str`
  (optionally `source_context: str`) (hallucination.py:548–571).
  `response` and `source` map directly.
- **CAP-5 Artifact Governance** consumes the full governance key
  set (dataset_id, dataset_version, rubric_id, judge_engine,
  dataset_hash, cases_total/evaluated, run_dir, manifest, metrics)
  — registry.py:85–99.

Proposed minimum field set for SingleEvaluationRequest (every field
justified by at least one capability; no speculative fields):

| Field | Type | Required | Consumed by | Evidence |
|-------|------|----------|-------------|----------|
| `response` | `str` | yes | CAP-7 (as `response`) | hallucination.py:548 param `response: str` |
| `source` | `str` | yes | CAP-7 (as `context` / `source_context`) | hallucination.py:550 param `context: str`, 552 param `source_context: str` |
| `rubric_id` | `str` | yes | CAP-2 (`load_plan_for_rubric`), part of CAP-5 governance keys | engine.py:208; registry.py:96 |
| `rubric_version` | `str` | yes | CAP-2 (`load_plan_for_rubric`) | engine.py:208 param `version` |
| `request_id` | `Optional[str]` | no | Runner assigns if absent; stamped into `related_ids` for CAP-5 event | event_registry.py:74 (related_ids channel) |
| `caller_id` | `Optional[str]` | no | CAP-5 event `actor` field | event_registry.py:122 param `actor` |

**Fields the architect must decide on before Turn 2** (not included
above; each is a composition question, not an obvious addition):

- `dataset_id` / `dataset_version`: CAP-1 cannot be invoked without
  these. Options the architect must choose between —
  - (a) add them and require SingleEvaluationRequest callers to
    register their input as a dataset first (heavyweight for "one
    instance"),
  - (b) skip CAP-1 for this request shape and stamp
    `dataset_registry_id="ad_hoc"` in the envelope (CAP-1 BYPASSED
    by design for this shape — documented bypass, not silent), OR
  - (c) defer CAP-1 integration to a future SingleEvaluationRequest
    variant.
- `judge_engine`: CAP-5's `append_run_registry_entry` requires it.
  Runner can derive from a config / environment, but this is a
  decision, not a field.
- `dataset_hash`: same as `judge_engine` — CAP-5 governance key
  that has no obvious source for an ad-hoc input unless the Runner
  computes `sha256(response + source)`.

## 4. Composition assessment

| Pair | Verdict | Evidence / gap description |
|------|---------|----------------------------|
| CAP-1 → CAP-2 | ADAPTABLE (in the multi-case eval/run.py flow); BLOCKED for SingleEvaluationRequest | `resolve` returns `ResolvedDataset` (path to a file of many rows); `run_rules` needs a single-case `RuleContext` per row. In `eval/run.py:182–185, 314–324` this is bridged by reading the JSONL file and building a `PredictRequest` per row. For a SingleEvaluationRequest (one instance), there is no dataset to resolve in the first place — CAP-1 does not operate on instances. Wrapper-bridging doesn't help; the capability's unit-of-work is a dataset, not a request. Architect decision. |
| CAP-2 → CAP-7 | BLOCKED as an ordered pair; CLEAN as siblings | `run_rules` returns `RuleResult(flags: list[Flag])` (types.py:19–21). `check_hallucination` takes `response: str, context: str` — it does not consume `RuleResult` or flags. The two are sibling evaluators with orthogonal inputs, not an upstream/downstream pair. In today's platform their outputs are combined by the judge engine (see `eval/run.py:326` `engine.evaluate(req)`) and by `IntegratedJudge`, not by chained handoff. Wrapper cannot reshape output X into input Y when they are about different things. |
| CAP-7 → CAP-5 | ADAPTABLE | `HallucinationResult` (risk_score, flags, layer_stats, sentence_results) does not map directly to `append_run_registry_entry`'s governance keys (dataset_id, dataset_version, rubric_id, judge_engine, dataset_hash, cases_total/evaluated, metrics). The Runner holds the envelope stamps and synthesizes `metrics` from the `HallucinationResult` fields. No capability change required; the wrapper is the adapter. |

## 5. Envelope stamp viability

| Stamp | Source CAP | Viability | Notes |
|-------|-----------|-----------|-------|
| `dataset_registry_id` | CAP-1 | DIRECT if CAP-1 engaged; N/A for ad-hoc | `ResolvedDataset.metadata.dataset_id` exists (registry.py:24–31). If SingleEvaluationRequest does not go through the registry (no `dataset_id` in its shape), this stamp cannot come from CAP-1. Architect must decide: stamp as `"ad_hoc"` sentinel, require dataset registration, or drop the stamp. |
| `input_hash` | CAP-1 | WRAPPER-COMPUTED | CAP-1's `content_hash` hashes the **whole dataset file** at registration (registry.py:81–88). There is no per-instance input hash computed by CAP-1. For a single-request input, the Runner would compute `sha256(canonical_bytes(response, source))` itself. No capability change. |
| `rule_set_version` | CAP-2 | DIRECT | `RulePlan.version` (engine.py:30–33) and `RulePlan.rubric_id` are produced by `load_plan_for_rubric` (engine.py:208–258). Wrapper reads `plan.version`. |
| `rules_fired` | CAP-2 | DIRECT | `run_rules` returns an object with `.flags: list[Flag]`; each `Flag` has `.id` (types.py:11–16). `rules_fired` = the flag ids list. |
| `prompt_version` | CAP-7 | BLOCKED | `check_hallucination`'s signature (hallucination.py:548–571) neither accepts nor returns a `prompt_version`. The L4 Gemini prompt is built ad-hoc inside `_l4_gemini_check` (hallucination.py:449) with no exposed version identifier on the function's contract. A `prompt_version` field DOES exist but elsewhere — on `JudgeMeta` in the judge registry (`calibration/__init__.py:77`), which is used by `CalibratedJudge`, which the hallucination pipeline is not wrapped in. Choices for Turn 2: (a) wrapper-inject a hard-coded constant (e.g. `"hallucination-pipeline-v1"`), (b) require CAP-7 to expose a prompt-version identifier (capability change), or (c) drop the stamp for this packet. |

## 6. Name collision check

No existing `PlatformRunner`, `Runner`, or `Orchestrator` classes in
`src/llm_judge/`. Grep match: `src/llm_judge/calibration/
pipeline_config.py:124 class PipelineConfig` (a dataclass holding
L1–L4 layer enables and thresholds) and
`pipeline_config.py:178 class PipelineConfigError(ValueError)`. A
new `PlatformRunner` would not collide; but "Pipeline" as a term is
already in use for the hallucination cascade's layer configuration
— the architect may want to pick a name that distinguishes
platform-level orchestration from pipeline-level cascade
configuration.

Related: `benchmarks/runner.py` has a module-level `run_benchmark`
function (runner.py:742) that orchestrates benchmark adapters; no
`Runner` class. Not a collision, but an adjacent naming.

## 7. Headline finding

**GO-WITH-ADJUSTMENTS** for Turn 2.

The three adjustments the architect must decide before Turn 2
writes code:

1. **SingleEvaluationRequest shape is under-specified.** The
   packet's 4-field set (`response, source, request_id, caller_id`)
   is sufficient for CAP-7 only. CAP-2 needs `rubric_id` and
   `rubric_version`; CAP-5 needs `judge_engine` and a
   `dataset_hash` (or an agreed stand-in for ad-hoc inputs). CAP-1
   has no meaningful entry for a single instance.
2. **CAP-1 → CAP-2 is BLOCKED and CAP-2 → CAP-7 is BLOCKED as
   ordered pairs.** CAP-1 operates on datasets; CAP-2/CAP-7 are
   sibling evaluators on single cases. The A0 audit's "canonical
   sequence" is an eval/run.py convention, not a composable
   pipeline. The Runner can still invoke all four capabilities, but
   it invokes them as a fan-out with envelope stamping — not as a
   chain. The packet text "CAP-1 → CAP-2 → CAP-7 → CAP-5" will
   need to be re-read as "Runner invokes CAP-1 (if applicable),
   CAP-2, and CAP-7, then stamps results and calls CAP-5."
3. **`prompt_version` stamp is BLOCKED** against CAP-7's current
   contract. Decide: wrapper-constant, capability extension, or
   drop.

If the architect accepts these adjustments, Turn 2 can proceed. If
any of the three require capability-level rework, that rework
belongs in a separate packet (CP-1 would wait).

## 8. Scope out

Questions this reconnaissance surfaced but does not answer; hand
off to architect chat.

- Should CAP-1 gain a second entry shape for "single ad-hoc
  instance" (e.g. `register_adhoc(response, source) ->
  ResolvedInstance`), or is ad-hoc a documented bypass?
- Is the "CAP-2 → CAP-7" handoff described in the packet actually
  meant to be "CAP-2 ∥ CAP-7" (parallel sibling evaluators whose
  outputs the Runner merges)?
- Should `append_run_registry_entry` remain a raw function, or be
  lifted to a class with a single `persist(envelope: Envelope,
  hallucination_result: HallucinationResult) -> None` method, to
  give CAP-5 a cohesive entry?
- Where should `prompt_version` live for non-LLM-judge evaluators
  (the hallucination pipeline is multi-layer; `"v1"` of which
  layer)?
- The test baseline claim (756) could not be verified on this host
  because Python deps are not installed. Is the 756 number from a
  CI run log? If so, reference the run so Turn 2 and follow-on
  audits have a traceable anchor.
- CAP-2 swallows rule-apply exceptions silently (engine.py:163–164,
  169–170). A future Control Plane concern — should a silent rule
  failure emit a `rule_change` or `rule_error` event via
  `append_event`? Out of CP-1 scope; flagged for the architect.
