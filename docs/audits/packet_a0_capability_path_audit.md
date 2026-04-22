# Packet A0 — Capability Path Audit

Date: 2026-04-22
Branch: feat/packet-a0-capability-path-audit
Commit: ce9db6a170130ece055a5d8b943df881b05ad780

Scope: read-only audit. No source, config, or pipeline code modified.
No benchmark executed. Findings are grounded in code citations only.

Note on module layout: the audit prompt listed
`src/llm_judge/datasets/validator.py`; no such file exists. The
equivalent logic lives at `src/llm_judge/dataset_validator.py` and is
imported by `datasets/registry.py:99–110`. All other expected modules
are present.

## 1. Canonical sequence

The platform's defined sequence for a benchmark run is expressed by
the orchestrator at `src/llm_judge/eval/run.py:main` (lines 165–463).
Reading top-down from that function, the canonical order is:

1. **CAP-1 Dataset Governance — resolve + validate dataset.**
   `eval/run.py:158–162` calls `_resolve_dataset_path` which in turn
   calls `DatasetRegistry.resolve(dataset_id, version)` at
   `datasets/registry.py:56–161`. `resolve()` enforces:
     - metadata id/version match (`registry.py:65–72`)
     - SHA-256 content-hash verification (`registry.py:81–88`)
     - integrity/security checks via
       `llm_judge.dataset_validator._check_integrity` and
       `_check_security` (`registry.py:98–152`).
   An explicit `RunSpec.dataset.path` is flagged with a
   `DeprecationWarning` at `run.py:147–153` as a bypass.

2. **CAP-2 Rule Engine — resolve the rubric's rule plan.**
   `eval/run.py:199–205` calls
   `rules.engine.load_plan_for_rubric(rubric_id, version)` at
   `rules/engine.py:208–258`, which loads
   `configs/rules/<rubric_id>_<version>.yaml` and excludes deprecated
   rules via `rules.lifecycle.get_deprecated_enforced_rules()`
   (`engine.py:221–223`).

3. **CAP-3 Rule Governance — verify runtime ↔ manifest alignment.**
   `eval/run.py:224–240` calls
   `rules.lifecycle.check_rules_governed()` at
   `rules/lifecycle.py:346–386` (warn-only at run.py:228–233). The
   same block calls `emit_rule_snapshot` at
   `rules/lifecycle.py:389–436` which writes an event via
   `eval.event_registry.append_event(event_type="rule_change", ...)`.

4. **CAP-7 Evaluation — execute the judge per case.**
   `eval/run.py:313–382` iterates rows, calls
   `engine.evaluate(req)` (line 326) where the engine is whatever
   `runtime.get_judge_engine()` returns. Metrics schema is enforced
   via `_enforce_metrics_schema` at `run.py:120–141` (required keys
   declared in `rubrics/registry.yaml`).

5. **CAP-5 Artifact Governance — persist manifest, judgments, metrics;
   register run; emit event.**
   `eval/run.py:291` writes `validation_report.json`; `run.py:384–402`
   writes `judgments.jsonl` + `metrics.json`; `run.py:413–425` calls
   `eval.registry.append_run_registry_entry` (append-only
   `run_registry.jsonl`); `run.py:428–448` calls
   `eval.event_registry.append_event(event_type="eval_run", ...)`
   which enforces an allow-listed `VALID_EVENT_TYPES` frozenset at
   `event_registry.py:41–50,135–139`; `run.py:460` writes the final
   `manifest.json`.

6. **CAP-4 Baseline Governance — promote / compare (separate step).**
   Triggered after step 5 via `eval/baseline.py`. Policy-gated
   promotion lives at `baseline.py:761–854` and emits a
   `baseline_promotion` event (`baseline.py:831–852`). This step is
   not inline in `eval/run.py`; it is a separate CLI invocation
   exposed at `eval/__main__.py:_cmd_baseline_promote` (lines 28–45).

7. **CAP-6 Drift Monitoring (horizontal) — check drift after runs
   accumulate.** `eval/drift.py:check_drift` (lines 758–930) reads
   the `run_registry.jsonl` written in step 5, compares the latest
   run's `metrics.json` to the baseline pointer
   (`baselines/<suite>/<rubric_id>/latest.json`), and emits a
   `drift_alert` event (`drift.py:420–455`). Invoked via
   `python -m llm_judge.eval.drift check --policy ...` after a run.

8. **CAP-10 Calibration (horizontal) — trust gate on the judge.**
   `calibration/__init__.py:check_trust_gate` (lines 349–413) is
   invoked by the `CalibratedJudge` wrapper at
   `calibration/__init__.py:421–475`. The wrapper raises
   `RuntimeError` in `__init__` if `check_trust_gate` returns
   `(False, ...)` (line 452).

CAP-8 (Developer Experience) is expressed as CLIs/wrappers around the
above (`datasets/cli.py`, `eval/__main__.py`, `eval/baseline.py`
CLI, `tools/run_ragtruth50.py`). CAP-9 (Platform Health) has no
distinct module: the signals come from pre-flight notes in
`eval/run.py:189–242` and the `ok/skipped` fields on drift heartbeat
in `drift.py:311–365`.

## 2. RAGTruthAdapter path traceability

Entry: `tools/run_ragtruth50.py:cmd_benchmark` (lines 181–284) →
`RAGTruthAdapter()` (tools line 223) →
`benchmarks.runner.run_benchmark(adapter, ...)` (tools line 246) →
`adapter.load_cases(...)` (runner.py:827) → per-case
`_evaluate_case_all_properties` (runner.py:183–444) which calls
`calibration.hallucination.check_hallucination` (runner.py:210,
hallucination.py:548).

| # | Capability | Verdict | Evidence (file:line) | Notes |
|---|------------|---------|----------------------|-------|
| 1 | CAP-1 Dataset Governance | BYPASSED | `src/llm_judge/benchmarks/ragtruth.py:97–113, 214–218`; `src/llm_judge/benchmarks/__init__.py:9` | `RAGTruthAdapter._load_sources` and `load_cases` read `source_info.jsonl` / `response.jsonl` directly via `open()`. No `DatasetRegistry.resolve` call; `benchmarks/__init__.py:9` states "Parallel to DatasetRegistry, not an extension (different trust model)". Content hash check at `datasets/registry.py:81–88` is never reached. |
| 2 | CAP-2 Rule Engine | BYPASSED | `src/llm_judge/benchmarks/runner.py` (no import of `llm_judge.rules`); `tools/run_ragtruth50.py` (no import of `llm_judge.rules`) | Runner hard-codes 28 property IDs in `runner.py:151–180` and executes each via property-specific helpers. Neither `RuleEngine` nor `load_plan_for_rubric` is invoked. |
| 3 | CAP-3 Rule Governance | BYPASSED | No caller of `rules.lifecycle.check_rules_governed` or `emit_rule_snapshot` under `benchmarks/` or `tools/run_ragtruth50.py` (grep confirms). | A rule added to `RULE_REGISTRY` without a manifest entry would be invisible on this path because no rule is consulted at all. |
| 4 | CAP-4 Baseline Governance | BYPASSED | `tools/run_ragtruth50.py:257–275` writes `results/ragtruth50_results.json` via `Path.write_text`; no call to `create_baseline_from_run`, `promote_baseline_from_run`, or `validate_latest_baseline`. | No baseline comparison before or after the benchmark; no `latest.json` read or write. |
| 5 | CAP-5 Artifact Governance | PARTIAL | Written: `tools/run_ragtruth50.py:274` (results/ragtruth50_results.json); `benchmarks/runner.py:823–825` (results/benchmark_checkpoint.json). Not called: `eval.registry.append_run_registry_entry`, `eval.event_registry.append_event`, `eval.io.build_manifest`, `validation_report.json` emission. | Artifact is produced but outside the governed schema. `eval_run` event type is never emitted, so `event_registry.py:41–50` allow-list is irrelevant. The run is invisible to any consumer that scans `state/run_registry.jsonl` or `state/event_registry.jsonl`. |
| 6 | CAP-6 Drift Monitoring | BYPASSED | `eval/drift.py:770` reads entries via `_iter_jsonl(registry_path)` (default `state/run_registry.jsonl`); the RAGTruth path writes no registry entry (see row 5). | `check_drift` would return `NO_DATA` (drift.py:778–792) for this benchmark's runs. Heartbeat at `drift.py:311–365` checks `eval_run` events, none of which exist from this path. |
| 7 | CAP-7 Evaluation | PARTIAL | Engaged: `benchmarks/runner.py:210` (`check_hallucination`), `runner.py:327–330` (`IntegratedJudge.evaluate_enriched` when `--with-llm`). Not engaged: `engine.evaluate(req)` at `eval/run.py:326`, metrics schema enforcement at `run.py:400`. | The evaluation that runs is the hallucination cascade + 28-property bank; it is not the engine/metrics contract that CAP-7 defines in `eval/run.py`. `_enforce_metrics_schema` (run.py:120–141) is never invoked. |
| 8 | CAP-8 Developer Experience | ENGAGED | `tools/run_ragtruth50.py:1–596` provides `preseed`, `benchmark`, `funnel`, `all` subcommands and environment-variable overrides (`HALLUCINATION_LAYERS`). | This capability is naturally present because the entry point *is* the DX layer. |
| 9 | CAP-9 Platform Health | BYPASSED | No caller of `eval/drift.py:_heartbeat_check` or pre-flight notes under `benchmarks/` or `tools/run_ragtruth50.py`. | The L2-cache prerequisite check at `tools/run_ragtruth50.py:108–173` is a pipeline-specific provisioning step, not a platform-health signal. |
| 10 | CAP-10 Calibration | PARTIAL | Engaged: `benchmarks/runner.py:17` imports `check_hallucination` and `_split_sentences` from `calibration/hallucination.py`; runner invokes the L1–L4 cascade per case. Not engaged: `calibration/__init__.py:CalibratedJudge` / `check_trust_gate`. | The hallucination *pipeline* (calibration subpackage content) runs. The calibration *governance* (`check_trust_gate`, `JudgeMeta.status`, `CalibratedJudge` wrapper) is never invoked — `IntegratedJudge` is instantiated raw at `runner.py:323–327` with no calibration wrapper. |

## 3. Legacy loader path traceability

Entry: `experiments/benchmark_loader.py:load_ragtruth_50` (lines
38–116). The function reads the three RAGTruth files directly with
`open()`, splits sentences with spaCy, and returns
`(responses, sentences)`. It has no evaluation step; any caller
would drive evaluation separately. The loader module has zero
imports from `llm_judge.datasets`, `llm_judge.eval`,
`llm_judge.rules`, or `llm_judge.calibration`.

| # | Capability | Verdict | Evidence (file:line) | Notes |
|---|------------|---------|----------------------|-------|
| 1 | CAP-1 Dataset Governance | BYPASSED | `experiments/benchmark_loader.py:54–72` | Direct `open()` on three file paths hard-coded at lines 22–24. No registry call. |
| 2 | CAP-2 Rule Engine | ABSENT (on this path) | Loader performs no evaluation. | Caller's responsibility; not exercised inside this file. |
| 3 | CAP-3 Rule Governance | ABSENT (on this path) | Loader performs no evaluation. | Same as row 2. |
| 4 | CAP-4 Baseline Governance | ABSENT (on this path) | Loader performs no evaluation. | Same as row 2. |
| 5 | CAP-5 Artifact Governance | ABSENT (on this path) | `benchmark_loader.py` returns in-memory tuples only; writes nothing. | No artifact produced; also means no ungoverned artifact. |
| 6 | CAP-6 Drift Monitoring | ABSENT (on this path) | No evaluation, no run, no registry entry. | Same as row 4. |
| 7 | CAP-7 Evaluation | ABSENT (on this path) | Loader returns data only. | Caller must evaluate. |
| 8 | CAP-8 Developer Experience | PARTIAL | `benchmark_loader.py:119–135` is a `__main__` stats print. | Utility only, not a governed DX surface. |
| 9 | CAP-9 Platform Health | ABSENT (on this path) | — | — |
| 10 | CAP-10 Calibration | ABSENT (on this path) | No judge is constructed. | — |

## 4. Divergence between paths

- **Adapter runs the hallucination pipeline; loader does not.** The
  adapter (`RAGTruthAdapter` → `run_benchmark`) invokes
  `check_hallucination` and the 28-property bank. The legacy loader
  returns raw data and leaves evaluation to the caller. Neutral
  structurally, but means the adapter at least measures *something*
  of the platform (CAP-10 calibration pipeline, CAP-7 evaluation
  pipeline), while the loader measures nothing on its own.
- **Adapter produces an artifact; loader does not.** Adapter writes
  `results/ragtruth50_results.json` (`tools/run_ragtruth50.py:274`)
  and a checkpoint (`benchmarks/runner.py:823–825`); loader writes
  nothing. Regression: the adapter's artifact bypasses the
  `run_registry.jsonl`/`event_registry.jsonl` contract, so it
  appears governed but is not indexed by any downstream capability.
- **Both bypass CAP-1/2/3/4/6.** Neither file imports
  `llm_judge.datasets`, `llm_judge.rules`, `llm_judge.eval.baseline`,
  or `llm_judge.eval.drift`. This is a symmetric gap, not an
  improvement of one path over the other.
- **Adapter declares a parallel trust model; loader declares none.**
  `benchmarks/__init__.py:9` explicitly states the adapter layer is
  "Parallel to DatasetRegistry, not an extension (different trust
  model)". This is an intentional architectural split. The legacy
  loader is silent — it is simply pre-capability-framework code.

## 5. Sequence-order findings

No capability in the RAGTruthAdapter path was marked ENGAGED with an
incorrect position. Reason: on the RAGTruth path, CAP-1/2/3/4/6/9
are BYPASSED; CAP-5/7/10 are PARTIAL (some entry points called,
others not). There is no capability that is both engaged and
mis-ordered on the adapter path.

Observations on partial engagements (for completeness, not mis-order):

- CAP-5 writes a result file **after** the evaluation produces it
  (`tools/run_ragtruth50.py:274`, post `run_benchmark`). Position is
  correct in the ordering sense; the failure is scope (skips
  registry/event emission), not order.
- CAP-10's hallucination cascade runs **during** evaluation, not
  before as a gate. `CalibratedJudge` is designed to gate *at
  initialization* (`calibration/__init__.py:445–452`) but is never
  constructed on this path. Again: scope, not order.
- CAP-7's metrics schema enforcement runs in `eval/run.py:400`
  **after** metrics computation; on the adapter path this step is
  skipped entirely rather than performed out of order.

Legacy loader: no ENGAGED capabilities beyond a thin DX surface, so
no order findings apply.

## 6. Sequence enforcement mechanism

Status: **ABSENT**.

Evidence (search commands and counts run at audit time):

- `grep -rn "assert" src/llm_judge/eval/ src/llm_judge/datasets/ src/llm_judge/calibration/ | wc -l` → **4 matches.** All four are
  `eval.schema.assert_compatible_schema` / its callers in
  `eval/diff.py:10, 454, 534` and the definition at `eval/schema.py:8`.
  Scope: version-string compatibility between two already-produced
  artifacts. Not a capability-entry pre-condition.
- `grep -rn "raise.*Error" src/llm_judge/benchmarks/` → **11 matches.**
  All are `raise FileNotFoundError` / `NotImplementedError` for
  missing data files or abstract-method stubs (e.g., `ragtruth.py:99`,
  `ragtruth.py:216`, `benchmarks/__init__.py:94,112`). None check
  upstream capability provenance.
- `grep -rn "@contract\|@requires\|@pre\|@invariant" src/` →
  **0 matches.** No decorator-based contracts.
- `grep -rn "provenance\|trace_id\|upstream" src/llm_judge/` →
  **1 match.** `calibration/hallucination_graphs.py:224` — a
  comment noting shape "is guaranteed upstream". No runtime check.

Additional classification (defensive coding vs. sequence enforcement):

- `DatasetRegistry.resolve` (`datasets/registry.py:81–152`) validates
  the dataset it loads, but only for callers who *choose* to load
  via the registry. It cannot guarantee downstream capabilities saw
  a registry-resolved dataset.
- `CalibratedJudge.__init__` (`calibration/__init__.py:445–452`)
  refuses to construct if trust gate fails. This *is* an entry
  pre-condition, but only for callers that explicitly wrap the
  judge in `CalibratedJudge`. `benchmarks/runner.py:327` instantiates
  `IntegratedJudge()` without wrapping.
- `event_registry.append_event` enforces an allow-list of event
  types (`event_registry.py:41–50,135–139`). That protects the
  event schema, not the sequence of capability calls.
- `eval/run.py:189–242` runs "pre-flight" checks (rubric exists,
  rule plan loads, dataset fields present, rule governance) but
  these are inline checks inside the `eval/run.py` orchestrator.
  Any caller that does not enter via `eval/run.py` — including the
  RAGTruthAdapter path — skips all of them.

No central orchestrator, middleware, or contract-check decorator
enforces that (for example) CAP-7 refuses input lacking CAP-1
provenance, or CAP-5 refuses to persist output lacking upstream
event linkage. The capability modules are point-to-point importable,
and any caller can invoke any subset in any order.

Blast radius (concrete RAGTruth-path examples):

1. **Silent dataset replacement.** `datasets/benchmarks/ragtruth/response.jsonl` can be modified between runs. The `RAGTruthAdapter._load_sources`/`load_cases` reads the file directly (`ragtruth.py:104–107, 221–223`); the content-hash verification at `datasets/registry.py:81–88` is never reached because the registry is never consulted. The `results/ragtruth50_results.json` would reflect the modified data without any integrity alert.
2. **Ungoverned LLM judge.** `IntegratedJudge()` at `benchmarks/runner.py:327` constructs without passing through `CalibratedJudge`. A replacement or mis-configured judge would produce Cat 2 / 6.3 / 6.4 scores (`runner.py:341–370`) with no `check_trust_gate` refusal, even though `calibration/__init__.py:452` is designed to raise `RuntimeError` for exactly this case.
3. **Invisible run for drift.** `eval/drift.py:check_drift` iterates `state/run_registry.jsonl` (`drift.py:770`). The RAGTruth run writes nothing there, so drift monitoring cannot compare the benchmark's latest run to its baseline or detect a trend drop — `check_drift` would return `NO_DATA` (`drift.py:778–792`) indefinitely.

Scope note: this audit reports enforcement presence/absence as fact.
It does not propose where enforcement should live, which capability
should own a pre-condition, or what pattern (contract decorator,
provenance token, central runner) should apply. That is an
architect-chat decision.

## 7. Headline finding

Can a RAGTruth-50 measurement via `RAGTruthAdapter` be trusted to
measure the platform (not just the pipeline)? **QUALIFIED NO.**

The current path measures the hallucination cascade (L1–L4) and a
28-property property-bank, but routes around CAP-1, CAP-2, CAP-3,
CAP-4, CAP-6, and the trust-gate half of CAP-10. The result file
`results/ragtruth50_results.json` is therefore a pipeline
measurement, not a platform measurement. Shape of the minimum fix
set (not implementations, not owners):

- A binding between benchmark dataset loads and a governed trust
  contract (so a replaced data file would fail loudly as it does
  for registry-resolved datasets).
- A linkage from benchmark runs to the existing `run_registry` /
  `event_registry` so CAP-5 indexes them and CAP-6 can see them.
- An activation of CAP-10's trust gate on the judge the benchmark
  uses, not only on judges constructed via `eval/run.py`.
- A mechanism that makes bypass loud rather than silent — i.e., a
  downstream capability entry refusing input without upstream
  provenance. (Presence of this mechanism is the subject of §6;
  its shape and ownership are out of scope per §8.)

## 8. Scope out

Questions this audit surfaced that it does not answer; for handoff
to architect chat.

- Is the "parallel to DatasetRegistry" trust model for
  `BenchmarkAdapter` (asserted at `benchmarks/__init__.py:9`) the
  intended long-term architecture, or a transitional shim?
- Should pre-flight (currently inline in `eval/run.py:189–242`) be
  hoisted into a shared capability-agnostic runner that any entry
  point — `eval/run.py`, `tools/run_ragtruth50.py`,
  `benchmarks/runner.py` — must traverse?
- Where does trust-gate enforcement live when the caller is a
  benchmark runner rather than the eval runner? Is the benchmark
  runner expected to wrap `IntegratedJudge` in `CalibratedJudge`, or
  is the wrap expected deeper (inside `runtime.get_judge_engine` or
  `IntegratedJudge.__init__`)?
- Should the 28-property catalog in `benchmarks/runner.py:151–180`
  be pulled into `rules/manifest.yaml` governance (CAP-3), or is it
  intentionally outside rule governance?
- Should `eval/drift.py` add a completeness check that flags when
  a benchmark that *should* have run did not register (positive
  liveness, not just metric drift)?
- Is `L2_flagged` a cascade artifact that should appear in
  `event_registry` as a distinct event type, given that benchmark
  runs today do not generate any event-registry entries at all?
