# Architecture Decision Records

This folder contains the Architecture Decision Records (ADRs) for the
LLM Judge platform. We use **MADR 4.0** format. Each ADR captures a
single architectural decision, its context, the options considered,
the outcome, and the consequences.

See [CONTRIBUTING.md](./CONTRIBUTING.md) for how to write a new ADR.
See [adr-template.md](./adr-template.md) for the template to copy.

## Index

| # | Title | Status | Category | Date |
|---|-------|--------|----------|------|
| [0001](./0001-use-madr-4-for-adrs.md) | Use MADR 4.0 for ADRs | Accepted | Foundation | 2026-04-16 |
| [0002](./0002-four-layer-cascade-architecture.md) | Four-layer cascade architecture | Accepted | A — Pipeline | 2026-04-16 |
| [0003](./0003-two-gate-deterministic-llm-architecture.md) | Two-gate deterministic + LLM architecture | Accepted | A — Pipeline | 2026-04-16 |
| [0004](./0004-28-properties-organised-into-6-categories.md) | 28 properties organised into 6 categories | Accepted | A — Pipeline | 2026-04-16 |
| [0005](./0005-sentence-and-response-as-dual-evaluation-units.md) | Sentence and response as dual evaluation units | Accepted | A — Pipeline | 2026-04-16 |
| [0006](./0006-config-driven-pipeline-via-yaml.md) | Config-driven pipeline via YAML | Accepted | A — Pipeline | 2026-04-16 |
| [0007](./0007-deploy-l1-l2-l3-defer-l4-l5.md) | Deploy L1→L2→L3 in this phase, defer L4 and L5 | Accepted | A — Pipeline | 2026-04-16 |
| [0008](./0008-dataset-governance-validate-register-pin.md) | Dataset governance: validate, register, pin | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0009](./0009-immutable-hashed-rbac-golden-baseline.md) | Immutable, hashed, RBAC-protected golden baseline | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0010](./0010-diff-engine-with-flip-tracking.md) | Diff engine with per-response flip tracking | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0011](./0011-three-way-ci-gate.md) | Three-way CI gate (block / approve / approve + promote) | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0012](./0012-standard-artifact-triplet.md) | Standard artifact triplet: manifest + metrics + judgments | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0013](./0013-ragtruth-50-fixed-benchmark.md) | RAGTruth-50 as the fixed hallucination benchmark | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0014](./0014-spacy-sentence-splitting.md) | spaCy sentence splitting (not regex) | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0015](./0015-flat-keys-in-metrics-json.md) | Flat keys in metrics.json for per-key diff tolerance | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0016](./0016-embed-sentence-detail-in-judgments-jsonl.md) | Embed sentence detail inside judgments.jsonl | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0017](./0017-l1-rules-substring-plus-structural-checks.md) | L1 rules: substring + structural checks | Superseded by [0024](./0024-l1-production-methods.md) | C — Experiment | 2026-04-16 |
| [0018](./0018-l2-multi-pass-knowledge-graph-extraction.md) | L2 multi-pass knowledge graph extraction | Accepted | C — Experiment | 2026-04-16 |
| [0019](./0019-l2-extraction-model-gemini-flash.md) | L2 extraction model: Gemini 2.5 Flash (SLM rejected) | Accepted | C — Experiment | 2026-04-16 |
| [0020](./0020-l2-flag-wins-ensemble-aggregation.md) | L2 flag-wins ensemble aggregation | Accepted | C — Experiment | 2026-04-16 |
| [0021](./0021-l3-minicheck-primary-deberta-fallback.md) | L3 classifier stack: MiniCheck primary, DeBERTa fallback | Superseded by [0027](./0027-l3-fact-counting-default-legacy-flag-gated.md) | C — Experiment | 2026-04-16 |
| [0022](./0022-l3-fact-counting-design.md) | L3 fact-counting design (Exp 43) | Superseded by [0027](./0027-l3-fact-counting-default-legacy-flag-gated.md) | C — Experiment | 2026-04-16 |
| [0023](./0023-cascade-resolution-rule.md) | Cascade resolution: skip remaining layers once resolved | Accepted | C — Experiment | 2026-04-16 |
| [0024](./0024-l1-production-methods.md) | L1 production methods (substring, ratio, Jaccard) | Accepted | C — Experiment | 2026-04-16 |
| [0025](./0025-fact-table-extraction-as-registration-step.md) | Fact-table extraction as a dataset-registration pipeline step | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0026](./0026-config-driven-feature-flags.md) | Config-driven feature flags as the pipeline's behavioral contract | Accepted | A — Pipeline | 2026-04-16 |
| [0027](./0027-l3-fact-counting-default-legacy-flag-gated.md) | L3 fact-counting as default; MiniCheck and DeBERTa flag-gated | Accepted | C — Experiment | 2026-04-16 |
| [0028](./0028-ci-tests-flagged-off-paths.md) | CI tests every major flag combination to prevent flag rot | Accepted | B — Baseline & Eval | 2026-04-16 |
| [0029](./0029-layer-renaming-code-vs-cascade-names.md) | Resolve the layer-name legacy in the cascade code | Accepted | A — Pipeline | 2026-04-16 |
| [0030](./0030-calibration-run-as-baseline-seeding-event.md) | The calibration run is the baseline-seeding event | Accepted | B — Baseline & Eval | 2026-04-16 |

## Status values

- **Proposed** — under discussion, not yet adopted
- **Accepted** — adopted; decision is in effect
- **Rejected** — considered and not adopted; kept as record of the alternative
- **Deprecated** — no longer recommended, but not yet replaced
- **Superseded by ADR-NNNN** — replaced by a newer decision; link to the replacement

## Categories

- **Foundation** — about the ADR system itself
- **A — Pipeline** — overall pipeline architecture and scope
- **B — Baseline & Eval** — how we measure quality, store artifacts, and gate CI
- **C — Experiment** — decisions informed by the 43-experiment research programme

## Retroactive ADRs

ADRs 0002–0023 document decisions that predate the adoption of the ADR system.
They were written on 2026-04-16 by reconstruction from the LLM Judge Architecture
Walkthrough, the Learning App, the 28 Metrics Evaluation Reference, and the
source code on master. Each retroactive ADR has a note in its *Context* section
indicating the reconstruction and the approximate date the decision was actually
made.
