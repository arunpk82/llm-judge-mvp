---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0025: Fact-table extraction as a dataset-registration pipeline step

## Context and Problem Statement

L2 (ADR-0018, ADR-0020) requires fact tables — multi-pass Gemini
extractions from source documents. Today, fact tables exist only for
RAGTruth-50, in `experiments/exp31_multipass_fact_tables.json`. For any
new dataset to flow through the L2 cascade, its fact tables must be
extracted somehow.

There are two natural shapes:

- **At runtime, on demand.** The L2 layer calls Gemini when it sees a
  source it does not recognise. Hot path includes external API; new
  failure modes (timeout, malformed JSON, rate limit) live inside the
  cascade.
- **At registration, before the dataset is usable.** Extraction happens
  once, when the dataset enters the system. Hot path is always a cache
  lookup.

The platform already has a dataset-governance pipeline (CAP-1, ADR-0008)
with validation, registration, content hashing, and version pinning. Adding
extraction as a step in this pipeline is the natural fit — fact tables are
*derived artifacts* that share the same trust requirements as datasets.

## Decision Drivers

- **P02 Trust.** Fact tables drive verdicts. They must be reproducible,
  immutable, and hash-verifiable, just like datasets.
- **Hot-path cleanliness.** Runtime should never make external API calls
  inside the cascade. Cache lookup is the only allowed I/O.
- **One pipeline, not two.** Datasets and fact tables share the same
  governance shape; reusing the existing pipeline avoids inventing a
  parallel one.
- **Failure isolation.** An extraction failure should fail registration,
  not contaminate a runtime evaluation.

## Considered Options

1. **Runtime extraction.** L2 calls Gemini on cache miss; failures bubble
   into the cascade.
2. **Registration-time extraction.** Extraction triggered when a dataset
   registers; fact tables become first-class artifacts; runtime is cache
   lookup only.
3. **Out-of-band batch script.** Extraction is a separate manual step
   the operator runs; registration neither triggers nor verifies it.

## Decision Outcome

**Chosen option: Option 2 — registration-time extraction.**

When a new dataset is registered through the existing CAP-1 pipeline,
registration triggers fact-table extraction as a downstream step:

1. Dataset arrives, passes structural and content validation
   (ADR-0008).
2. Source documents are extracted from the dataset.
3. For each unique source, fact-table extraction runs the five Gemini
   passes (ADR-0018) plus the optional verification pass (P6).
4. Extracted fact tables are validated (well-formed JSON, expected
   pass keys present, non-empty extraction).
5. Each fact table is registered as a new immutable artifact, keyed
   by source SHA-256, with its own version, hash, and metadata.
6. The dataset registration succeeds only if all fact-table extractions
   succeed.
7. At runtime, L2 looks up fact tables by source SHA-256. Cache lookup
   only. No fallback to runtime extraction.

The Exp 31 fact tables are imported as the pre-seeded population for
RAGTruth-50.

## Consequences

### Positive

- Fact tables are first-class artifacts under the same governance as
  datasets — immutable, hashed, version-pinned.
- Runtime is deterministic: no external API in the hot path, no new
  failure modes in the cascade.
- New datasets are blocked at registration if their fact tables cannot
  be extracted, surfacing the problem early.
- Graphs are dataset-scoped (per source) by construction — the cache
  key is the source SHA-256, so graphs from one source cannot leak
  into another.
- Re-extraction (e.g., after a Gemini model upgrade) is a registration-
  pipeline operation, not a runtime concern.

### Negative

- Dataset registration becomes more expensive — a registration that used
  to take seconds may now take minutes for large datasets.
- The extraction-prompt versions become part of the trust chain. If
  prompts change, fact tables for all affected sources should be
  invalidated.
- An extraction failure blocks the dataset entirely. This is correct
  behaviour but means the operator must be able to debug extraction
  failures in the registration log.

## More Information

- Pre-seed source: `experiments/exp31_multipass_fact_tables.json`
- Related: ADR-0008 (dataset governance pipeline), ADR-0018 (multi-pass
  extraction), ADR-0019 (Gemini as extraction model), ADR-0020 (L2
  ensemble aggregation), ADR-0026 (config-flag governance)
- Revisit trigger: if extraction cost becomes binding for very large
  datasets, consider lazy/incremental extraction with the same artifact
  contract.
