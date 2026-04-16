---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0008: Dataset governance — validate, register, pin

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The dataset governance
> pipeline has been in effect since approximately early 2025 and is
> documented in the Walkthrough as CAP-1.

An evaluation platform's results are only as trustworthy as the datasets
it runs on. Corrupt, incomplete, duplicated, or silently modified data
produces misleading metrics. Two weeks later, a customer complains, and
nobody can reconstruct what was actually run.

We need a pipeline that ensures every dataset entering the system is
validated for structure and content, registered with a version, and
pinned by cryptographic hash so that tampering is detectable.

## Decision Drivers

- Reproducibility: the same dataset version must produce the same
  results on every run.
- Tamper detection: a dataset that silently changes must not pass
  unnoticed.
- Auditability: when a run is questioned, the exact bytes used must be
  retrievable.
- Early failure: bad data should fail ingestion, not appear as a quality
  regression three weeks later.

## Considered Options

1. **Trust-on-use** — any file path works; no validation.
2. **Validate-on-register** — datasets enter through a registration API
   that validates and pins.
3. **Validate-on-load** — every load re-validates and re-hashes.

## Decision Outcome

**Chosen option: validate, register, pin (Option 2 with pin verification
on every load).**

The pipeline is: validate structure (schema compliance, required
columns) → validate content (no duplicates, no missing IDs, expected
row counts) → register with SHA-256 content hash → verify hash on
every subsequent load. On failure at any step, ingestion is blocked
with a structured error report.

## Consequences

### Positive

- Every evaluation run traces back to a specific, verifiable dataset
  version.
- Silent data modification is impossible; hash mismatch fails the run.
- Early failure surfaces data problems at the point of entry, not in
  the output.

### Negative

- A dataset cannot be edited in place; updates require a new version.
- The registry and its metadata must themselves be governed (part of
  CAP-5 artifact governance).

## More Information

- Walkthrough Section 5.1 (CAP-1 Dataset Governance)
- Makefile targets: `registry-list`, `registry-show`
- Related: ADR-0013 (RAGTruth-50 is registered via this pipeline)
