---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0009: Immutable, hashed, RBAC-protected golden baseline

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The baseline mechanism
> is documented as EPIC 4.1 in the Walkthrough and implemented in
> `src/llm_judge/eval/baseline.py`.

Regression detection requires a stable reference. Without an immutable
baseline, there is no "before" state to compare a candidate run against,
and "did quality regress?" becomes an unanswerable question. A baseline
that can be overwritten silently is worse than no baseline, because it
creates the illusion of a standard while destroying the reference point.

## Decision Drivers

- Reliability: the baseline must never be corrupted, lost, or silently
  overwritten.
- Security: only authorised users may promote a new baseline.
- Testability: promote, retrieve, and hash-verify must be a deterministic
  round trip.
- Auditability: for every active baseline, the system must record which
  evaluation run justified it.

## Considered Options

1. **Overwritable baseline file** — just a file; any run can replace it.
2. **Versioned, mutable baselines** — each baseline has a version, but
   versions can be edited.
3. **Immutable, hashed, RBAC-protected baselines** — write-once, hashed
   at creation, promotion requires authorisation.

## Decision Outcome

**Chosen option: immutable, hashed, RBAC-protected (Option 3).**

A baseline is a snapshot of `manifest.json`, `metrics.json`, and
`judgments.jsonl` (see ADR-0012) from a specific evaluation run. Once
written, it is never modified. Every file is SHA-256 hashed at creation;
the hash is verified before every read. Promotion requires explicit
authorisation and records which run ID produced the baseline.

## Consequences

### Positive

- The baseline is trustworthy evidence; if the hash verifies, the bytes
  are exactly what was promoted.
- Every baseline can be traced to the run that produced it.
- Rolling back a bad promotion is a matter of pointing `latest.json` at
  the previous snapshot.

### Negative

- Storage grows over time (one snapshot per promotion); retention
  policy is needed.
- Promotion is friction; the CI gate (ADR-0011) compensates by
  automating the promote-worthy case.

## More Information

- Implementation: `src/llm_judge/eval/baseline.py`
- Makefile: `baseline-validate`, `baseline-dry-run`, `baseline-promote`
- Walkthrough EPIC 4.1
- Related: ADR-0010 (diff engine), ADR-0011 (three-way CI gate),
  ADR-0012 (artifact triplet)
