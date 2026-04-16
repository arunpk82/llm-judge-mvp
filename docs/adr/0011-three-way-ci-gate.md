---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0011: Three-way CI gate — block, approve, or approve-and-promote

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The CI gate pattern is
> documented as EPIC 4.3 in the Walkthrough.

A CI gate over evaluation metrics must handle three distinct outcomes,
not two:

1. The candidate run is **worse** than the baseline by more than the
   allowed tolerance. Block the merge.
2. The candidate run is **within tolerance** of the baseline. Approve
   the merge.
3. The candidate run is **better** than the baseline by more than the
   promotion threshold. Approve the merge and flag the result as a
   candidate for baseline promotion.

A two-way gate (pass/fail) either ignores improvements or treats them
as regressions. Neither is correct.

## Decision Drivers

- Regressions must be blocked reliably.
- Improvements must not be silently overwritten; the baseline evolves
  through explicit promotion.
- The gate must be fast enough not to strangle CI.
- The decision must be structured and auditable.

## Considered Options

1. **Two-way gate** (pass/fail) — improvements treated the same as
   within-tolerance.
2. **Three-way gate** — distinct handling of regressions, within-
   tolerance, and significant improvements.
3. **Automatic promotion** — any improvement over baseline
   automatically becomes the new baseline.

## Decision Outcome

**Chosen option: three-way gate (Option 2).**

The gate reads the diff output (ADR-0010), applies per-metric tolerance
(set by `--max-metric-drop` style flags), and emits one of three
decisions: `block`, `approve`, or `approve_and_promote_for_review`.
Promotion is never automatic; it is a human step with RBAC (ADR-0009).

## Consequences

### Positive

- Every CI run has a clear disposition. No silent passes, no silent
  fails.
- Improvements are surfaced deliberately, not absorbed into the baseline
  unnoticed.
- Tolerance is per-metric, so a 0.01 drop in `f1_fail` can fail the gate
  while a 0.05 drift in `l2_cleared` may not.

### Negative

- Tolerance values must be chosen and maintained. Too tight → CI churn
  on noise; too loose → regressions slip through.
- Promotion becomes a separate governance step. Teams need a clear
  promotion procedure.

## More Information

- Walkthrough EPIC 4.3
- Related: ADR-0009 (baseline), ADR-0010 (diff), ADR-0015 (flat keys
  make tolerance per-metric easy)
