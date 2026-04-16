---
status: superseded by ADR-0024
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0017: L1 rules design — substring plus structural checks

> **Superseded.** This ADR is superseded by [ADR-0024](./0024-l1-production-methods.md). The B1–B4 structural rules described here were never implemented in production code; L2's property graphs (G4 numbers, G5 negations) cover those structural disagreements. ADR-0024 documents what L1 actually does (substring + ratio + Jaccard).

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The design emerged from
> Exp 29 and 29b, which cleared 21 sentences (~7.4 %) at 100 % precision.

L1 is the cheapest layer of the cascade: deterministic, microsecond-
latency, zero cost. The design question is what L1 should actually check.
If it is too narrow, it contributes no coverage. If it is too broad, it
loses precision and defeats the cascade model.

The experimental exploration converged on a small set of rules that each
have a near-zero false-positive rate on RAGTruth-50.

## Decision Drivers

- Precision: L1's value depends on being right. Any rule that flips on
  ambiguous cases must be deferred to L2 or later.
- Coverage: L1 should resolve as many sentences as it can at that
  precision, to reduce what flows downstream.
- Simplicity: L1 rules must be auditable by eye.

## Considered Options

1. **Verbatim-match only** — if a sentence appears verbatim in the
   source, mark grounded. Otherwise pass to L2.
2. **Substring + structural** — verbatim match plus number mismatch,
   entity missing, negation flip, qualifier shift.
3. **Heavy rule suite** — dozens of regex-based checks.

## Decision Outcome

**Chosen option: substring plus structural (Option 2).**

Five rules:

- **A1 Exact match** — sentence appears verbatim in source → grounded.
- **B1 Number mismatch** — a number in the response does not appear in
  the source → flag.
- **B2 Entity missing** — a proper noun in the response does not appear
  in the source → flag.
- **B3 Negation flip** — negation polarity disagrees between response
  and source → flag.
- **B4 Qualifier shift** — universal/existential quantifier mismatch
  (e.g., "all" vs "some") → flag.

Exp 29/29b confirmed 100 % precision on the 21 sentences these rules
cleared.

## Consequences

### Positive

- 7 % of sentences never reach L2 or beyond; their verdicts are free.
- Rules are individually testable and documentable.
- Misfires are immediately debuggable — a specific rule fired on a
  specific substring.

### Negative

- 7 % is a small slice. Most of the cascade's load remains downstream.
- Each rule is a maintenance surface; updates require experimental
  re-validation.

## More Information

- Experiments: Exp 29, Exp 29b
- Config: `hallucination_pipeline_config.yaml`, section `l1_rules`
- Related: ADR-0002 (cascade architecture)
