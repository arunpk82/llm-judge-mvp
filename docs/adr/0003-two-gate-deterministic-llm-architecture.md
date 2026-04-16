---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0003: Two-gate architecture (deterministic + LLM) for property evaluation

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16 to document a decision
> that has been in effect since approximately mid-2025. Reconstructed
> from the LLM Judge Architecture Walkthrough and the `integrated_judge.py`
> runtime code.

Property evaluation (the 28 properties across 6 categories) has a tension.
Some properties are cleanly deterministic — numeric matches, citation
verification, format compliance. Others are inherently judgment calls —
tone, relevance, correctness. A pure-deterministic system cannot score the
latter; a pure-LLM system is expensive, non-reproducible, and hard to audit
on the former.

We needed a structure that routed each property to the right kind of judge,
while preserving a single user-facing verdict per property.

## Decision Drivers

- Reproducibility: deterministic properties should give the same answer on
  every run, always.
- Cost: LLM calls should only happen where deterministic checks cannot
  decide.
- Explainability: the verdict source (deterministic vs LLM) must be visible.
- Calibration: LLM judges drift; we need to know when they flip on cases
  the deterministic gate already decided.

## Considered Options

1. **Deterministic only** — reject properties that cannot be scored by rules.
2. **LLM only** — score everything with an LLM judge.
3. **Two-gate** — Gate 1 is deterministic; Gate 2 is LLM. Each property
   chooses which gate is authoritative, with the other acting as a check.

## Decision Outcome

**Chosen option: two-gate architecture.**

Properties that can be decided deterministically (e.g., citation match,
numeric correctness) are scored at Gate 1. Properties requiring judgment
(e.g., tone) are scored at Gate 2. Some properties use both — Gate 1
resolves clear-cut cases, and ambiguous/failing cases route to Gate 2 via
`gate2_routing`.

This yields a reproducible-where-possible, LLM-where-necessary pipeline,
and the routing decision is per-property rather than pipeline-wide.

## Consequences

### Positive

- 15 of 28 properties score entirely at Gate 1, with no LLM cost and
  deterministic behaviour.
- The three auto-gated properties (tone F1=0.87, relevance F1=0.83,
  correctness F1=0.80) demonstrate the model's reliability where it is
  applied.
- LLM cost is bounded: only properties that need Gate 2 pay for it.

### Negative

- Two code paths per property must be maintained.
- A property can flip between gates over time as calibration improves;
  this requires versioning of gate assignment.

## More Information

- LLM Judge Architecture Walkthrough, Section 5.2 and 5.7
- Runtime: `src/llm_judge/integrated_judge.py`
- Related: ADR-0004 (the 28 properties), ADR-0002 (the hallucination
  cascade as a special case of Gate-2-like architecture)
