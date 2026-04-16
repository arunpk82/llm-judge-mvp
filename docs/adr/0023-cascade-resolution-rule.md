---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0023: Cascade resolution — sentences skip remaining layers once resolved

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The rule is implemented
> in `src/llm_judge/calibration/hallucination.py` and is documented in
> `hallucination_pipeline_config.yaml`.

A cascade of four layers can be operated two ways. In **run-all**
mode, every sentence passes through every enabled layer, and verdicts
are aggregated at the end. In **early-resolution** mode, a sentence
that one layer resolves at sufficient confidence skips remaining
layers.

Run-all has the appeal of maximum evidence per sentence. Early-
resolution has the appeal of matching cost to uncertainty.

## Decision Drivers

- Cost: L4 (deferred) is expensive per sentence. Cascading only the
  hard cases to it is the core economic argument of the cascade.
- Precision: if each layer is 100 % precise when it chooses to decide,
  later layers add no information to an already-decided sentence.
- Complexity: aggregation rules across four layers are more complex
  than early termination.

## Considered Options

1. **Run-all then aggregate** — every sentence through every layer;
   aggregation handles disagreements.
2. **Early resolution with skip** — a sentence skips remaining layers
   once a layer decides it at sufficient confidence.
3. **Hybrid** — L1 and L2 always run; L3 and L4 only on undecided
   sentences.

## Decision Outcome

**Chosen option: early resolution with skip (Option 2).**

Each sentence enters L1. L1 produces one of: `grounded` (resolved),
`flagged` (resolved with disagreement), or `unknown` (pass down).
On `grounded` or `flagged`, the sentence skips L2, L3, L4. On
`unknown`, it cascades to L2. Same rule applies at L2 and L3.

Sentences that reach L4 (when enabled) are those that L1, L2, L3 all
returned `unknown`. These are the hardest 5 %ish of cases.

## Consequences

### Positive

- L4 cost is bounded by the fraction of sentences L1–L3 leave
  undecided (~20–25 %, dropping as the cascade matures).
- The funnel report (sentences entering, cleared, flagged, cascaded
  per layer) is a natural observability output.
- Layer promotion experiments (does enabling L3 change outcomes?) are
  tractable because the cascade is stateful per sentence.

### Negative

- A resolved sentence is not re-evaluated by later layers. If a later
  layer has better calibration, its verdict is not applied. Mitigated
  by requiring each layer to hold 100 % precision at its resolution
  threshold.
- Debug tooling must surface which layer resolved each sentence; the
  funnel report serves this role.

## More Information

- Implementation: `src/llm_judge/calibration/hallucination.py`,
  `check_hallucination()`
- Config: `hallucination_pipeline_config.yaml`
- Related: ADR-0002 (four-layer cascade), ADR-0005 (sentence as
  evaluation unit), ADR-0017–0022 (per-layer designs)
