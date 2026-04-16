---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0020: L2 flag-wins ensemble aggregation across five property graphs

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The design emerged from
> Exp 37, which is the current L2 breakthrough result: 78 sentences
> cleared at 100 % precision, 11 of 16 hallucinations caught, zero safety
> violations.

L2 builds five NetworkX property graphs from the multi-pass fact tables:
entities (G1), events (G2), relationships (G3), numbers (G4), negations
(G5). For each sentence, each graph can independently yield a verdict
— grounded, flagged, or unknown. The question: how to aggregate five
possibly-conflicting verdicts into a single L2 decision?

Three natural aggregations exist: majority vote, unanimity, or
flag-wins. Exp 37 tested flag-wins and achieved the 100 %-precision
result that unblocked the whole L2 layer.

## Decision Drivers

- Precision must be 100 % — L2's role in the cascade depends on it.
- A single graph finding a definite disagreement with the source is
  stronger evidence than four graphs finding vague agreement.
- Aggregation must be explainable; an operator must be able to point
  to *why* a sentence was flagged.

## Considered Options

1. **Majority vote** — three or more graphs grounded → grounded; three
   or more flagged → flagged.
2. **Unanimity required to flag** — all five graphs must flag to
   produce a flag.
3. **Flag-wins** — any single graph flag (if justified by its graph
   traversal) beats any number of grounded verdicts.

## Decision Outcome

**Chosen option: flag-wins (Option 3).**

If any graph finds evidence of disagreement (e.g., G4 finds a number in
the response that is not in the source), the sentence is flagged even
if G1, G2, G3, and G5 all return grounded. The rationale: a specific
disagreement is high-signal; a general agreement is low-signal. Exp 37
validated this at 100 % precision.

Graphs that return "unknown" (insufficient information to decide) do
not participate in aggregation.

## Consequences

### Positive

- Preserves the 100 % precision target required by L2.
- Evidence is concrete — the flagging graph is named, and its traversal
  path can be inspected.
- Naturally handles the case where only one dimension of the fact is
  wrong (e.g., the number is wrong but the entity, event, and
  relationship are all correct).

### Negative

- Recall at L2 is lower than it would be under majority vote: any
  graph extraction error that yields a false flag is a false positive.
  This is compensated by the caching and the multi-pass extraction
  quality (ADR-0019).
- Adding a sixth graph in the future means revisiting the aggregation
  rule; a new ADR will be needed.

## More Information

- Experiments: Exp 37 (78 sentences cleared, 100 % precision, 11/16
  hallucinations caught)
- PR: #147 (merged to master)
- Related: ADR-0018 (fact tables feed these graphs), ADR-0019
  (extraction model)
