---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0005: Sentence and response as dual evaluation units

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16 to document a decision
> that emerged from the Exp 17b–43 experiment series.

Hallucination detection has two natural units:

- **Sentence**, at which each cascade layer makes a verdict and at which
  ground truth is annotated in benchmarks like RAGTruth.
- **Response**, at which the user experiences the verdict ("is this
  answer hallucinated, yes or no?").

Early experiments measured at sentence level because ground truth was
sentence-level. But end users do not see sentences; they see responses.
A response with five sentences where one sentence is hallucinated is a
hallucinated response, full stop.

Collapsing either unit into the other loses information. Sentence-only
loses the user-facing question. Response-only loses the diagnostic
ability to say which layer caught which failure.

## Decision Drivers

- Benchmarks label at sentence level; we must evaluate at that level.
- Users consume at response level; we must report at that level.
- Diagnostics at sentence level explain where the cascade succeeded
  or failed.
- Baseline diffing must work at both levels to detect regressions
  that hide at one level but appear at the other.

## Considered Options

1. **Sentence only** — every metric is at sentence level; response-level
   is derived by OR-reduction post hoc.
2. **Response only** — every metric is at response level; sentence-level
   detail is diagnostic, not scored.
3. **Dual units** — every run produces both sentence-level and
   response-level metrics, and both can be gated.

## Decision Outcome

**Chosen option: dual units.**

Both sentence-level and response-level metrics are computed on every run
and both are emitted into `metrics.json` (see ADR-0015 for the schema).
Response-level is the user-facing metric; sentence-level is the
diagnostic and regression-detection metric.

## Consequences

### Positive

- Regression detection catches flips that are invisible at either unit
  alone.
- Reports can answer both "how good is the response-level verdict?" and
  "where in the cascade did things go right or wrong?"
- The telemetry stays coherent even as L4 or L5 are added later.

### Negative

- Both schemas must be maintained; adding a metric means adding it at
  both levels.
- Interpreting diffs requires understanding both levels — a sentence-level
  precision drop with stable response-level precision is possible and
  meaningful.

## More Information

- Related: ADR-0013 (RAGTruth-50 benchmark), ADR-0015 (flat metrics
  keys at both levels), ADR-0016 (sentence-level detail embedded in
  judgments.jsonl)
