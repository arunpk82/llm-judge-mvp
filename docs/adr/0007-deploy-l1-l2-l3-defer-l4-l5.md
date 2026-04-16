---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0007: Deploy L1→L2→L3 in this phase; defer L4 and L5

## Context and Problem Statement

Forty-three experiments (Exp 17b–43) have validated the four-layer cascade.
L1 (Rules), L2 (Knowledge-Graph Ensemble), and L3 (MiniCheck + fact-counting)
each hit 100 % precision on RAGTruth-50 in isolation. L4 (Gemini per-sentence
judge) and L5 (human adjudication) are architecturally specified but have
**not** been measured end-to-end in a production pipeline with ground-truth
comparison. They are simpler to wire than L1–L3 (L4 is an API call + verdict
routing; L5 is a queue + UI), but they are not ready to deploy in this
phase.

Meanwhile, the integration gap means L1–L3 are not reaching production
callers either. `IntegratedJudge.evaluate_enriched()` and `BenchmarkRunner`
invoke `check_hallucination()` without the parameters that enable L2 and L3.

The question is: what is the scope of the current deployment phase?

## Decision Drivers

- L1–L3 are validated; deploying them captures 43 experiments' worth
  of work.
- L4 and L5 need separate, small experiments before production.
- A phased rollout is lower-risk than a big-bang deployment.
- Wave 6 (user adoption) is blocked until the validated cascade reaches
  production callers.

## Considered Options

1. **Deploy L1 only** — safest, but wastes L2 and L3 experimental work.
2. **Deploy L1–L3** — deploys what is validated, leaves L4/L5 for next
   phase.
3. **Deploy L1–L5** — deploys everything, including unvalidated L4 and
   L5.

## Decision Outcome

**Chosen option: Deploy L1–L3. L4 and L5 in a follow-up phase.**

The current phase closes the integration gap so that `IntegratedJudge`
and `BenchmarkRunner` invoke the full L1→L2→L3 cascade. The phase ends
with a calibration run on RAGTruth-50 that becomes the seeded baseline
for CI regression gating (see ADR-0009 and ADR-0011).

The follow-up phase runs experiments on L4 and L5, using the now-live
L1–L3 pipeline as the substrate.

## Consequences

### Positive

- Production captures the validated cascade with zero architectural risk
  on the layers that are live.
- L4/L5 experiments run against a real production pipeline, not a
  bespoke script.
- The scope of this phase is precise and measurable: F1 on RAGTruth-50
  through the production entry points should match the experiment F1
  within tolerance.

### Negative

- Two phases means two deployment events.
- The config (ADR-0006) will show `l4_enabled: false` and `l5_enabled: false`
  for some time, which must be clearly explained.
- Stakeholders who have seen L4-style results in experiments will need
  to understand why they are not yet in production.

## More Information

- Related: ADR-0002 (the four-layer cascade), ADR-0013 (RAGTruth-50
  benchmark)
- Experiments: Exp 17b–43
