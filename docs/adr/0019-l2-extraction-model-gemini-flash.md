---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0019: L2 extraction model — Gemini 2.5 Flash (SLM alternative rejected)

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The decision emerged
> from Exp 32–36 (the "Science Gate" series) which evaluated small
> language models as a cost-reduction alternative to Gemini for L2
> fact-table extraction.

L2 extraction (ADR-0018) runs five single-task prompts per source
document. Gemini 2.5 Flash handles these well but costs API money per
call. A self-hosted small language model (SLM) such as Flan-T5 would
be cheaper per call and more controllable. The question: is an SLM
good enough to replace Gemini at L2 extraction?

Exp 32–36 tested this systematically through chained extraction and
entity micro-queries. The SLM reached 85.7 % of the Gemini baseline on
extraction quality — closer than expected but not close enough for
L2's role as a primary extractor.

## Decision Drivers

- Extraction quality governs downstream graph quality and therefore
  L2 precision.
- L2 runs at source-document granularity with caching, so per-call
  cost is bounded.
- Operational reliability: a hosted API is one less model to deploy.

## Considered Options

1. **Gemini 2.5 Flash** — the experimental baseline.
2. **SLM (Flan-T5 or similar)** — cheaper, self-hosted.
3. **Hybrid** — SLM for easy extractions, Gemini for hard ones.

## Decision Outcome

**Chosen option: Gemini 2.5 Flash (Option 1).**

The Science Gate experiments established that the SLM reached 85.7 %
of Gemini's extraction quality. For a layer that must preserve 100 %
precision downstream, a 14 % extraction-quality gap is not acceptable.
The hybrid approach was considered and deferred: the routing logic
("which extractions are easy?") is itself an open problem that would
consume more effort than the cost saving justifies at current scale.

## Consequences

### Positive

- L2 precision is preserved at the validated level.
- Operational simplicity: one extraction model, one prompt family,
  one set of tuning knobs.
- The SLM path is not lost; Exp 32–36 results are the benchmark for
  any future SLM revisit.

### Negative

- Per-document extraction cost is higher (though bounded by caching).
- A dependency on an external API. Rate limits, outages, and model
  updates affect L2.
- If Gemini's behaviour changes in a future model release, L2 fact
  tables may need regeneration.

## More Information

- Experiments: Exp 32, 33, 34, 35, 36 (Science Gate series)
- Result: SLM at 85.7 % of Gemini baseline
- Revisit trigger: if a future SLM reaches ≥ 95 % of Gemini
  baseline on the same benchmark, or if cost becomes binding
- Related: ADR-0018 (multi-pass extraction uses this model)
