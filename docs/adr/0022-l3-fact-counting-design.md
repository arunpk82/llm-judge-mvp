---
status: superseded by ADR-0027
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0022: L3 fact-counting design (Exp 43)

> **Superseded.** This ADR is superseded by [ADR-0027](./0027-l3-fact-counting-default-legacy-flag-gated.md). The fact-counting *design* described here is correct; ADR-0027 makes it the production default and specifies how the legacy MiniCheck + DeBERTa paths are preserved behind feature flags.

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The design emerged
> from Exp 43 and is the basis for the 76 % auto-clear rate on
> RAGTruth-50. The underlying principle is L61 ("derived confidence
> outperforms self-reported").

Within L3, MiniCheck produces a per-sentence `prob_supported` between
0 and 1. Using this probability as the gating signal is tempting — a
simple threshold. However, self-reported confidence from LLM-family
models is notoriously unreliable. A sentence at `prob_supported = 0.75`
is not 75 % likely to be grounded.

Exp 43 took a different approach: **decompose the sentence into
atomic facts, count how many are supported by the source, and use
the ratio as the derived confidence.** A sentence with 5 atomic
facts and 5 supported scores 1.0 grounded. A sentence with 5 facts
and 4 supported scores 0.8.

The result: 76 % of sentences auto-cleared at grounding ratio ≥ 0.8,
with 0 false negatives at that threshold on RAGTruth-50.

## Decision Drivers

- Precision invariant: L3 must not flip hallucinations to grounded.
- L61 — derived confidence from decomposition outperforms self-reported
  LLM scores.
- Explainability: "4 of 5 facts supported" is more defensible than
  "0.82 probability supported".

## Considered Options

1. **Self-reported probability threshold** — use `prob_supported ≥ 0.5`
   or similar.
2. **Fact decomposition + ratio** — decompose into atomic facts, count
   supported, use ratio.
3. **Hybrid** — require both self-reported probability AND fact ratio
   above thresholds.

## Decision Outcome

**Chosen option: fact decomposition + ratio (Option 2).**

For each sentence, decompose into atomic facts (one per logical
proposition). For each fact, classify against the source as one of:
SUPPORTED, NOT_FOUND, CONTRADICTED, SHIFTED, or INFERRED. Compute the
supported ratio. Auto-clear at ratio ≥ 0.8 with 0 FN verified on
RAGTruth-50.

Facts that are SHIFTED (meaning the response asserts a variant of
what the source says) are counted separately and contribute to the
flag side of the ratio.

## Consequences

### Positive

- 76 % auto-clear at 100 % grounding precision on RAGTruth-50 (Exp 43).
- Evidence is per-fact, not per-sentence, which is the right grain
  for an operator reviewing a flag.
- The five-status taxonomy (SUPPORTED / NOT_FOUND / CONTRADICTED /
  SHIFTED / INFERRED) is the diagnostic contract for post-hoc
  analysis.

### Negative

- Decomposition requires an additional LLM pass (Gemini), increasing
  L3 cost.
- The five statuses are model-interpreted; their boundaries
  (especially SHIFTED vs CONTRADICTED) have edge cases that require
  ongoing calibration.

## More Information

- Experiment: Exp 43
- Principle: L61 — derived confidence > self-reported
- Checkpoint artifacts: `checkpoint_p1.json`, `checkpoint_p2.json`
- Related: ADR-0021 (MiniCheck sits alongside fact-counting within L3)
