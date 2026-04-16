---
status: superseded by ADR-0027
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0021: L3 classifier stack — MiniCheck primary, DeBERTa NLI fallback

> **Superseded.** This ADR is superseded by [ADR-0027](./0027-l3-fact-counting-default-legacy-flag-gated.md). MiniCheck and DeBERTa NLI remain in `src/` behind feature flags (`l3_minicheck_enabled`, `l3_deberta_enabled`, both default `false`); L3 fact-counting from Exp 43 is the new default.

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The stack emerged from
> the NLI-improvement experiment series, with learning L24 ("purpose-
> built factual-consistency models outperform generic NLI").

L3 is the classifier layer of the cascade. It receives sentences that
L1 and L2 could not resolve. The architectural question: which
classifier(s) does L3 use?

Generic NLI models (DeBERTa-v3 on MNLI) had been the standard choice.
However, experiments showed DeBERTa NLI alone achieved only 39 %
sentence coverage on RAGTruth-50. The NLI improvement experiments
introduced MiniCheck (Flan-T5-Large trained specifically for factual
consistency) which reached 78 % coverage — a +97 % improvement.

The two models have different failure modes. MiniCheck occasionally
returns "unknown" on cases where DeBERTa has a clear verdict. A
fallback pattern can capture the union.

## Decision Drivers

- Coverage: every sentence L3 cannot decide becomes expensive
  downstream.
- Precision invariant: L3 must hold 100 % precision when used as a
  gating classifier.
- Latency: L3 runs on every sentence that reaches it.
- L25: MiniCheck takes full document context — no retrieval step
  needed. L26: screening gates should be signals, not stoppers,
  when deeper verification is available.

## Considered Options

1. **DeBERTa NLI only** — the historical default.
2. **MiniCheck only** — best standalone coverage.
3. **MiniCheck primary + DeBERTa fallback** — MiniCheck first;
   on "unknown" verdict, DeBERTa decides.
4. **Ensemble** — both always, aggregate.

## Decision Outcome

**Chosen option: MiniCheck primary + DeBERTa fallback (Option 3).**

For each sentence:

1. Run MiniCheck (L3a). If verdict is grounded or hallucinated with
   confidence above threshold, accept it.
2. Otherwise run DeBERTa NLI (L3b). Its verdict becomes L3's output.

Sentences that L3 cannot decide (both models unknown) cascade to L4
(when L4 is enabled; currently deferred per ADR-0007).

## Consequences

### Positive

- L3 coverage is the union of MiniCheck and DeBERTa; ~76 % on RAGTruth-50
  at 100 % precision (Exp 43).
- Fallback pattern isolates each model's strengths without the weight
  of a full ensemble.
- Adding or replacing either layer is a local change; the interface
  stays stable.

### Negative

- Two models loaded in memory (~3.6 GB total). Startup cost matters
  for short-lived processes.
- DeBERTa NLI scores are bimodal (L23) — threshold tuning does not
  help much. Its role is narrow: decide when MiniCheck is unsure.

## More Information

- Learnings: L23 (DeBERTa bimodal), L24 (purpose-built > generic),
  L25 (MiniCheck context handling), L26 (screening gates as signals)
- Result: +97 % improvement over DeBERTa alone
- Related: ADR-0022 (the fact-counting design used alongside MiniCheck)
