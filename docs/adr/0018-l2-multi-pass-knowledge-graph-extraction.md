---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
---

# ADR-0018: L2 multi-pass knowledge graph extraction

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The design emerged from
> Exp 30, 30-2b, 30-2c, and 31, with learning L13 ("One Prompt One Task")
> as the underlying principle.

L2 is the knowledge-graph layer of the cascade. It operates on fact
tables extracted from source documents by an LLM (Gemini 2.5 Flash; see
ADR-0019). The question: should a single prompt extract all fact types
at once, or should extraction run as multiple single-task passes?

Single-prompt extraction is tempting — one API call, one response to
parse. But early experiments showed that asking Gemini to do multiple
extraction tasks in one prompt degraded every task. Entity extraction
was sloppier when combined with relationship extraction. Number extraction
missed values when negation extraction was also requested.

## Decision Drivers

- Extraction precision dominates downstream graph quality; sloppy
  extraction propagates into wrong L2 verdicts.
- L13 ("One Prompt One Task") — multi-task prompts degrade quality.
- Cost matters but is secondary to correctness at this layer; L2 runs
  once per source document and is cached.

## Considered Options

1. **Single multi-task prompt** — one call extracts entities, events,
   relationships, numbers, negations.
2. **Five single-task passes** — one call per extraction type.
3. **Two medium-task passes** — group related extraction types.

## Decision Outcome

**Chosen option: five single-task passes (Option 2).**

- **P1** entities
- **P2** events (subject–verb–object)
- **P3** relationships
- **P4** numbers (with units and context)
- **P5** negations and corrections

Each pass uses its own prompt optimised for that task. P2 cross-
references P1's entity list to resolve actors; otherwise passes are
independent. A sixth verification pass (P6) is run optionally to
dedup and canonicalise.

## Consequences

### Positive

- Extraction quality approaches single-task prompt quality on each
  dimension.
- Each pass is independently iterable — improving entity extraction
  does not require retesting number extraction.
- The passes cache at the source-document level; re-extraction cost
  is amortised.

### Negative

- Five API calls per source where one would have sufficed. Offset by
  cross-document caching.
- Prompt library is larger; five prompts to maintain and version.

## More Information

- Experiments: Exp 30, Exp 30-2b, Exp 30-2c, Exp 31
- Principle: L13 — "One Prompt One Task"
- Fact tables artifact: `experiments/exp31_multipass_fact_tables.json`
- Related: ADR-0019 (which model runs these passes), ADR-0020 (ensemble
  aggregation over the extracted graphs)
