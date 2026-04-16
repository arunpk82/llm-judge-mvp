---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0012: Standard artifact triplet — manifest + metrics + judgments

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The triplet has been
> the canonical artifact set since the early LLM Judge work and is
> documented throughout the Walkthrough.

Every evaluation run produces outputs. Without a standardised shape,
each capability (baseline, diff, drift, dashboard) would have to know
how to parse every run's outputs individually, and adding a new
capability would mean touching every existing run schema.

We need one contract between "running an evaluation" and "consuming its
results" — stable enough that capabilities can be built independently,
simple enough that new evaluation modes fit into it without rewriting.

## Decision Drivers

- Schema stability: changes to the artifact shape must be rare and
  versioned.
- Capability independence: baseline, diff, drift, and dashboards all
  read the same shape.
- Small surface: the smaller the triplet, the easier to reason about.
- Completeness: the triplet must carry enough information to reconstruct
  the run.

## Considered Options

1. **Single artifact** — one JSON file with everything.
2. **Triplet** — `manifest.json` (context), `metrics.json` (aggregates),
   `judgments.jsonl` (per-case detail).
3. **N-tuple** — separate files for each consumer's needs.

## Decision Outcome

**Chosen option: triplet (Option 2).**

- `manifest.json`: run context — dataset path and hash, rubric version,
  git SHA, timestamp, engine, random seed.
- `metrics.json`: aggregate metrics (flat keys; see ADR-0015).
- `judgments.jsonl`: one row per evaluated case with decision, scores,
  flags, and (for hallucination runs) sentence-level detail embedded
  (ADR-0016).

This mirrors the three natural questions: "what was this run?" (manifest),
"how did it do overall?" (metrics), "what did it decide on each case?"
(judgments).

## Consequences

### Positive

- Every consumer reads the same three files. Adding a consumer is
  additive, not breaking.
- The triplet is the unit of baseline promotion (ADR-0009).
- Legacy consumers that expected `results.jsonl` get a compatibility
  symlink to `judgments.jsonl`.

### Negative

- Three files instead of one means three points of truth to keep in
  sync. Mitigated by atomic write (all-or-nothing per run).
- Schema evolution of any of the three is load-bearing across the
  whole platform; every change needs an ADR.

## More Information

- Baseline implementation copies exactly these three files:
  `src/llm_judge/eval/baseline.py:376`
- Walkthrough Section 5.2 (CAP-2), Section 5.5 (CAP-5)
- Related: ADR-0015 (metrics schema), ADR-0016 (judgments schema)
