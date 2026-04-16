---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0013: RAGTruth-50 as the fixed hallucination benchmark

## Context and Problem Statement

> This decision was made on 2026-04-15 and merged as PR #148. This ADR
> was written retroactively on 2026-04-16 to document it in the formal log.

Hallucination F1 numbers from different experiments must be comparable.
If Exp 30 measures on 200 responses and Exp 37 measures on 300 responses,
the F1 values cannot be directly compared, and architectural decisions
built on them may be wrong. Across 43 experiments, this comparability
was managed by convention; we need it to be enforced by artifact.

## Decision Drivers

- Cross-experiment comparability: every F1 comparison must be on the
  same sentences.
- Statistical sufficiency for operational decisions (not conclusive, but
  sufficient for cascade calibration).
- Small enough to run end-to-end in minutes (CI friendliness).
- Balanced hallucinated/clean distribution.

## Considered Options

1. **Full RAGTruth** (~18 000 annotated cases) — maximum statistical
   power but slow and expensive.
2. **Fixed 50-response subset** — fast, balanced, reproducible.
3. **Rotating sample** — different sentences each run, averaged over
   time.

## Decision Outcome

**Chosen option: fixed 50-response subset (Option 2).**

`datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json` pins response
IDs 0–49 across 9 sources. After spaCy sentence splitting (ADR-0014),
this yields approximately 306 sentences with 18 hallucinated and 288
clean. The benchmark file itself contains rules: "NEVER change the
response_ids list"; "ALL experiments comparing F1 must use this exact
set"; "Sentence splitting must use spaCy doc.sents (not regex)."

The benchmark is accepted as operationally sufficient for cascade
calibration and CI regression gating. A future adversarial 30 k-case
expansion is on the Wave 7 backlog for broader statistical claims.

## Consequences

### Positive

- F1 numbers across experiments are now directly comparable.
- CI runs in under 10 minutes.
- The benchmark file is versioned; changing it requires a version bump
  and an ADR superseding this one.

### Negative

- 306 sentences with 18 hallucinated is statistically thin for fine-
  grained claims. Decisions requiring stronger evidence (e.g., gating
  L4 promotion) will require the 30 k expansion.
- The fixed subset introduces selection bias; cascade behaviour on
  RAGTruth-50 may not reflect behaviour on arbitrary production input.

## More Information

- Benchmark: `datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json`
- Loader: `experiments/benchmark_loader.py`
- Related: ADR-0014 (spaCy splitting required by this benchmark),
  ADR-0008 (dataset governance applies)
