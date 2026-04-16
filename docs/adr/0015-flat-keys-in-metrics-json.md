---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0015: Flat keys in metrics.json for per-key diff tolerance

## Context and Problem Statement

`metrics.json` is consumed by the diff engine (ADR-0010). Inspection of
`src/llm_judge/eval/diff.py:_diff_metrics()` shows the engine iterates
**top-level keys only** and passes each value through `_numeric()`.
Nested dicts return `None` from `_numeric()` and are silently dropped
from the delta computation.

The existing schema already contains `confusion_fail_positive: {tp, fp,
tn, fn}` as a nested object, which is present for inspection but not
actually diff-gated. This confirms the pattern: nested objects are
non-diffable.

For hallucination cascade metrics, we want per-key tolerance: a 0.01
drop in `sentence_f1` may be acceptable while a 0.01 drop in `l1_precision`
may not. Nested metrics cannot support this.

## Decision Drivers

- The diff engine is production code with many downstream consumers;
  changing it to recurse into nested keys has wide blast radius.
- Per-metric tolerance is the critical feature for a useful CI gate.
- Readability of `metrics.json` should not suffer; we can keep nested
  objects for inspection.

## Considered Options

1. **Flat keys only** — all diff-gated metrics at top level; no
   nesting.
2. **Nested keys, extend diff engine** — change `_diff_metrics` to
   recurse.
3. **Hybrid** — flat keys for gated metrics, nested objects allowed for
   inspection-only data.

## Decision Outcome

**Chosen option: hybrid (Option 3).**

All metrics that can be diff-gated live at the top level as flat keys
(`l1_precision`, `sentence_f1`, `response_recall`, etc.). Nested objects
are permitted for inspection-only data such as `confusion_fail_positive`.
The diff engine is not changed.

Concretely, the hallucination cascade will emit approximately 30 flat
keys at the top level plus nested confusion matrices for inspection.

## Consequences

### Positive

- Per-key tolerance works from day one without touching the diff engine.
- Existing consumers of `metrics.json` continue to work.
- The convention is simple: if you want it gated, put it at the top.

### Negative

- Top-level namespace fills up (~30 keys just for hallucination). Naming
  discipline becomes important to keep keys scannable
  (`l1_` / `l2_` / `l3_` prefixes for cascade layers).

## More Information

- Diff engine: `src/llm_judge/eval/diff.py`, `_diff_metrics`
- Related: ADR-0010 (diff engine), ADR-0012 (artifact triplet),
  ADR-0016 (judgments schema)
