---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0010: Diff engine with per-response flip tracking

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The diff engine is
> documented as EPIC 4.2 in the Walkthrough and implemented in
> `src/llm_judge/eval/diff.py`.

Comparing a candidate evaluation run to the golden baseline by
aggregate metrics alone is unsafe. A rule change that causes 50
responses to flip pass→fail and another 50 to flip fail→pass leaves
Kappa, F1, and precision essentially unchanged — yet 100 responses
have changed verdict. The aggregate hides the compensating error.

The diff engine must therefore surface both aggregate metric deltas
and per-response flips, with the flips quantified and inspectable.

## Decision Drivers

- False-positive-rate and false-negative-rate must both be trackable;
  aggregate metrics can mask movement between them.
- Engineers reviewing a change need to see *which* responses changed,
  not just *how many*.
- Flip tracking must scale to thousands of responses without blowing
  the memory budget.

## Considered Options

1. **Metric deltas only** — diff the two `metrics.json` files.
2. **Metric deltas + flip count** — add a count of flipped responses.
3. **Metric deltas + per-response flip detail** — surface each flipped
   response with its before/after verdict.

## Decision Outcome

**Chosen option: metric deltas + per-response flip detail (Option 3).**

The diff output contains three sections: `metrics.deltas` (per-key
baseline/candidate/delta triples), `decision_flips` (list of responses
whose `judge_decision` changed), and `score_deltas`/`flag_diffs`
(changes within responses that did not flip the top-level verdict).

## Consequences

### Positive

- Compensating changes are caught.
- A reviewer can jump from the aggregate delta to the specific cases
  that explain it.
- The diff is structured data, not text, so downstream tools
  (dashboards, alerts) can consume it directly.

### Negative

- Diff output grows linearly with dataset size. For very large
  datasets, streaming may be needed (currently out of scope).
- The schema for `judge_decision` is now load-bearing for regression
  detection; changes to it require an ADR.

## More Information

- Implementation: `src/llm_judge/eval/diff.py`
- Walkthrough EPIC 4.2
- Related: ADR-0009 (baseline), ADR-0011 (CI gate consumes diff output),
  ADR-0015 (flat metrics keys enable clean per-key deltas)
