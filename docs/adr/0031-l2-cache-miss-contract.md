---
status: proposed
date: 2026-04-21
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0031: L2 graph cache miss contract

## Context and Problem Statement

ADR-0025 specified the *hit* side of the L2 graph cache: fact tables are
extracted at dataset registration, keyed by source SHA-256, and consulted
from the cache at runtime. It did not specify what happens on a *miss*.

Today, the cache-consult site in
`src/llm_judge/calibration/hallucination.py:674-702` treats a miss as a
quiet fall-through: it sets `layer_stats["L2_cache_miss"] = 1`, logs
`l2.cache_miss` at `DEBUG`, and continues the cascade without L2. The
underlying `GraphCache.get()`
(`src/llm_judge/calibration/graph_cache.py:87-136`) returns `None`
indistinguishably for three different failure modes:

- **not_exist** — no cache file for this source hash (preseed drift, or
  the dataset was never registered through ADR-0025).
- **expired** — file present but past TTL (logged at DEBUG inside
  `graph_cache`).
- **corrupt** — file present but JSON is unreadable (logged at WARNING
  inside `graph_cache`, but the caller collapses it back to `None`).

A silent miss is indistinguishable from L2 correctly abstaining. That is
a P02-Trust problem: a regression in dataset registration (stale preseed,
hash-algorithm change, partial cache corruption) would silently turn L2
into a no-op, and the cascade would keep shipping — with downstream
metrics that look fine because L3 and L4 pick up what L2 dropped. The
monitoring-system-must-monitor-itself principle (L65) says the cascade
must surface its own degradation rather than hide behind other layers.

`hallucination.py` has no import of `extract_fact_tables` or
`kg_extraction`, so there is no runtime path that could fall back to
live Gemini extraction even if we wanted one; the decision is between
*observable abstention* and *silent abstention*, not between *cache* and
*fallback*.

## Decision Drivers

- **P02 Trust.** A degraded L2 must not masquerade as a working L2.
  Misses must be visible in run artifacts, not just DEBUG logs.
- **Hot-path cleanliness (ADR-0025).** Runtime must not call Gemini.
  The decision space for a miss is bounded to abstain-and-surface.
- **Fail-open, not fail-loud.** A missing fact table for one source
  must not abort the whole evaluation — other cases still need to run.
- **Diagnosability.** The operator must be able to tell *why* a miss
  happened (not_exist vs expired vs corrupt) without re-running.
- **CI gate coverage (ADR-0028).** Whatever contract we pick must be
  testable by a CI smoke so it cannot silently regress.

## Considered Options

1. **Status quo — silent miss.** DEBUG log + per-case
   `layer_stats["L2_cache_miss"]` only. No run-summary aggregation, no
   WARNING, no reason breakdown.
2. **Silent abort.** Treat a miss as a fatal registration-contract
   violation and abort the run. Forces the operator to fix preseed
   before any evaluation can proceed.
3. **Observable abstention.** Miss emits a WARNING with the source-hash
   prefix and miss reason, bumps `layer_stats["L2_cache_miss"]`, and
   aggregates into the run summary. L2 abstains for that case; cascade
   continues.
4. **Runtime extraction fallback.** L2 calls Gemini on miss. Rejected
   up-front by ADR-0025 (hot path must not touch external APIs) and by
   the current code shape (no extraction import in `hallucination.py`).

## Decision Outcome

**Chosen option: Option 3 — observable abstention.**

On an L2 cache miss, the cascade MUST:

1. Emit a `WARNING` log containing the source-hash prefix (first 12
   hex chars, to match `graph_cache` log format) and the miss reason,
   drawn from the enum `{"not_exist", "expired", "corrupt"}`.
2. Increment `layer_stats["L2_cache_miss"]` for the case, and record
   the reason alongside the count so it survives into the per-case
   artifact.
3. Aggregate the per-case miss count into the run-level summary (e.g.
   `sentence_level_metrics.by_layer.L2.cache_miss_count` with a
   breakdown by reason), so the operator sees the total without
   grepping logs.
4. Continue the cascade as if L2 abstained on that case. L3 and L4
   run unchanged; the overall evaluation MUST NOT fail.

The cascade MUST NOT:

- Skip silently with only a DEBUG log (violates clause 1, the status
  quo failure this ADR exists to fix).
- Attempt live Gemini extraction on miss (violates ADR-0025 hot-path
  contract).
- Fail the overall evaluation because a single source had a miss
  (violates fail-open; isolates unrelated cases).

`GraphCache.get()` must be extended to return the miss reason alongside
the `None`-on-miss signal (e.g. `(None, reason)` or a small result
object), so the caller can emit clause 1 accurately instead of guessing.

## Consequences

### Positive

- A cache-miss rate above ~5% becomes a legible operator signal of
  preseed drift, hash-algorithm change, or cache corruption — today it
  is invisible without a log grep.
- Downstream metrics remain interpretable: L2-abstained cases are
  counted as abstention rather than pretending L2 ran.
- Satisfies L65 (monitoring-system-must-monitor-itself): the cascade
  surfaces its own degradation.
- Reason breakdown (not_exist vs expired vs corrupt) lets the operator
  decide whether to re-register, bump TTL, or rebuild the cache
  without guessing.
- The CI smoke in #173 pins the contract so a future refactor cannot
  silently regress to DEBUG-only logging.

### Negative

- `GraphCache.get()` signature widens — callers must handle a
  `(data, reason)` shape (or equivalent) instead of a raw `None`.
  One caller today (`hallucination.py:674-702`), so the blast radius
  is small.
- Run-summary schema gains a new field. Any consumer that strictly
  validates the summary schema needs to admit the addition; the
  existing schema tolerates additive keys.
- WARNING-level logs on miss raise the baseline log volume. Acceptable
  because miss should be rare under a healthy preseed; if it becomes
  noisy, that is itself the signal the contract is designed to
  surface.

## More Information

- Companion to ADR-0025 (registration-time extraction — the *hit*
  contract). This ADR is the *miss* contract.
- Related: ADR-0018 (multi-pass extraction), ADR-0020 (L2 ensemble),
  ADR-0028 (CI flag-combo coverage).
- Implementation tracking: #171 implements clauses 1–3 (WARNING log,
  per-case reason, run-summary aggregation). #173 provides the CI
  smoke that pins the contract. #172 covers the unit smoke for
  `GraphCache.get()`'s miss-reason plumbing.
- Code anchors at adoption time (master HEAD `24c6a95`):
  `src/llm_judge/calibration/hallucination.py:674-702` (cache-consult
  site), `src/llm_judge/calibration/graph_cache.py:87-136` (`_is_expired`
  and `get`).
- Revisit trigger: if a legitimate use case for runtime extraction
  appears (e.g. ad-hoc single-source evaluation outside registered
  datasets), this ADR and ADR-0025 must be revisited together.
