---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0030: The calibration run is the baseline-seeding event

## Context and Problem Statement

The L1→L2→L3 cascade has been validated layer-by-layer in 43
experiments but has never run as a composed cascade through production
callers (`IntegratedJudge.evaluate_enriched`, `BenchmarkRunner`). The
F1, layer-clearance percentages, and runtime characteristics of the
*composed* cascade are unknown. Standalone L3 cleared 76 % at 100 %
precision on RAGTruth-50, but standalone L3 sees all 306 sentences;
production L3 sees only what L1 and L2 leave undecided.

The CI regression gate (ADR-0011) requires a baseline. There is no
baseline for the cascade because the cascade has never produced one.
This is a chicken-and-egg situation that must be resolved by a
deliberate, one-time event: the calibration run.

The calibration run is not just "a run that happens to be first." It
is an instrument — its purpose is to *measure* the cascade's
production behaviour for the first time, and to seed the baseline
that all subsequent runs are gated against. The discipline around it
must reflect that.

## Decision Drivers

- **No silent baseline.** Promotion of the calibration run as the
  baseline must be an explicit, documented act, not a side-effect of
  the first `make benchmark` invocation.
- **Sanity gate.** If the calibration run produces numbers wildly
  inconsistent with experimental expectations, that indicates a wiring
  bug, not a true baseline. Promotion must be blocked.
- **Reproducibility.** The run must be reproducible from the recorded
  manifest — same code, same config, same fact tables, same dataset.
- **Honest accounting.** The numbers from this run will set
  expectations for months. They must not be cherry-picked or repeated
  until favourable.

## Considered Options

1. **First run wins.** Whatever the first cascade run produces becomes
   the baseline. Operationally simplest; no editorial control.
2. **Calibration run with sanity gate.** A specific run is designated
   the calibration; it must satisfy explicit sanity criteria before
   being promoted as the baseline.
3. **Multiple-run average.** Run the cascade N times; promote the
   average as baseline.

## Decision Outcome

**Chosen option: Option 2 — calibration run with sanity gate.**

The calibration is a single, explicit event in the deployment phase:

1. **Pre-conditions.** All EPIC work (D1.1–D1.3, D2.1–D2.2, D3.1–D3.2,
   D5.1) is complete and merged. The cascade is fully wired,
   feature-flagged per ADR-0026 to default-on, and tested per
   ADR-0028. No pending work blocks the run.
2. **The run.** `make benchmark` is invoked on RAGTruth-50, with the
   default config and the master commit at the time of the run. The
   triplet (`manifest.json`, `metrics.json`, `judgments.jsonl`) is
   produced under `state/runs/calibration-<timestamp>/`.
3. **Sanity gate.** Before promotion, the run must satisfy:
   - L1 invocations > 0; L2 invocations > 0; L3 invocations > 0.
     (Proves the cascade is wired.)
   - L1 cleared ≥ 5 %; L2 cleared ≥ 10 %; L3 cleared ≥ 50 %.
     (Loose lower bounds; if any layer is wildly below experimental
     expectations, it points to a wiring bug, not a real result.)
   - Per-layer precision ≥ 95 % on RAGTruth-50 ground truth.
     (100 % was the experimental result; 95 % allows for
     splitting-boundary differences without being so loose that a
     real precision regression slips through.)
   - `manifest.json` contains git SHA and config SHA.
4. **Promotion.** If the sanity gate passes, `make baseline-promote`
   is invoked with the calibration run as source. The promotion is
   recorded with this ADR's number as the justification.
5. **Record.** The calibration result is appended to the 28 Metrics
   Evaluation Reference document and to a new entry in the baseline
   registry.

If the sanity gate fails, the cascade is debugged, the failing
sub-system is fixed, and the calibration is re-run. Each calibration
attempt is preserved (not overwritten). The promoted calibration is
the one that passed; the predecessors are kept as forensic record.

## Consequences

### Positive

- The transition from "no baseline" to "have baseline" is a single,
  documented, deliberate event.
- Wiring bugs cannot silently become the baseline.
- The calibration run becomes the artifact future ADRs and runs
  reference: "performance has not regressed since the calibration of
  YYYY-MM-DD."
- Multiple calibration attempts are preserved, providing visibility
  into how many iterations the cascade needed.

### Negative

- A failed sanity gate blocks the deployment phase from completing.
  This is correct — silent baselining is worse — but it means the
  phase has no fixed end date.
- The sanity gate thresholds are themselves judgement calls. They
  encode an expectation that may be wrong. If the calibration legitimately
  produces numbers below the gate (e.g., L1 cleared 4 % because RAGTruth-50
  truly has fewer substring-resolvable sentences than expected), the
  thresholds need revision and a new ADR.
- The promotion is not automatable; it requires human attention to
  the sanity-gate report. This is by design.

## More Information

- Run target: `make benchmark` (EPIC D5.1)
- Promotion target: `make baseline-promote` (existing, EPIC 4.1)
- Related: ADR-0009 (immutable baseline), ADR-0011 (CI gate consumes
  the baseline), ADR-0013 (RAGTruth-50 — the calibration dataset),
  ADR-0027 (cascade implementation that this run measures)
- Revisit trigger: when adding a second benchmark dataset, this ADR's
  pattern must be applied to seed that benchmark's baseline as well
  (likely a new ADR per benchmark, referencing this one).
