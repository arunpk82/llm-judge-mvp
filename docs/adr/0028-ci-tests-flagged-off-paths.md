---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0028: CI tests every major flag combination to prevent flag rot

## Context and Problem Statement

ADR-0026 commits the platform to feature-flag governance. ADR-0027
applies it: legacy L3 paths (MiniCheck, DeBERTa) live in `src/` with
their flags off by default. The benefit of flag-gated rollback only
holds if the flagged-off code actually works when the flag is flipped.

Code that does not run, rots. Imports break under newer dependency
versions. Function signatures change in callers without the flagged-
off path being updated. Two years later, an emergency rollback flips
the flag and discovers the code does not start.

The world-class solution is what Netflix calls "dark canaries" and Meta
calls "shadow mode" — flagged-off code is exercised by automated
systems even when it does not serve user traffic. Adapted to a smaller
scale: CI runs the test suite under every major flag combination, on
every build or at minimum nightly.

## Decision Drivers

- **Real rollback, not theoretical.** A rollback path is only real if
  the rolled-back code is known to work today.
- **Fail early, not in an emergency.** A flag-flip failure should
  surface in CI on the day it breaks, not on the day someone needs the
  rollback.
- **Discipline scales.** As more flags are added (ADR-0026 commits to
  this), the test surface grows; this ADR establishes the pattern
  before bloat sets in.

## Considered Options

1. **Test only the default flag combination.** Lowest CI cost; flagged-
   off paths rot.
2. **Test every individual flag flip.** Each flag tested by enabling it
   alone (or disabling it alone) in an otherwise-default config. N+1
   test runs for N flags.
3. **Test every combination.** Cartesian product of all flag values.
   2^N test runs; impractical past a handful of flags.
4. **Test named scenarios.** A small set of flag combinations
   corresponding to operationally meaningful states (current production,
   legacy rollback, all-experimental, all-disabled). Fixed cost,
   coverage of intended states.

## Decision Outcome

**Chosen option: Option 4 — test named scenarios.**

The platform maintains a named set of *test scenarios* in the test
suite, each scenario being a complete config combination plus an
expected behavioural assertion. Initially:

- **`current_production`** — defaults as shipped. Must pass full
  benchmark.
- **`l3_legacy_rollback`** — `l3_factcounting_enabled: false`,
  `l3_minicheck_enabled: true`, `l3_deberta_enabled: true`. Must run
  end-to-end without import errors and produce verdicts on a small
  fixture.
- **`l1_only`** — L2 and L3 disabled. Must produce verdicts using
  L1 substring rules alone; functions as the irreducible-fallback
  test.
- **`l2_only`** — L1 and L3 disabled, L2 enabled with pre-seeded fact
  tables. Tests L2 in isolation; useful for L2 regression detection.
- **`all_layers_off`** — all layers disabled. Must error explicitly
  with a clear message ("at least one layer must be enabled"), not
  silently produce empty verdicts.

A new Make target `make test-scenarios` runs all scenarios. The
existing `preflight` chain runs `current_production` plus
`l3_legacy_rollback` (the most likely rollback target). The full
scenario set runs nightly on CI.

Adding a new flag (per ADR-0026) requires adding or extending a
scenario that exercises it. This is enforced by ADR review, not by
tooling.

## Consequences

### Positive

- Every documented rollback path is known to work today.
- Flag rot is detected at the next CI run, not at emergency time.
- The scenario list doubles as documentation: reading it tells a new
  contributor what operational states the system is designed to
  support.
- Test cost stays bounded — five scenarios, not 32.

### Negative

- Scenarios need maintenance. As flags are added, scenarios must be
  extended; otherwise a new flag may not be tested in any scenario.
- Some flag combinations are not covered. A combination that is not in
  the named-scenarios list could rot. We accept this trade-off: the
  combinations we name are the ones we will actually use.
- CI time grows by the cost of running each scenario. Mitigated by
  using a small fixture for non-production scenarios.

## More Information

- Industry references: Netflix dark canaries; Meta shadow mode;
  Google's "all variants must build" rule
- Related: ADR-0026 (feature-flag governance), ADR-0027 (first ADR
  whose rollback this protects)
- Revisit trigger: if scenarios reach roughly 20, consider tooling-
  based combinatorial coverage (matrix CI) instead of named scenarios.
