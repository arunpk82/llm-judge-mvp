---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0029: Resolve the layer-name legacy in the cascade code

## Context and Problem Statement

The cascade architecture (ADR-0002) names four layers L1, L2, L3, L4 in
the order Rules, Patterns (knowledge graph), Classifiers, LLM-as-Judge.
The code in `src/llm_judge/calibration/hallucination.py` carries names
from an earlier version of the cascade where the numbering was
different:

- `_l0_deterministic_match` is what current ADRs call L1.
- `_l1_gate1_check` (MiniLM whole-response embedding similarity) is what
  the current cascade calls part of L3.
- `_l2a_minicheck` and `_l2_nli_check` are what the current cascade
  calls L3.
- `_l3_graphrag_check` is a legacy implementation superseded by L2 (the
  knowledge graph ensemble in `hallucination_graphs.py`).
- `_l4_gemini_check` matches the current L4.

A new contributor reading this code reasonably concludes the layer
numbering means something it does not. Comments help; names mislead.

## Decision Drivers

- **Newcomer onboarding.** A contributor opening
  `hallucination.py` should not need an oral history to understand the
  layer mapping.
- **ADR–code consistency.** Every ADR that says "L2" should refer to the
  same thing as every code symbol that says "L2".
- **Refactor cost vs payoff.** Renaming is mechanical but touches every
  test, every grep, every contributor's mental model.
- **Reversibility.** Renaming is a one-way change. Worth doing once,
  cleanly, with the new ADR landing alongside the rename.

## Considered Options

1. **Status quo.** Names stay; comments and docstrings explain the
   mapping. Cheapest; slowest to onboard newcomers.
2. **Rename code symbols to match cascade architecture.** `_l0_*` →
   `_l1_*`, current `_l1_gate1_check` → `_l3_minilm_gate_check`,
   `_l2a_*` and `_l2_*` → `_l3_*`, `_l3_graphrag_check` removed (it is
   superseded), `_l4_*` stays.
3. **Rename cascade architecture to match code.** Update ADRs and
   docs to use the historical numbering.

## Decision Outcome

**Chosen option: Option 2 — rename code symbols to match cascade
architecture.**

The cascade architecture (L1 Rules → L2 Patterns → L3 Classifiers → L4
LLM-Judge) is the canonical numbering, established by ADR-0002 and
referenced by every subsequent ADR. The code is the outlier and must
move.

The rename:

- `_l0_deterministic_match` → `_l1_substring_match`
- `_l1_gate1_check` → `_l3_minilm_gate_check`
- `_l2a_minicheck` → `_l3_minicheck` (legacy; flag-gated per ADR-0027)
- `_l2_nli_check` → `_l3_deberta_nli` (legacy; flag-gated per ADR-0027)
- `_l3_graphrag_check` → removed (superseded by `l2_ensemble_check`
  in `hallucination_graphs.py`; legacy paths under ADR-0027 reference
  the new graph code)
- `_l4_gemini_check` → unchanged

Layer-stat keys in `HallucinationResult.layer_stats` are renamed in
the same commit: `"L1"`, `"L2"`, `"L2_flagged"`, `"L3"`, `"L3_flagged"`,
`"L4"`, `"L4_flagged"`. Existing keys that do not survive the rename
(e.g., the historic `"L3_gate1_fail"`) are removed.

The rename is a single atomic commit. The PR description points at this
ADR. Tests must be updated in the same PR. The rename happens *before*
ADR-0027's L3 fact-counting work, so that the new fact-counting code
lands with the correct names from day one.

## Consequences

### Positive

- ADR-code consistency restored. A grep for `l3_` finds the L3
  implementations, not the legacy MiniLM gate.
- New contributors can read the code and the ADRs together without
  translation.
- Future ADRs do not need to footnote the legacy names.

### Negative

- A large mechanical change touches many files. PR is noisy.
- Anyone with in-flight work against the old symbol names must
  rebase.
- Git blame on renamed symbols loses one hop of history. Documented
  in the commit message.

## More Information

- Implementation files affected: `src/llm_judge/calibration/hallucination.py`,
  associated unit tests, `hallucination_pipeline_config.yaml` if any
  layer-name strings appear there
- Related: ADR-0002 (cascade architecture — canonical), ADR-0027 (lands
  after this rename)
