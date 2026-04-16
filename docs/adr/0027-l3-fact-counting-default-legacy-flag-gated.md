---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
supersedes: ADR-0021, ADR-0022
---

# ADR-0027: L3 fact-counting as default; MiniCheck and DeBERTa flag-gated (supersedes ADR-0021 and ADR-0022)

## Context and Problem Statement

ADR-0021 documented L3 as MiniCheck primary + DeBERTa NLI fallback —
the implementation currently in `src/llm_judge/calibration/hallucination.py`.
ADR-0022 documented Exp 43's fact-counting as the L3 design. These two
ADRs describe two different L3s and were both marked accepted. The
production L3 is what 0021 describes; the validated-and-superior L3 is
what 0022 describes.

Exp 43 produced 76 % auto-clear at 100 % grounding precision on
RAGTruth-50, with 0 false negatives at grounding ratio ≥ 0.8. This is
materially better than what MiniCheck + DeBERTa achieved. The
fact-counting design is the L3 that should run in production.

The implementation lives only in `experiments/exp43_end_to_end.py`. It
must be ported into `src/`. The MiniCheck + DeBERTa code already in
`src/` represents real integration work and validated experimentation
(L24 — purpose-built > generic NLI). It should not be deleted.

The pattern from ADR-0026 says: every behavioural switch is a flag,
defaults are explicit, the rollback path stays alive.

## Decision Drivers

- **Truth in production.** The L3 that runs in production should be the
  L3 that the experiments validated as best.
- **Reversibility.** A future failure of fact-counting in production
  should be reversible by flag flip, not by code restoration from git
  history.
- **No phantom limbs.** Flagged-off code must remain tested (ADR-0028).

## Considered Options

1. **Replace and delete.** Port fact-counting into `src/`; delete
   MiniCheck + DeBERTa code. Rollback via `git revert`.
2. **Replace and preserve.** Port fact-counting into `src/`; keep
   MiniCheck + DeBERTa code in `src/` behind feature flags; CI tests
   exercise both paths.
3. **Run in parallel.** Both L3 implementations run simultaneously;
   verdicts compared.

## Decision Outcome

**Chosen option: Option 2 — flag-gated coexistence.**

Three new flags in `hallucination_pipeline_config.yaml` under
`l3_classifiers`:

- `l3_factcounting_enabled: true` (default) — Exp 43 implementation:
  decompose sentence into atomic facts via Gemini fact-counting prompt,
  classify each as SUPPORTED / NOT_FOUND / CONTRADICTED / SHIFTED /
  INFERRED, plus four micro-passes (entity binding, numbers/dates,
  semantic shifts, additions), aggregate into final L3 verdict.
- `l3_minicheck_enabled: false` (default off, superseded) — legacy
  MiniCheck Flan-T5-Large per-sentence factual consistency.
- `l3_deberta_enabled: false` (default off, superseded) — legacy
  DeBERTa-v3 NLI fallback.

The schema (ADR-0026) declares: at least one L3 flag must be true; the
three are mutually exclusive in production routing (the first true flag
wins) but all may be enabled simultaneously in test mode for shadow
comparison.

The MiniCheck + DeBERTa code stays in `src/` under
`src/llm_judge/calibration/hallucination_legacy.py` (or a similar
namespace). It is exercised by CI per ADR-0028.

Implementation work for porting Exp 43 fact-counting into `src/`:

1. Move `PROMPT_FACT_COUNT`, `PROMPT_P1_ENTITY`, `PROMPT_P2_NUMBERS`,
   `PROMPT_P3_SEMANTIC`, `PROMPT_P4_ADDITIONS` into
   `src/llm_judge/calibration/prompts.py` or a new `prompts_l3.py`.
2. Implement `_l3_factcounting_check(sentence, source_doc)` in
   `hallucination.py`, returning the same `(verdict, confidence,
   evidence)` shape as existing L3 functions.
3. Implement four micro-pass functions and the ensemble aggregator.
4. Wire the dispatch in `check_hallucination()` to honour
   `l3_factcounting_enabled`.
5. Add unit tests covering each micro-pass and the aggregator.
6. Add an integration test that runs the full L3 fact-counting path
   on a small fixture and asserts the verdict.

## Consequences

### Positive

- The validated L3 reaches production.
- Rollback to MiniCheck + DeBERTa is a config flag flip, no code
  change required.
- ADR-0021 and ADR-0022 are reconciled; the supersession is explicit
  and discoverable.
- The pattern from ADR-0026 is exercised on its first major
  application; teaches the team how flag-gated transitions work.

### Negative

- `src/` carries two L3 implementations. Some maintenance overhead on
  imports and dispatch.
- Fact-counting depends on Gemini API calls in the L3 hot path. This
  is an addition to the runtime cost surface (previously L3 was
  in-process model inference only).
- The Gemini dependency means L3 latency increases under fact-counting.
  This must be measured in the calibration run; if it exceeds the
  acceptable budget for production callers, a derived `ADR-00xx` will
  decide between caching, batching, or routing back to MiniCheck under
  load.

## More Information

- Source of fact-counting design: `experiments/exp43_end_to_end.py`
- Source of validation evidence: `experiments/exp43_results/`
  (`checkpoint_p1.json`, `checkpoint_p2.json`, etc.); 76 %, 0 FN @ 0.8
- Related: ADR-0021 (superseded), ADR-0022 (superseded), ADR-0026
  (feature-flag pattern), ADR-0028 (CI tests for flagged-off paths)
