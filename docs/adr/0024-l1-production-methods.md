---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: C — Experiment
supersedes: ADR-0017
---

# ADR-0024: L1 production methods — substring, ratio, Jaccard (supersedes ADR-0017)

## Context and Problem Statement

ADR-0017 described L1 as a five-rule structural set: A1 exact match, B1
number mismatch, B2 entity missing, B3 negation flip, B4 qualifier shift.
Code review of `src/llm_judge/calibration/hallucination.py:108`
(`_l0_deterministic_match`) shows the actual L1 implementation is
different: exact substring match, plus `SequenceMatcher` ratio above
0.85, plus Jaccard token overlap above 0.80. The B1–B4 rules exist as
keys in `hallucination_pipeline_config.yaml` but have no implementation.

The 7 % / 21-sentence number from Exp 29 and 29b that L1 is credited
with was achieved by the substring + ratio + Jaccard implementation
that exists in code, not by the five-rule structural set. Meanwhile,
the L2 knowledge-graph ensemble (ADR-0020) does detect numeric
disagreements (G4), entity attribution errors (G1, G2), and negation
mismatches (G5) — i.e., L2 already covers what B1, B2, B3, B4 were
intended to cover.

ADR-0017 therefore describes a state of the world that is neither what
production does nor what we intend to build. It must be superseded.

## Decision Drivers

- **Code-doc reality.** ADRs must describe what the code does, not what
  the code aspires to.
- **Avoid duplication.** L2's property graphs already check the
  structural disagreements that B1–B4 were meant to catch. Implementing
  them at L1 would duplicate L2's job and complicate aggregation.
- **Preserve the validated 7 %.** The substring + ratio + Jaccard
  implementation is the validated method. Replacing it with B1–B4
  would mean re-validating L1 from scratch.

## Considered Options

1. **Implement B1–B4 in L1 to match ADR-0017.** Validate from scratch;
   resolve aggregation conflicts where L1 and L2 both flag the same
   thing.
2. **Supersede ADR-0017 with this ADR; document the actual L1.** No
   code changes; the validated state is the production state.
3. **Keep ADR-0017 as the target; mark current L1 as interim.** Defer
   the decision; live with the doc-code mismatch.

## Decision Outcome

**Chosen option: Option 2.** Supersede ADR-0017. L1 is, and remains,
substring-based:

- **A1 Exact substring** — response sentence appears verbatim in source
  (case-folded, whitespace-normalised) → grounded.
- **A2 Sequence ratio** — `SequenceMatcher(None, sent, src).ratio() > 0.85`
  against any source sentence → grounded.
- **A3 Jaccard token overlap** — token Jaccard with any source sentence
  greater than 0.80 → grounded.

Sentences not cleared by L1 cascade to L2 (ADR-0023). Structural
disagreements (numbers, entities, negations, qualifiers) are L2's
responsibility via the property-graph ensemble (ADR-0018, ADR-0020).

## Consequences

### Positive

- The ADR log tells the truth about L1.
- No duplicate flagging between L1 and L2 to aggregate.
- The validated 7 % clearance rate from Exp 29/29b stands without
  re-validation.
- L1 stays the cheapest layer — three deterministic string operations,
  no model load, no graph traversal.

### Negative

- The five `b*` keys in `hallucination_pipeline_config.yaml` are now
  vestigial. They must be removed in a config cleanup, with a new
  ADR if their removal is non-trivial.
- Anyone who read ADR-0017 and built mental models around B1–B4 must
  re-read this ADR. The supersession link makes this discoverable.

## More Information

- Implementation: `src/llm_judge/calibration/hallucination.py:108`
  (`_l0_deterministic_match`)
- Related: ADR-0017 (superseded), ADR-0018 (L2 multi-pass extraction),
  ADR-0020 (L2 flag-wins aggregation), ADR-0023 (cascade resolution)
- Experiments: Exp 29, Exp 29b (validated current implementation)
