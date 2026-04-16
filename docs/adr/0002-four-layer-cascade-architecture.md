---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0002: Four-layer cascade architecture for hallucination detection

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16 to document a decision
> that has been in effect since approximately 2025. Reconstructed from
> the LLM Judge Architecture Walkthrough (CAP-2 and CAP-7), the pipeline
> config `hallucination_pipeline_config.yaml`, and the experiment series
> Exp 17b–43.

Hallucination detection over LLM outputs has conflicting requirements.
It must be cheap to run on every sentence at scale, which rules out
calling a frontier model on every sentence. It must be reproducible
across runs, which strains any single non-deterministic approach. It
must achieve high recall on subtle cases, which rules out rules alone.
And it must expose evidence for every verdict, so a QA Lead can defend
the decision.

No single technique satisfies all four requirements. We needed an
architecture that could compose techniques, each playing to its
strengths, with clear handoff rules.

## Decision Drivers

- Cost: LLM-judge per sentence is expensive; most sentences should never
  reach the LLM.
- Precision: deterministic layers can achieve 100 % precision on the
  subset of cases they can decide.
- Recall: subtle semantic hallucinations require semantic models or LLMs.
- Evidence trail: every verdict must be auditable.
- Independent validation: each layer must be testable on its own.

## Considered Options

1. **Single-LLM judge** — call a frontier model on every sentence and
   trust its verdict.
2. **Rules-only** — deterministic checks on the response; anything not
   cleared is flagged.
3. **Classifier-only** — run an NLI model or MiniCheck on every sentence
   and trust its verdict.
4. **Four-layer cascade** — L1 Rules → L2 Patterns (knowledge graph) →
   L3 Classifiers → L4 LLM-as-Judge, with sentences skipping remaining
   layers once resolved.

## Decision Outcome

**Chosen option: four-layer cascade.**

Each layer contributes a different slice of coverage at 100 % precision
in the experiments: L1 cleared ~7 % (Exp 29/29b), L2 cleared ~19 %
additional (Exp 30/31), L3 cleared ~76 % standalone (Exp 43, 0 FN at
grounding ratio ≥ 0.8). The layers compose naturally because each
later layer is more expensive and more capable than the previous.
Sentences that one layer can decide at high precision never need to
reach the next.

The cascade is also the industry-standard shape for this problem, which
matters for knowledge transfer and for recruiting contributors.

## Consequences

### Positive

- Cost scales with ambiguity, not with total sentence count.
- Each layer can be validated, tuned, and replaced independently.
- Funnel reporting (how many sentences each layer decided) is a natural
  observability output.
- The architecture admits later extension (L5 human adjudication) without
  restructuring.

### Negative

- Four layers is more moving parts than one. Integration bugs are harder
  to diagnose (see ADR-0007 for the current integration gap).
- Pipeline configuration becomes important (see ADR-0006).
- Layer enable/disable combinations multiply the test matrix.

## More Information

- LLM Judge Architecture Walkthrough, CAP-2 and CAP-7
- Pipeline config: `hallucination_pipeline_config.yaml`
- Experiments: Exp 17b–43 (see 28 Metrics Evaluation Reference v2.1)
- Related: ADR-0007 (scope of current phase), ADR-0023 (cascade
  resolution rule)
