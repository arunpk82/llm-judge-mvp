---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0004: 28 properties organised into 6 categories

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The taxonomy has been
> in effect since approximately early 2025 and is documented in the 28
> Metrics Evaluation Reference v2.1.

Evaluating LLM-generated responses requires a structured set of quality
dimensions. A flat list of "quality checks" is hard to reason about; teams
cannot tell whether the list is complete, nor can they prioritise which
checks to build first. A taxonomy is needed so that coverage can be
measured and gaps can be identified.

Existing academic and industry taxonomies (HELM, Holistic Evaluation,
RAGAS, TruthfulQA) each emphasise different dimensions. We needed a
taxonomy that was coherent for a production evaluation platform, not
just for a benchmark paper.

## Decision Drivers

- Completeness: every failure mode we care about must fit somewhere.
- Non-overlap: each property belongs in one category.
- Stability: the taxonomy should not churn as new properties are added.
- Communicability: a stakeholder should understand the categories without
  a glossary.

## Considered Options

1. **Flat list of properties** — e.g., 28 items in a bullet list.
2. **Taxonomy from an existing benchmark** — adopt HELM's or RAGAS's
   taxonomy wholesale.
3. **Custom six-category taxonomy** — design categories that fit the
   specific problem space of the LLM Judge platform.
4. **Tag-based** — properties have one or more tags; no hierarchy.

## Decision Outcome

**Chosen option: custom six-category taxonomy.**

The categories are: Faithfulness, Semantic Quality, Safety, Task Fidelity,
Robustness, Performance. Each of the 28 properties belongs to exactly one
category. Categories align to stakeholder concerns — a safety officer
cares about "Safety"; a product manager cares about "Task Fidelity"; a
platform engineer cares about "Performance." This mapping makes the
taxonomy legible to people with different responsibilities.

## Consequences

### Positive

- Each property has one home; there is no ambiguity about where a new
  failure mode goes.
- Coverage discussions become category-by-category, which is tractable.
- Stakeholder communication is easier because categories map to roles.

### Negative

- Edge cases exist — "hallucination" touches both Faithfulness and
  Semantic Quality. We place it in Faithfulness by definition.
- If a new top-level concern emerges (e.g., "Cost Awareness"), the
  taxonomy must be extended, which requires a new ADR.

## More Information

- 28 Metrics Evaluation Reference v2.1
- Related: ADR-0003 (two-gate architecture handles scoring within each
  property)
