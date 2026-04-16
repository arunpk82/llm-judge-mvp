---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: Foundation
---

# ADR-0001: Use MADR 4.0 for Architecture Decision Records

## Context and Problem Statement

As the LLM Judge platform has grown, architectural decisions have been made
across many chats, documents, and experiments. The reasoning behind these
decisions lives in session continuity briefs, scattered notes, and tribal
memory. This creates three concrete problems:

1. A decision made four months ago is difficult to retrieve when a new
   conversation revisits the same ground.
2. When reasoning is lost, the risk of a wrong feedback call or of making
   an existing solution redundant increases.
3. Future collaborators — human or AI — have no canonical source for why
   the system is the way it is.

We need a durable, searchable, version-controlled record of architectural
decisions that travels with the code.

## Decision Drivers

- Decisions must survive across chats, sessions, and team-member changes.
- The format must be lightweight (markdown), reviewable (diff-able in git),
  and accessible to non-authors.
- The industry already has a convergent answer; we should use it rather
  than invent our own.

## Considered Options

1. **Michael Nygard's original ADR template (2011)** — minimal: Title,
   Status, Context, Decision, Consequences.
2. **MADR 4.0** (Markdown Architectural Decision Records, released
   September 2024) — an evolution of Nygard's template with structured
   *Considered Options* and *Pros and Cons* sections.
3. **Y-Statements** (Sustainable Architectural Decisions) — a single
   structured sentence capturing a decision; very terse.
4. **Formless** — write decisions in whatever shape the author prefers,
   no template.

## Decision Outcome

**Chosen option: MADR 4.0.**

MADR 4.0 is the current world-class standard. It is recommended by Martin
Fowler, the Microsoft Azure Well-Architected Framework, Google Cloud's
Cloud Architecture Center, and AWS Prescriptive Guidance. It extends
Nygard's template with the explicit *Considered Options* section, which
forces the author to record what was rejected and why — the single most
valuable habit for future readers.

## Consequences

### Positive

- Every architectural decision has a fixed home, discoverable by reading
  the repo.
- The act of writing an ADR forces clarity about alternatives and costs.
- Supersession (rather than editing) preserves the history of how thinking
  evolved.
- New collaborators can read the ADR log to orient themselves.

### Negative

- Writing an ADR takes time (typically 30–60 minutes for a new decision).
- Retroactive ADRs require archaeological work to reconstruct.
- Over time the log grows; some tooling or tagging may eventually be
  needed to keep it navigable.

## More Information

- [MADR 4.0 template](https://github.com/adr/madr)
- [Martin Fowler on ADRs](https://martinfowler.com/bliki/ArchitectureDecisionRecord.html)
- [Microsoft Azure Well-Architected Framework — ADRs](https://learn.microsoft.com/en-us/azure/well-architected/architect-role/architecture-decision-record)
