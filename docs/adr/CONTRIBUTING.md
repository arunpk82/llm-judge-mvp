# How to write an ADR

## When to write one

Write an ADR when you are about to make a decision that:

1. Affects the structure of the system (a capability boundary, a pipeline
   stage, a data contract between modules), **or**
2. Changes a key quality attribute (reliability, observability, performance,
   testability, security, maintainability), **or**
3. Is difficult or expensive to reverse.

If the decision fails all three tests, it probably does not need an ADR.
Examples of things that *do not* need ADRs: naming conventions for local
variables, choice of logging library when existing code already uses one,
implementation of a specific function.

Examples of things that *do* need ADRs: choice of serialization format for an
inter-module contract, selection of an external model family, change to the
cascade resolution rule, introduction of a new layer in the pipeline.

## How to write one

1. Pick the next available number. Look at the highest existing ADR in this
   folder and add one. Numbers are sequential and never reused.
2. Copy `adr-template.md` to `NNNN-kebab-case-title.md`.
3. Fill in the template. Sections marked *optional* may be omitted if they
   do not apply.
4. Set `status: proposed` until the decision is adopted. Open a PR. Discuss.
5. When accepted, change `status` to `accepted` and merge.
6. Add an entry to the table in `README.md`.

## Immutability rule

Once an ADR is in `accepted` status, **do not edit it**. If the decision
changes:

1. Write a new ADR with the updated decision.
2. In the new ADR, add a reference to the one being replaced:
   *"This ADR supersedes ADR-NNNN."*
3. In the old ADR, change the status to `superseded by ADR-NNNN` and link
   to the replacement. This is the only edit allowed on an accepted ADR.

The point of immutability is that the decision log tells the truth about
what was decided when, and why. Editing accepted ADRs destroys that history.

## One decision per ADR

If you find yourself documenting two decisions in one ADR, split it. The
exception is a decision and its immediate implications — those belong
together.

## Honest rationale

The *Considered Options* section should list alternatives that were
genuinely considered, not straw men. If you did not consider an alternative,
do not pretend you did. If the decision was made under time pressure or
with incomplete information, say so.

## Retroactive ADRs

If you are writing an ADR for a decision that was made before the ADR
system existed, mark this clearly in the *Context* section. Write the
decision as it was made (not as you would make it today with hindsight).
If the original rationale is not documented anywhere, write "original
rationale reconstructed from code; no contemporaneous record found."
