# LLM-Judge

An evaluation platform for LLM outputs. Scores responses against
rubrics across 28 properties using a four-layer cascade (rules,
knowledge graphs, factual-consistency models, LLM judges) with
governance, lineage, and auditable attestation.

## Quick start

    make demo

This runs the platform end-to-end on a single example evaluation,
showing each capability's contribution with timing and integrity.
The demo output is saved to `reports/demos/<timestamp>/` for review.

## Next steps

- [Getting started](docs/GETTING_STARTED.md) — install and setup
- [Developer guide](docs/DEV_GUIDE.md) — daily workflows
- Architecture portal — see the living architecture document
