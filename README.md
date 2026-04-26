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

## Run a batch demonstration

    make demo-batch          # full RAGTruth-50, ~7-10 minutes
    make demo-batch-quick    # first 10 cases, ~1-2 minutes

This runs the platform's Control Plane on real benchmark cases with full
per-capability and per-sub-capability visibility. The aggregated report
is saved to `reports/batch_runs/<batch_id>/aggregated_report.html`.

> **Note:** `make demo-batch` is the canonical Runner-governed batch path.
> `tools/run_ragtruth50.py` is a legacy script that bypasses the Control
> Plane and is being retired in a future cleanup packet — prefer
> `make demo-batch` for any new work.

Other batches and custom case files:

    make demo-batch-benchmark BENCHMARK=halueval
    make demo-batch-file FILE=examples/batch_input_example.yaml

## Next steps

- [Getting started](docs/GETTING_STARTED.md) — install and setup
- [Developer guide](docs/DEV_GUIDE.md) — daily workflows
- Architecture portal — see the living architecture document
