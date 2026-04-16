---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0006: Config-driven pipeline via YAML

## Context and Problem Statement

> This ADR was written retroactively on 2026-04-16. The config file
> `hallucination_pipeline_config.yaml` has existed since approximately
> late 2025 and is the current source of truth for pipeline behaviour.

The cascade pipeline has many behavioural knobs: which layers are
enabled, which models L2/L3 use, cache settings, ensemble aggregation
rules, generic-verb lists for graph traversal, thresholds. Hard-coding
these in Python creates several failure modes:

- Changing a threshold requires a code change, a PR, and a deploy.
- Experiments cannot easily A/B different configurations.
- An operator cannot see the current configuration without reading code.
- Silent drift is possible when two callers hard-code different values.

## Decision Drivers

- Operator visibility: one file should answer "how is the pipeline
  configured right now?"
- Experiment reproducibility: config + code commit + data hash uniquely
  identifies a run.
- Single source of truth: no caller hard-codes what the config specifies.
- Safe defaults: if the config is missing a key, behaviour must not
  silently diverge.

## Considered Options

1. **Hardcoded in Python** — pipeline behaviour lives in module constants.
2. **YAML config file** — one file, version-controlled, loaded at startup.
3. **Database-backed config** — config in a database, hot-reloadable.
4. **Environment variables** — one env var per knob.

## Decision Outcome

**Chosen option: YAML config file.**

The canonical file is `hallucination_pipeline_config.yaml` at repo root.
Layer enables, model names, cache settings, and aggregation rules all
live there. The file is loaded once at pipeline startup. Callers read
config values through a single loader; no caller hard-codes a knob that
the config specifies.

YAML was chosen over JSON for its comment support, which matters for a
config file that documents its own rationale inline.

## Consequences

### Positive

- One file answers "what is the pipeline doing right now?"
- Experiments can branch the config without touching code.
- The config hash can be recorded in the run manifest for reproducibility.
- Comments inside the config explain why values are what they are.

### Negative

- A config schema must be maintained; YAML alone does not validate shape.
  A schema file (JSON Schema or Pydantic) is needed to catch typos.
- "Config drift" becomes possible — the file disagrees with code
  expectations. Must be guarded by a startup validator.
- Hot reload is not supported; pipeline must restart to pick up changes.

## More Information

- Config: `hallucination_pipeline_config.yaml`
- Related: ADR-0002 (four-layer cascade, which this configures),
  ADR-0007 (scope of layer enables for current phase)
