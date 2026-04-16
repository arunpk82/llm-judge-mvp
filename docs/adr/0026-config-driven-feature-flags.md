---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: A — Pipeline
---

# ADR-0026: Config-driven feature flags as the pipeline's behavioral contract

## Context and Problem Statement

The pipeline has many behavioural switches: which layers are enabled,
which model serves L3, whether spaCy or regex splits sentences, whether
graph cache is on, what TTL applies. As the cascade evolves, more
switches will be added — fact-counting vs MiniCheck for L3, L4 enable,
L5 enable, future experimental classifiers.

Without a disciplined pattern, switches accumulate as ad-hoc Python
constants, environment variables, function parameters, and YAML keys
in unrelated places. A reader cannot answer "what is the pipeline
doing right now?" by looking at one place. An operator cannot enable
or disable a behaviour without a code change. A future ADR cannot
reliably point at "the flag that controls this."

We need one canonical pattern for behavioural switches that the entire
platform follows.

## Decision Drivers

- **Single source of truth.** One file should answer the operator's
  question "what's enabled?"
- **Trust chain.** Every run's manifest should record which flags were
  active, by config hash.
- **Safe defaults.** A missing flag must not silently change behaviour.
- **Lifecycle.** Flags must be addable, deprecatable, and removable
  without breaking deployments.
- **No phantom limbs.** Flagged-off code paths must remain testable, or
  they will rot until the day someone needs them.

## Considered Options

1. **Ad-hoc switches.** Python constants, env vars, YAML keys mixed
   freely. The default state of most systems.
2. **Lightweight feature flags.** A YAML config file with declared
   flags, schema-validated at startup, manifest-recorded per run.
   No external service.
3. **External feature-flag service.** LaunchDarkly, Unleash, or similar
   — flags managed centrally, hot-reloadable, per-user targeting.
4. **Compile-time flags.** Flags as code; flipping requires a deploy.

## Decision Outcome

**Chosen option: Option 2 — lightweight feature flags via YAML config.**

The pattern, applied to the LLM Judge platform:

1. **One config file** — `hallucination_pipeline_config.yaml` (already
   exists). Future pipelines may add their own files; the pattern is
   shared.
2. **A schema** — Pydantic model or JSON Schema declaring every flag's
   type, allowed values, default, and dependencies. Lives next to the
   config, version-controlled.
3. **Startup validator** — runs once at process start. Loads the config,
   validates against the schema, asserts dependency constraints (e.g.,
   `l3_factcounting_enabled=true` requires `l2_enabled=true`). Fails
   fast on illegal combinations.
4. **Read-once semantics** — config is loaded once at startup and
   frozen. No hot reload. Changing flags requires restart.
5. **Manifest capture** — every run's `manifest.json` records the SHA-256
   of the config file used, plus the resolved flag values. The diff
   engine can detect config-attributable changes.
6. **Lifecycle states** — every flag has an explicit state in the
   schema: `active` (in use), `deprecated` (still readable, validator
   emits warning), `removed` (validator rejects with an error message
   pointing at the replacement).
7. **One ADR per behavioural flag** — adding a new flag is itself a
   decision. Don't bloat the config without recording why.

## Consequences

### Positive

- The operator can answer "what is the pipeline doing?" by reading
  one file.
- Every run is reproducible to the flag level, via the manifest's
  config hash.
- Rollback is a flag flip plus restart, not a code change.
- New features land behind a default-off flag, then graduate to
  default-on, then their flag is removed when stable. The lifecycle
  is documented and predictable.

### Negative

- Discipline is now load-bearing. If contributors ignore the pattern
  and add ad-hoc switches, the value evaporates.
- The schema must be maintained alongside the config. A flag added
  without a schema entry will pass validation but is not contracted.
- Flagged-off paths must still be tested in CI (ADR-0028) or they
  will rot.

## More Information

- Existing config: `hallucination_pipeline_config.yaml`
- Industry references: Netflix Fast Properties, Meta's GateKeeper,
  Google's Flagfile pattern, Spotify's Backstage feature-flag plugin
- Related: ADR-0006 (config-driven pipeline — this ADR formalises the
  pattern), ADR-0027 (first major application: L3 fact-counting),
  ADR-0028 (testing flagged-off paths)
