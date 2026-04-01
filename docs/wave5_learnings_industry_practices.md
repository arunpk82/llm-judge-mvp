# Wave 5 Learnings — Industry Best Practices for Deployment Architecture

**Purpose**: Capture Wave 5 (EPIC D1–D3) learnings as world-class best practices, grounded in how leading companies and industry standards handle deployment architecture.

**Documents to update**: AI-Architects-Playbook_v2_5.docx, LLM_Judge_Architecture_Walkthrough_v13.docx, AI-Architects-Playbook-Learning app

**Date**: 2026-04-01

---

## Industry Foundations Referenced

The practices below are grounded in four industry sources:

1. **The Twelve-Factor App** (Heroku/Adam Wiggins, 2011; open-sourced 2024) — the canonical methodology for building cloud-native applications, adopted by Netflix, Spotify, Stripe, and most modern SaaS platforms.

2. **Google SRE / Kubernetes Probe Model** — the liveness/readiness/startup probe separation that Kubernetes (originally from Google Borg) standardised for container health management.

3. **Spotify's Container Journey** (2014–present) — early Docker adopter, migrated from homegrown Helios orchestration to Kubernetes; learned hard lessons about container lifecycle, config management, and critical infrastructure rollouts.

4. **Immutable Infrastructure Pattern** (Netflix/Chad Fowler, 2013) — the principle that deployed artifacts are never modified in place; changes require building and deploying a new artifact.

---

## 1. PLAYBOOK: P07 — Design for Deployment (Rewritten with Industry Grounding)

**Insert after P06 in Part II.**

### Principle 07: Design for Deployment

"Where will this run — and what survives a restart?"

A system that works on a developer's laptop but fails in production is a prototype, not a product. Deployment architecture is a first-class design concern — not something handed to an ops team after code freeze.

This principle draws from three industry standards that every architect should know:

**The Twelve-Factor App (Factor III: Config, Factor V: Build-Release-Run).** The Twelve-Factor methodology, created at Heroku and adopted across the industry by companies like Netflix and Spotify, establishes two rules that directly shaped this principle. Factor III states that configuration must be strictly separated from code and stored in environment variables — the app should be open-sourceable at any moment without exposing credentials or environment-specific paths. Factor V states that the build, release, and run stages must be strictly separated — a build is immutable, a release combines build plus config, and the run stage executes the release. This means changing configuration never requires changing code.

**Immutable Infrastructure (Netflix).** Netflix popularised the principle that deployed artifacts — containers, VM images, serverless packages — are never modified in place. You don't SSH into a running container to change a config file. You build a new image, deploy it, and destroy the old one. This means the system must be designed so that any runtime state worth keeping lives outside the deployment artifact, on persistent storage that survives the artifact lifecycle.

**The Baked-In vs. Persistent Separation.** From these two industry practices, one architectural question emerges: for every file your system reads or writes, is it *baked-in* (part of the immutable deployment artifact, versioned with code) or *persistent* (produced at runtime, must survive artifact rebuilds)?

Getting this classification wrong has concrete consequences. Spotify's engineering team documented how Docker container rebuilds during their 2016-2017 migration could restart critical services affecting millions of users. Any runtime data stored inside containers would be lost during these restarts. The industry solution: persistent volumes that outlive container lifecycle.

**Three sub-principles follow:**

**a) Environment Independence (12-Factor, Factor III).** No hardcoded filesystem paths in source code. Every path resolves through a central module that reads environment variables with sensible fallbacks. The Twelve-Factor methodology is explicit: env vars are language-agnostic, OS-agnostic, and impossible to accidentally commit to source control (unlike config files). The litmus test: can you change where the system reads config and writes state without changing a single line of code?

**b) Pit-of-Success Configuration.** When designing env vars and volume mounts, prefer designs where the operator falls into correct behaviour by default. One env var for all persistent state (= one volume mount) is safer than three separate vars that can point to different locations. This is an *asymmetric cost decision* — the safer choice costs nothing extra. The Kubernetes ecosystem reinforces this: a single PersistentVolumeClaim per stateful concern is the standard pattern, not one PVC per subdirectory.

**c) Know Your Deployment Model.** The Twelve-Factor App assumes a specific deployment model (PaaS with ephemeral dynos). Kubernetes assumes another (orchestrated containers with restart policies). Bare-metal assumes yet another (long-lived processes with SIGHUP reload). P07 decisions require knowing which model you're targeting. Example: in the container model, restarting a container reimports all Python modules, so module-level constants that read env vars will pick up new values. In bare-metal, they won't. Design for your actual target, not a hypothetical one.

**d) The Three-Category Classification: Not All "Config" Is Config.** The Twelve-Factor App's Factor III ("store config in environment variables") is often misread as "put all configuration in env vars." This is a dangerous oversimplification. The word "config" conflates three fundamentally different categories of data, each with different lifecycle, trust, and deployment requirements:

| Category | What it is | Examples | Where it lives | How it changes |
|---|---|---|---|---|
| **Operational config** | Environment-specific settings that differ between deploys but don't change system behaviour | API keys, database URLs, port numbers, timeout values, path locations | Environment variables (`.env`, ConfigMaps, secrets managers) | Operator changes env var, restarts container. No code review needed. |
| **Behavioural config** | Files that define *what the system does* — changing them changes outcomes | Rule plans, scoring prompts, judge registry, evaluation rubrics, guardrail thresholds | Baked into the image, versioned in git | Developer commits change → code review → CI runs regression tests → new release deployed. Full audit trail. |
| **Runtime state** | Data produced by the system during operation, accumulates over time | Evaluation reports, baseline snapshots, audit logs, drift analysis, telemetry | Persistent volume (external storage that survives container rebuilds) | System writes it at runtime. Never manually edited. |

**Why this matters:** The temptation is to make behavioural config editable at runtime — "it's just a config file, why do I need a release?" Because changing a scoring prompt is not like changing a port number. A prompt change alters evaluation outcomes. Without git versioning, you lose the commit history. Without CI, you skip regression tests. Without a new release, you bypass the trust gates (P02) that the entire L1–L4 architecture was built to enforce. Making behavioural config editable on a running container is the architectural equivalent of letting someone modify the database schema without a migration script.

The Twelve-Factor App's Factor III applies specifically to *operational config*. Behavioural config follows Factor V (Build, Release, Run) — it is part of the build, immutable per release. Runtime state follows Factor VI (Processes) — stateless processes with external persistence.

**The litmus test for each file:** "If someone changes this file on a running system without code review, what breaks?" If the answer is "nothing important" (a port number, a timeout) → operational config, use env vars. If the answer is "scoring results change, trust evidence becomes unreproducible" → behavioural config, bake it into the image. If the answer is "it shouldn't be manually edited at all" → runtime state, put it on a persistent volume.

Three-Domain Examples:

- LLM-Judge: The three categories are explicit in the deployment layout. Operational config: env vars (`JUDGE_ENGINE`, `LLM_JUDGE_DATA_DIR`, API keys). Behavioural config: rule plans, judge registry, prompt templates — baked into the image at `/app/configs`, versioned in git, changes require a new release with CI gates. Runtime state: reports, baselines, audit logs — persistent volume at `/data`, produced by the system, never manually edited. The `/ready` endpoint validates the same paths that runtime code uses.
- GenAI: Operational config: model provider URL, temperature, max tokens (env vars). Behavioural config: prompt templates, guardrail thresholds, RAG retrieval parameters — baked into image, new prompt version = new release (Factor V). Runtime state: conversation logs, user feedback, vector store indexes — persistent volumes. The litmus test applies: changing a guardrail threshold changes safety outcomes, so it needs code review, not an env var.
- Classical ML: Operational config: serving port, batch size, feature store URL (env vars). Behavioural config: model artifacts, feature schemas, prediction thresholds — baked into serving image, versioned with model registry. Runtime state: prediction logs, A/B results, retraining datasets — persistent storage. Changing a prediction threshold is a model decision, not an operational decision.

---

## 2. PLAYBOOK: P07 Implementation Lesson — Functions vs. Constants

**Insert as a subsection under P07, or under Part VII (Architecture Decision Frameworks).**

### Implementation Pattern: Runtime Resolution vs. Import-Time Binding

When code reads from environment variables to resolve configuration, there is a subtle but important implementation choice:

```python
# Option A: Module-level constant (import-time binding)
CONFIG_ROOT = Path(os.environ.get("CONFIG_DIR", "configs"))

# Option B: Function (runtime resolution)
def config_root() -> Path:
    return Path(os.environ.get("CONFIG_DIR", "configs"))
```

Both work identically in production. The difference appears in testability. Constants evaluate when Python first imports the module and are cached forever. Functions evaluate on each call. This means:

- With constants, a test that sets an env var *after* import gets the old value. Tests require `importlib.reload()` hacks or must set env vars before any import.
- With functions, `monkeypatch.setenv` works naturally — the next call to `config_root()` sees the new value.

**Industry context:** The Twelve-Factor App says env vars should be "granular controls, each fully orthogonal to other env vars." Functions preserve this orthogonality in test code. The Kubernetes config pattern (ConfigMaps + environment injection) also assumes runtime resolution — config can change between pod restarts without rebuilding the image.

**Pragmatic compromise:** When downstream modules store the function result in a module-level constant (e.g. `CALIBRATION_DIR = state_root() / "calibration"`), this is acceptable in container deployments where restart = reimport. But recognise it as a deployment-model-specific compromise, not the general-purpose pattern.

---

## 3. PLAYBOOK: P07 Health Probe Architecture (Kubernetes/Google SRE Model)

**Insert as a subsection under P07, or as a new entry under "Deployment Topology Patterns" in Part III.**

### Health Probe Architecture: The Three-Probe Model

The Kubernetes probe model, originating from Google's Borg system, defines three distinct health checks that serve fundamentally different purposes. Conflating them is one of the most common deployment mistakes.

**Liveness Probe** (`/health`): "Is the process alive?" Returns 200 unconditionally if the application process is running. Never checks dependencies, storage, or configuration. A failed liveness probe triggers a container *restart* — so making it too strict causes restart loops. Google Cloud's best practice guidance is explicit: liveness probes should only detect unrecoverable states like deadlocks, not transient dependency failures.

**Readiness Probe** (`/ready`): "Can this instance serve traffic?" Validates that configuration exists, storage is writable, required datasets are accessible. A failed readiness probe removes the instance from the load balancer *without restarting it* — the instance stays alive and keeps checking until it becomes ready again. This is the correct place to validate that volume mounts are present and that the baked-in vs. persistent separation is intact.

**Dependency Probe** (`/health/dependencies`): "Are external services reachable?" Checks LLM provider APIs, databases, external services. This is a deep health check — useful for operators diagnosing issues, but too slow and too dependent on external factors for liveness or readiness decisions.

**The critical design rule:** The readiness probe must validate the *same paths* that runtime code uses. If your readiness check validates `/app/configs` but your code reads from `config_root()` which resolves to a different path, you have a silent failure mode — the probe passes but the first real request crashes. This is P05 (Design for Failure) applied to the health check itself.

Industry examples: Kubernetes documentation explicitly warns against putting dependency checks in liveness probes — if a database goes down temporarily, you don't want every pod restarting simultaneously. Spotify's infrastructure team documented how misconfigured Docker health checks during their 2016 migration caused cascading service restarts across thousands of containers.

---

## 4. PLAYBOOK: Asymmetric Cost Decisions (New Framework)

**Insert under Part VII: Architecture Decision Frameworks, after "Common AI Trade-Offs".**

### Asymmetric Cost Decisions

Not every architectural choice is a trade-off. Some decisions are *asymmetric*: one option costs nothing extra but eliminates a category of failure. These are not trade-offs — they are free wins. The architectural discipline is recognising which decisions are genuinely symmetric (require trade-off reasoning) and which are asymmetric (have a correct answer that costs nothing).

**Recognition pattern:** When comparing two designs, ask three questions:

1. Does the safer option cost more to implement? (If no → asymmetric)
2. Does the riskier option provide a concrete benefit? (If no → asymmetric)
3. Is the risk theoretical or has it caused real incidents? (Even theoretical risks count when the mitigation is free)

**Examples from industry and this platform:**

- **One data volume vs. multiple volumes** (LLM-Judge, Wave 5): Three persistent directories could each have their own env var. One parent env var eliminates the risk of fragmented data across volumes. Implementation cost is identical. The Kubernetes pattern reinforces this — one PersistentVolumeClaim per stateful concern, not per subdirectory.
- **Hash verification on read** (LLM-Judge, EPIC 1.2): Near-zero runtime cost, eliminates silent data corruption. This is why the Twelve-Factor App treats backing services as attached resources with explicit identity — you always verify you're talking to the right resource.
- **Functions vs. constants for env-var resolution** (LLM-Judge, Wave 5): Same runtime behaviour, but functions enable test isolation. No trade-off — just a free improvement in testability.
- **Structured error responses from health endpoints** (Kubernetes best practice): Returning `{"ok": false, "config_root": {"exists": false}}` costs the same as returning `{"status": "error"}`, but gives operators actionable information without requiring log access.

**Anti-pattern:** Treating asymmetric decisions as trade-offs. Saying "we chose three env vars for flexibility" when the flexibility has no concrete use case and the risk is real. Or saying "we'll add structured error responses later" when they cost nothing now. The habit of recognising asymmetric decisions accelerates architectural judgment because it removes entire categories of decisions from the "needs analysis" queue.

---

## 5. PLAYBOOK: Container Deployment Mental Model (New Section)

**Insert under Part III (AI Platform Reference Architecture), as a new subsection under "Deployment Topology Patterns".**

### Container Deployment Mental Model

Understanding container deployment is prerequisite knowledge for applying P07. The following concepts are drawn from the Twelve-Factor App (Factors V and VI) and the Kubernetes operational model.

**The container lifecycle.** A container is built from a Dockerfile (creating an image), then run. The image is immutable — it contains code, configs, and dependencies frozen at build time. When the container runs, it gets a writable filesystem layer, but this layer is destroyed when the container stops or is rebuilt. Netflix formalised this as "immutable infrastructure" — you never modify a running deployment, you replace it with a new one.

**Rebuild = fresh start.** Deploying a code fix means building a new image and starting a new container. Everything inside the old container disappears. Spotify's engineering team documented how this property, combined with their fleet of thousands of Docker instances, meant that any runtime data stored inside containers was at risk during upgrades. A volume mount connects external storage to a path inside the container — the data persists regardless of container lifecycle.

**Restart = reimport.** When a container restarts, the application process starts fresh. In Python, all modules are reimported, all module-level code re-executes. This means module-level constants that call `os.environ.get()` will pick up new env var values on restart. In this deployment model, the distinction between "function called at runtime" and "constant set at import time" matters only for testability, not for operational flexibility.

**Configuration flow (12-Factor, Factor III).** Environment variables are set in docker-compose.yaml, Kubernetes ConfigMaps, or .env files, injected into the container at start time, and read by the application. Changing an env var requires restarting the container. There is no "hot reload" of env vars in the standard container model. The Twelve-Factor App is explicit: env vars are "granular controls, each fully orthogonal to other env vars" — they are never grouped into named environments like "staging" or "production."

**The probe contract (Kubernetes).** The orchestrator (Docker, Kubernetes) needs to know three things about each container: is it alive (liveness), can it serve traffic (readiness), and has it finished starting (startup). The application exposes HTTP endpoints that answer these questions. Getting the probe boundaries wrong — for example, putting a database check in the liveness probe — causes cascading failures when the database has a brief outage and the orchestrator restarts every container simultaneously.

---

## 6. PLAYBOOK: Additions to Validation Checklist (Part X)

**Add to the validation questions list:**

- How does configuration reach the running system? (Env vars, config files, feature flags — and which requires a restart vs. hot reload?)
- What data survives a deployment? (Volumes, databases, external storage — and is the baked-in vs. persistent boundary explicitly documented?)
- Does the readiness probe validate the same paths that runtime code uses? (If not, you have a silent failure mode — the probe passes but the first real request crashes.)

---

## 7. WALKTHROUGH v13: Wave 5 Implementation Notes

**Insert as a new section in the Walkthrough.**

### Wave 5 Implementation Notes: Deployment Foundation (P07)

#### What We Built

EPIC-D1 (Environment-Independent Configuration) created `src/llm_judge/paths.py` — a central path resolution module following Twelve-Factor App Factor III. Two environment variables control all path resolution:

- `LLM_JUDGE_CONFIGS_DIR` (default: `configs/`) — baked-in configuration, immutable per release
- `LLM_JUDGE_DATA_DIR` (default: `.`) — parent of all persistent data directories (`reports/`, `baselines/`, `datasets/`)

The audit found 20 hardcoded paths across 13 files, classified into three categories matching the baked-in vs. persistent separation: config reads (4 references → `config_root()`), state writes (13 references → `state_root()`), and data read/writes (3 references → `datasets_root()`, `baselines_root()`).

EPIC-D2 (Health & Readiness Probes) implemented the Kubernetes three-probe model: `/health` (liveness — always 200), `/ready` (readiness — validates config exists, storage writable, datasets present), `/health/dependencies` (LLM provider reachability). The readiness probe calls the same `config_root()` / `state_root()` functions as runtime code, eliminating the "validates different path" silent failure mode.

EPIC-D3 (Docker Packaging) created `docker-compose.yaml` with a single named volume for all persistent data, `.env.example` documenting all env vars, and updated the Dockerfile with the baked-in vs. persistent separation.

#### Key Decisions Mapped to Industry Practices

| Decision | Industry Practice | Rationale |
|---|---|---|
| Env vars for path resolution | 12-Factor App, Factor III | Language-agnostic, impossible to accidentally commit |
| Single `data_root()` for all persistent data | Kubernetes PVC pattern | One volume = one mount = pit-of-success for operators |
| Functions over constants in `paths.py` | 12-Factor orthogonality | Testable without import-order hacks |
| Pragmatic module-level caching | Container restart model | Restart = reimport; acceptable in this deployment model |
| Three separate health endpoints | Kubernetes probe model | Liveness ≠ readiness ≠ dependency; conflating them causes cascading failures |
| `/ready` validates runtime paths | P05 applied to probes | The meta-failure: a broken failure detector |
| Configs baked into image, not on volume | 12-Factor Factor V + P02 Trust Architecture | Behavioural config (rule plans, prompts) changes outcomes — needs git versioning, CI gates, audit trail. Not all "config" is Factor III config. |

#### Pending Decisions

- `rules/manifest.yaml` path remains hardcoded — project-root config baked into the image, outside `configs/` scope
- If hot-reload of config is ever needed (bare-metal deployment), module-level caching must be refactored to pure function calls
- `PlatformStore` interface design (EPIC 9.1) will determine whether SQLite lives under `data_root()` or needs its own mount
