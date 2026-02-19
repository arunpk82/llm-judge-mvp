# Architecture

## Overview
`llm-judge-mvp` is an evaluation service that scores a candidate answer against a versioned rubric.

At a high level:

1. Client sends a `PredictRequest` (`conversation`, `candidate_answer`, `rubric_id`).
2. Runtime selects a judge engine (deterministic vs LLM-backed).
3. Engine returns a `PredictResponse` (decision, scores, confidence, flags, explanations).
4. The evaluation harness runs benchmarks over a dataset and writes reports to `reports/`.

## Key components

### API
- `src/llm_judge/main.py` exposes `/healthz` and `/predict`.
- Request/response contracts are defined in `src/llm_judge/schemas.py`.

### Judge engines
The platform supports multiple judge engines:

- **DeterministicJudge** (`src/llm_judge/deterministic_judge.py`)
  - CI-grade: stable, fast, explainable.
  - Rubric-driven policy and dimensions.
  - Designed as the default gate for PRs.

- **LLMJudge** (`src/llm_judge/llm_judge.py`)
  - Higher fidelity semantic evaluation.
  - Intended for nightly and release certification runs.

The selected engine is controlled via runtime configuration (`src/llm_judge/runtime.py`).

### Rubrics
Rubrics are versioned YAML files under `rubrics/`.

- `rubrics/registry.yaml` maps `rubric_id → latest version`.
- `src/llm_judge/rubric_store.py` resolves and loads rubrics.

Rubrics define:
- score scale
- dimensions
- decision policy
- optional dimension weights

### Deterministic scoring
Deterministic scoring lives in `src/llm_judge/scorer.py`.

It implements:
- per-dimension deterministic heuristics
- rubric-driven scoring policy
- consistent flags for governance and analytics

See `docs/DETERMINISTIC_JUDGE.md` for extension guidance.

### Benchmarking
The evaluation harness (`src/llm_judge/eval/harness.py`) runs the judge over a dataset
(e.g., `datasets/golden/v1.jsonl`) and writes aggregated results to `reports/`.

## Design principles
- **Deterministic first** for CI stability.
- **Rubric versioning** for auditability.
- **Artifact-driven outputs** (JSON reports) for reproducibility.
- **Modularity** to add new metrics and judge engines in Phase 3+.
