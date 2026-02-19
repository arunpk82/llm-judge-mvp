# LLM Judge MVP

![CI](https://github.com/arunpk82/llm-judge-mvp/actions/workflows/ci.yml/badge.svg)


# LLM Judge MVP

![CI](https://github.com/arunpk82/llm-judge-mvp/actions/workflows/ci.yml/badge.svg)

An enterprise-ready **LLM-as-a-Judge evaluation platform** designed to deliver deterministic, auditable, and extensible scoring of AI-generated responses using structured, versioned rubrics.

This solution is architected with production discipline from day one, emphasizing:
* Deterministic-first evaluation (reproducibility & auditability)
* Rubric-driven governance (versioned YAML contracts)
* Pluggable evaluation engines (deterministic + LLM)
* Strongly typed API contracts (Pydantic schemas)
* CI-enforced quality controls (ruff, mypy, pytest)
* Extensibility without contract breakage

---

# Executive Summary
Enterprise AI evaluation often lacks:
* Reproducibility
* Governance over rubric evolution
* Transparent scoring logic
* Strict API contracts
* Clear abstraction boundaries

`llm-judge-mvp` establishes a structured evaluation framework that enables:
✔ Deterministic baseline scoring for auditability
✔ Version-controlled rubric registry
✔ Pluggable LLM evaluation engines
✔ Offline and API-driven execution
✔ Strict schema enforcement
✔ CI-governed code quality

This repository serves as a reference architecture for building scalable LLM evaluation services.

---

# System Architecture Overview
```
Client Request
        ↓
FastAPI (/predict)
        ↓
JudgeEngine.evaluate()
        ↓
Scoring Pipeline
        ↓
Rubric Registry (Versioned YAML)
        ↓
PredictResponse (scores + decision + confidence)
```

## Architectural Guarantees
* Deterministic scoring as a reproducible baseline
* Explicit rubric versioning and registry control
* Strict API schema contract enforcement
* Clear separation of concerns (engine vs scoring vs registry)
* Testable components with integration coverage

---

# Quickstart

## Prerequisites
* Python 3.11+
* Poetry

## Installation
```bash
poetry install
```

## Run Service
```bash
poetry run uvicorn llm_judge.main:app --reload
```

## Health Check
```bash
curl http://localhost:8000/health
```

## Sample Prediction Call
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

# API Surface

## `GET /health`
Liveness probe suitable for orchestration platforms.

## `POST /predict`
Evaluates a candidate response against a registered rubric.

### Response Contract
* Dimension-level scores
  * Relevance
  * Clarity
  * Correctness
  * Tone
* Aggregate score
* Decision flag
* Confidence score

The canonical schema is defined in:
```
src/llm_judge/schemas.py
```
Schema stability is considered a core contract.

---

# Repository Structure
```
.
├── src/
│   └── llm_judge/
│       ├── main.py                 # FastAPI entrypoint
│       ├── runtime.py              # Engine selection & dependency wiring
│       ├── judge_base.py           # JudgeEngine protocol
│       │
│       ├── deterministic_judge.py  # Deterministic evaluation engine
│       ├── llm_judge.py            # LLM-based evaluation engine (extensible)
│       │
│       ├── scorer.py               # Core scoring pipeline
│       ├── correctness.py          # Deterministic correctness evaluator
│       ├── llm_correctness.py      # LLM correctness adapter
│       │
│       ├── rubric_store.py         # Versioned YAML rubric loader
│       ├── rubrics.py              # Rubric data models
│       ├── schemas.py              # API request/response contracts
│       │
│       └── eval/harness.py         # Offline evaluation harness
│
├── tests/                          # Unit & integration tests
├── rubrics/                        # Version-controlled rubric definitions
├── .github/workflows/              # CI pipeline
├── pyproject.toml
├── README.md
└── (future) docs/
```

---

# Rubric Governance Model
Rubrics are YAML-defined and version-controlled.

## Add a New Rubric Version
```
rubrics/<rubric_name>/v1.yaml
```

## Update Active Registry Mapping
```yaml
latest:
  chat_quality: v1
```

### Governance Rules
* Published rubric versions are immutable.
* Registry defines the active version.
* Scoring logic derives strictly from rubric structure.
* Backward compatibility must be maintained unless explicitly versioned.

---

# Evaluation Engines

## Deterministic Engine (Default)
* Rule-based scoring
* Fully reproducible
* Baseline audit reference
* CI-verifiable

## LLM Engine (Extensible)
* Designed for integration with external LLM providers
* Runtime pluggable
* Contract-bound output validation

### Adding a New Engine
1. Implement `JudgeEngine` protocol
2. Register engine in `runtime.py`
3. Add unit + integration coverage
4. Validate output against `schemas.py`

Engine extensibility is intentional and isolated from scoring logic.

---

# Quality & Governance Controls
Run full local validation:
```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

CI enforces:

* Static typing compliance
* Linting standards
* Test coverage validation
* Contract integrity

Quality gates are non-optional for merge approval.

---

# Engineering Principles
* Determinism as foundation
* Explicit versioning over implicit mutation
* Clear abstraction boundaries
* Schema-first API contracts
* Extensibility without destabilization
* Testability before feature expansion

---

# Strategic Roadmap
Planned enterprise enhancements:
* Production LLM provider integration
* Async evaluation batching
* Containerization (Docker)
* Continuous deployment pipeline
* Observability (metrics, structured logging, tracing)
* Evaluation calibration framework

---

# Contribution Guidelines
Enhancements must:
* Preserve deterministic baseline behavior
* Maintain schema stability
* Include tests
* Pass all CI quality gates

---

# License
Private repository.
