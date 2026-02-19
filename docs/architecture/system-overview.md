# System Overview

## 1. Purpose
The LLM Judge MVP system provides a structured, reproducible, and extensible framework for evaluating AI-generated responses against versioned rubrics.

The architecture is intentionally designed to:
* Ensure deterministic baseline evaluation
* Support pluggable LLM-based scoring engines
* Enforce strict API schema contracts
* Enable rubric governance via version control
* Maintain separation of concerns across evaluation layers

This document defines the high-level architecture, component responsibilities, data flow, and extension boundaries.

---

## 2. High-Level Architecture
```
Client
  ↓
FastAPI Service Layer
  ↓
Runtime Engine Selector
  ↓
JudgeEngine (Deterministic | LLM)
  ↓
Scoring Pipeline
  ↓
Rubric Registry (YAML + Versioning)
  ↓
Structured PredictResponse
```

The architecture follows a layered evaluation model with clear contracts between each layer.

---

## 3. Component Responsibilities

### 3.1 FastAPI Service Layer
Location:
```
src/llm_judge/main.py
```
Responsibilities:
* Expose `/predict` and `/health` endpoints
* Validate request schema
* Return strictly typed response schema
* Remain stateless

This layer must not contain business logic.

---

### 3.2 Runtime Engine Selector
Location:
```
src/llm_judge/runtime.py
```
Responsibilities:
* Instantiate and wire the selected JudgeEngine
* Abstract engine selection logic from API layer
* Enable future engine injection (e.g., feature flags, env config)

This ensures evaluation engines are pluggable without modifying API code.

---

### 3.3 JudgeEngine Abstraction
Location:

```
src/llm_judge/judge_base.py
```
Defines the protocol contract for all evaluation engines.
Contract Requirements:
* Accept normalized input
* Produce structured evaluation output
* Conform to `schemas.py`

All engines must implement this interface.

---

### 3.4 Deterministic Judge Engine
Location:
```
src/llm_judge/deterministic_judge.py
```
Responsibilities:
* Provide reproducible scoring
* Serve as audit baseline
* Enable deterministic CI validation

Design Goal:
The deterministic engine is the canonical evaluation reference.

---

### 3.5 LLM Judge Engine
Location:
```
src/llm_judge/llm_judge.py
```
Responsibilities:
* Integrate with external LLM providers
* Translate rubric requirements into prompts
* Validate LLM outputs against schema
* Remain isolated from scoring pipeline logic

LLM variability must be controlled via strict output validation.

---

### 3.6 Scoring Pipeline
Location:
```
src/llm_judge/scorer.py
```

Responsibilities:
* Execute scoring logic per rubric dimension
* Aggregate dimension-level scores
* Compute final decision + confidence

This layer must remain independent of engine implementation.

---

### 3.7 Rubric Registry
Location:
```
src/llm_judge/rubric_store.py
rubrics/
```

Responsibilities:
* Load versioned YAML rubrics
* Resolve active version via registry mapping
* Enforce immutability of published versions

Governance Rule:
Rubrics are versioned artifacts and must not be modified in-place.

---

### 3.8 API Schemas
Location:
```
src/llm_judge/schemas.py
```

Responsibilities:
* Define canonical request/response models
* Serve as contract boundary
* Protect against malformed output

Schema stability is treated as a production contract.

---

## 4. Data Flow
1. Client submits request to `/predict`
2. FastAPI validates request against schema
3. Runtime selects JudgeEngine
4. Engine invokes scoring pipeline
5. Rubric registry resolves active rubric version
6. Scores computed per dimension
7. Aggregate score + decision calculated
8. PredictResponse returned

Each step maintains strict contract boundaries.

---

## 5. Determinism & Reproducibility Model
The system is designed with deterministic evaluation as the baseline.
Key Guarantees:
* Deterministic engine produces identical outputs for identical inputs
* Rubric versions are immutable
* CI validates scoring logic
* Schema enforcement prevents ambiguous outputs
This model enables auditability and regression safety.

---

## 6. Extension Model
The system supports extension along three axes:

### 6.1 Add a New Rubric Version
* Create new versioned YAML
* Update registry mapping
* Add validation tests

### 6.2 Add a New Evaluation Engine
* Implement JudgeEngine protocol
* Register via runtime
* Add test coverage
* Ensure schema conformity

### 6.3 Add New Scoring Dimensions
* Update rubric structure
* Extend scoring logic
* Update response schema
* Maintain backward compatibility via versioning

All extensions must preserve API contracts.

---

## 7. Non-Goals (Current Scope)
* Persistent storage layer
* Distributed execution
* Horizontal scaling mechanisms
* Multi-tenant isolation
* Production-grade observability

These are planned for future enterprise evolution.

---

## 8. Architectural Principles
* Determinism first
* Schema-first API governance
* Explicit versioning over mutation
* Clear abstraction boundaries
* Extensibility without destabilization
* CI-enforced integrity

---

## 9. Future Evolution Path
Planned architectural enhancements:
* LLM provider abstraction layer
* Async evaluation batching
* Containerization and orchestration support
* Structured logging and metrics
* Calibration and evaluation benchmarking layer

This document will evolve alongside the system and serve as the architectural source of truth.
