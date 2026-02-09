# ADR-0001: Project Scope and Engineering Principles

## Status
Accepted

## Context
We are building an LLM-as-a-Judge MVP with strong emphasis on:
- determinism
- schema-first design
- auditability
- production realism

## Decisions
- Python 3.11 with Poetry
- src/ layout
- FastAPI for runtime
- JSON Schema + Pydantic for contracts
- LLD-first development approach

## Consequences
- Slightly higher upfront rigor
- Significantly lower long-term risk
- Easier onboarding and governance
