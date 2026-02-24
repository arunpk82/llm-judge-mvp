
# Milestone A Integration Guide

## New Files Added
- src/llm_judge/rubric_yaml.py
- tests/unit/test_rubric_yaml_schema.py
- docs/governance_contracts.md

## Steps to Integrate

1. Copy the `src/`, `tests/`, and `docs/` folders into your repo root.
2. Ensure you have `pydantic` installed:
   poetry add pydantic
3. Run:
   poetry run pytest
   poetry run ruff check .
   poetry run mypy .

4. Commit:
   git add -A
   git commit -m "feat(governance): rubric schema validation (Milestone A)"
