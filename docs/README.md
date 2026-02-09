# LLM Judge MVP

A lightweight “LLM-as-a-Judge” service that scores a candidate answer against a rubric across multiple dimensions (relevance, clarity, correctness, tone), producing a structured `PredictResponse`.

## Quickstart
```bash
poetry install
poetry run ruff check .
poetry run mypy src
poetry run pytest
poetry run uvicorn llm_judge.main:app --reload
