# LLM Judge MVP

A lightweight **LLM-as-a-Judge** evaluation service that scores candidate answers against structured rubrics.

This MVP is designed to provide:

- Deterministic + LLM-based evaluation
- Rubric-driven scoring
- Consistent API contracts
- Testable and extensible architecture

---

## 🚀 Key Features

- FastAPI service with `/predict` endpoint
- Rubric-based scoring (YAML-driven)
- Pluggable judge engines:
  - Deterministic Judge (default)
  - LLM Judge (future-ready)
- Scoring dimensions:
  - Relevance
  - Clarity
  - Correctness
  - Tone
- Confidence + decision flags
- Full unit + integration test coverage

---

## 📌 High-Level Flow
Client Request
↓
FastAPI /predict
↓
JudgeEngine.evaluate()
↓
score_candidate()
↓
Rubric + Scoring Logic
↓
PredictResponse (decision + scores)


## 📂 Repository Structure

src/llm_judge/
│
├── main.py # FastAPI entrypoint
├── runtime.py # Engine selection + wiring
├── judge_base.py # JudgeEngine protocol contract
│
├── deterministic_judge.py # Rule-based judge engine
├── llm_judge.py # LLM-based judge engine (future)
│
├── scorer.py # Core scoring pipeline
├── correctness.py # Proxy correctness evaluator
├── llm_correctness.py # External LLM correctness support
│
├── rubric_store.py # Loads rubric YAML files
├── rubrics.py # Rubric data structures
├── schemas.py # API request/response models
│
└── eval/harness.py # Offline evaluation harness


---

## 🧪 Running Tests
Run full quality gate:
bash
poetry run ruff check .
poetry run mypy src
poetry run pytest


## 🧪 Running Locally
poetry run uvicorn llm_judge.main:app --reload

## 🧪 Health Check
curl http://localhost:8000/health

## 🧪 Prediction Call
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json


🧩 Adding a New Rubric
rubrics/<rubric_name>/v1.yaml


🧩 Update Registry
latest:
  chat_quality: v1
  
📈 Next Roadmap
Replace proxy correctness with real LLM provider
Add async batching for evaluation
Add CI/CD automation (GitHub Actions)
Deploy as containerized service




