# LLM-Judge API Reference

## FastAPI Service

The LLM-Judge service exposes a REST API via FastAPI.

**Entry point:** `src/llm_judge/main.py`

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check — returns `{"status": "ok"}` |
| POST | `/predict` | Score a candidate answer against a rubric |

### POST /predict

**Request body** (`PredictRequest`):
```json
{
  "conversation": [
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "candidate_answer": "Paris is the capital of France.",
  "rubric_id": "chat_quality"
}
```

**Response** (`PredictResponse`):
```json
{
  "decision": "pass",
  "overall_score": 4.25,
  "scores": {"relevance": 5, "clarity": 4, "correctness": 4, "tone": 4},
  "confidence": 0.7,
  "flags": [],
  "explanations": {
    "relevance": "Answer is relevant to the user's question",
    "clarity": "Answer is clear and well structured",
    "correctness": "No obvious uncertainty markers; proxy assumes likely correct.",
    "tone": "Tone is polite and appropriate"
  }
}
```

**Schemas:** See `src/llm_judge/schemas.py` for Pydantic model definitions.

### Running the service

```bash
poetry run uvicorn llm_judge.main:app --host 0.0.0.0 --port 8000
```

### Engine selection

Set `JUDGE_ENGINE` environment variable:
- `deterministic` (default) — rule-based scoring
- `llm` — LLM-based scoring with deterministic fallback
