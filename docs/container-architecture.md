# Container Architecture

## Multi-stage Build

Stage 1: Builder
- Python 3.11-slim
- Poetry export
- Build requirements.txt

Stage 2: Runtime
- Minimal Python 3.11-slim
- Install runtime dependencies
- Security patching
- Metadata validation
- Non-root user execution

---

## Runtime User

Container runs as:
UID 10001
Non-root principle enforced.

---

## Ports

Exposes:
8000

Command:
uvicorn llm_judge.main:app --host 0.0.0.0 --port 8000
