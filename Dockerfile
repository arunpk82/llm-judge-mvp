# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.1.1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" && poetry self add poetry-plugin-export

# Copy dependency files first
COPY pyproject.toml poetry.lock* ./

# Export locked deps to requirements.txt (runtime install will be pip-based)
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes --without dev

# Copy source needed to build wheel
COPY src ./src
COPY rubrics ./rubrics

############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN useradd -m -u 10001 appuser

# Install runtime deps only (no poetry in runtime)
COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application code + rubrics
COPY --from=builder /app/src /app/src
COPY --from=builder /app/rubrics /app/rubrics

USER appuser

EXPOSE 8000

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
