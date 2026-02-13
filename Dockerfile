# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.0.1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# Minimal build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copy only dependency files first (better caching)
COPY pyproject.toml poetry.lock* ./

# Install runtime deps only
RUN poetry install --no-interaction --no-ansi --without dev --no-root

############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Non-root user
RUN useradd -m -u 10001 appuser

# Copy installed deps from builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy app and rubrics
COPY src ./src
COPY rubrics ./rubrics

USER appuser

EXPOSE 8000

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
