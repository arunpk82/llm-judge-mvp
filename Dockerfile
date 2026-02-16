# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

ARG POETRY_VERSION=2.1.1
RUN python -m pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
    && poetry self add poetry-plugin-export

COPY pyproject.toml poetry.lock* ./

# Export runtime requirements only (exclude dev)
RUN poetry export \
      -f requirements.txt \
      --output requirements.txt \
      --without-hashes \
      --without dev


############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

RUN useradd -m -u 10001 appuser

COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install runtime deps + force patched versions for Trivy findings
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/requirements.txt \
    && python -m pip install --no-cache-dir --upgrade \
         "packaging>=24.0" \
         "backports.tarfile>=1.2.0" \
         "wheel==0.46.2" \
         "jaraco.context==6.1.0"

# Remove stale dist-info folders that may remain and get detected by scanners
RUN python - <<'PY'
import site, pathlib, shutil

targets = {
    "wheel-0.45.1.dist-info",
    "jaraco.context-5.3.0.dist-info",
}

for sp in map(pathlib.Path, site.getsitepackages()):
    if not sp.exists():
        continue
    for d in sp.glob("*.dist-info"):
        if d.name in targets:
            print(f"Removing stale metadata: {d}")
            shutil.rmtree(d, ignore_errors=True)
PY

RUN python -m pip check

# Copy app code
COPY src ./src
COPY rubrics ./rubrics

USER appuser

EXPOSE 8000
CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
