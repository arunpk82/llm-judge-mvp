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

# Poetry + export plugin (Poetry 2.x needs the plugin for `poetry export`)
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}" \
  && poetry self add poetry-plugin-export

# Copy dependency files first (for build cache)
COPY pyproject.toml poetry.lock* ./

# Export locked deps to requirements.txt (runtime install will be pip-based)
RUN poetry export \
    -f requirements.txt \
    --output requirements.txt \
    --without-hashes \
    --without dev

############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN useradd -m -u 10001 appuser

# Bring in locked runtime requirements
COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install app deps, enforce patched toolchain libs (Trivy gate),
# remove stale dist-info metadata that Trivy may detect,
# then verify dependency integrity.
RUN python -m pip install --no-cache-dir --upgrade pip \
  && python -m pip install --no-cache-dir -r /app/requirements.txt \
  && python -m pip install --no-cache-dir --upgrade \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0"

# Remove stale dist-info metadata
RUN python - <<'PY'
import site
import pathlib
import re
import shutil

paths = site.getsitepackages()
stale = [
    re.compile(r"^wheel-0\.45\.1\.dist-info$"),
    re.compile(r"^jaraco\.context-5\.3\.0\.dist-info$"),
]

for p in paths:
    sp = pathlib.Path(p)
    if not sp.exists():
        continue

    for child in sp.iterdir():
        if any(pat.match(child.name) for pat in stale):
            print(f"Removing stale metadata: {child}")
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
PY

RUN python -m pip check

# Copy application code + rubrics
COPY src ./src
COPY rubrics ./rubrics

USER appuser

EXPOSE 8000

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
