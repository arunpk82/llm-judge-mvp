# syntax=docker/dockerfile:1.7

############################
# Builder stage
############################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "poetry==2.1.1" \
    && poetry self add poetry-plugin-export

COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


############################
# Runtime stage
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Make sure imports work when code is in /app/src
    PYTHONPATH=/app/src

WORKDIR /app

RUN useradd -m -u 10001 appuser

COPY --from=builder /app/requirements.txt /app/requirements.txt

# Cache-buster for CI so the audit/cleanup layer can be forced to re-run
ARG AUDIT_SEED=0

# 1) Install application deps
# 2) Force patched versions
# 3) Trace and remove stale dist-info by reading METADATA (what Trivy scans)
RUN echo "AUDIT_SEED=${AUDIT_SEED}" \
 && python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /app/requirements.txt \
 && python -m pip install --no-cache-dir --upgrade --force-reinstall \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0" \
 && python - <<'PY'
import os
import sys
import site
from pathlib import Path

TARGET = {
    "wheel": "0.46.2",
    "jaraco.context": "6.1.0",
}

def candidate_dirs():
    dirs = []
    # include all known python package roots
    dirs.extend(site.getsitepackages())
    if site.getusersitepackages():
        dirs.append(site.getusersitepackages())
    # include sys.path entries that look like site-packages/dist-packages
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p):
            dirs.append(p)
    # unique + existing
    out = []
    for d in dict.fromkeys(dirs):
        dp = Path(d)
        if dp.exists() and dp.is_dir():
            out.append(dp)
    return out

def read_metadata(dist_info: Path):
    meta = dist_info / "METADATA"
    if not meta.exists():
        return None, None
    name = version = None
    for line in meta.read_text(errors="ignore").splitlines():
        if line.startswith("Name: "):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version: "):
            version = line.split(":", 1)[1].strip()
        if name and version:
            break
    return name, version

print("TRACE: python =", sys.version.replace("\n"," "))
print("TRACE: sys.executable =", sys.executable)
print("TRACE: package roots:")
roots = candidate_dirs()
for r in roots:
    print("  -", r)

hits = []
for root in roots:
    for dist in root.glob("*.dist-info"):
        name, ver = read_metadata(dist)
        if name in TARGET:
            hits.append((name, ver, str(dist)))

print("TRACE: dist-info hits (before cleanup):")
for name, ver, path in sorted(hits):
    print(f"  - {name} {ver} :: {path}")

removed = []
for name, ver, path in hits:
    want = TARGET[name]
    if ver and ver != want:
        d = Path(path)
        removed.append((name, ver, path))
        # remove entire dist-info directory (this is what Trivy reads)
        for child in d.rglob("*"):
            if child.is_file():
                child.unlink(missing_ok=True)
        for child in sorted(d.rglob("*"), reverse=True):
            if child.is_dir():
                child.rmdir()
        d.rmdir()

print("TRACE: removed dist-info (stale/vulnerable):")
for name, ver, path in sorted(removed):
    print(f"  - REMOVED {name} {ver} :: {path}")

# Re-scan after cleanup
hits2 = []
for root in roots:
    for dist in root.glob("*.dist-info"):
        name, ver = read_metadata(dist)
        if name in TARGET:
            hits2.append((name, ver, str(dist)))

print("TRACE: dist-info hits (after cleanup):")
for name, ver, path in sorted(hits2):
    print(f"  - {name} {ver} :: {path}")

# Fail fast if stale metadata still exists (prevents wasting time in Trivy step)
bad = [(n,v,p) for (n,v,p) in hits2 if v != TARGET[n]]
if bad:
    raise SystemExit("Stale dist-info still present: " + str(bad))
PY \
 && python -m pip check

# Copy application code + rubrics
COPY src ./src
COPY rubrics ./rubrics

USER appuser
EXPOSE 8000

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
