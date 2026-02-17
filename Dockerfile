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

# Cache-buster for CI so audit/cleanup actually reruns
ARG AUDIT_SEED=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN useradd -m -u 10001 appuser

COPY --from=builder /app/requirements.txt /app/requirements.txt

# 1) Install app deps
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /app/requirements.txt

# 2) Force patched versions (what Trivy wants)
RUN python -m pip install --no-cache-dir --upgrade --force-reinstall \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0"

# 3) TRACE (pip-level) — where did these land?
RUN echo "AUDIT_SEED=${AUDIT_SEED}" \
 && python -m pip show -f wheel jaraco.context || true

# 4) TRACE + CLEAN (filesystem-level) — scan METADATA like Trivy does
# IMPORTANT: keep heredoc in its OWN RUN and do NOT use "\" line continuations here.
RUN python - <<'PY'
import sys, site
from pathlib import Path
import importlib.metadata as md

TARGET = {
    "wheel": "0.46.2",
    "jaraco.context": "6.1.0",
}

def candidate_roots():
    roots = []
    roots.extend(site.getsitepackages())
    usp = site.getusersitepackages()
    if usp:
        roots.append(usp)
    for p in sys.path:
        if p and ("site-packages" in p or "dist-packages" in p):
            roots.append(p)

    uniq = []
    seen = set()
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        rp = Path(r)
        if rp.exists() and rp.is_dir():
            uniq.append(rp)
    return uniq

def read_metadata(dist_info: Path):
    meta = dist_info / "METADATA"
    if not meta.exists():
        return None, None
    name = ver = None
    for line in meta.read_text(errors="ignore").splitlines():
        if line.startswith("Name: "):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version: "):
            ver = line.split(":", 1)[1].strip()
        if name and ver:
            break
    return name, ver

print("TRACE: python =", sys.version.replace("\n", " "))
print("TRACE: executable =", sys.executable)

print("TRACE: importlib.metadata versions:")
for pkg in ["wheel", "jaraco.context"]:
    try:
        print(f"  - {pkg} = {md.version(pkg)}")
    except Exception as e:
        print(f"  - {pkg} = <not found> ({e})")

roots = candidate_roots()
print("TRACE: discovered package roots:")
for r in roots:
    print("  -", r)

hits = []
for r in roots:
    for dist in r.glob("*.dist-info"):
        name, ver = read_metadata(dist)
        if name in TARGET:
            hits.append((name, ver, dist))

print("TRACE: dist-info hits (before cleanup):")
for name, ver, dist in sorted(hits, key=lambda x: (x[0], x[1] or "", str(x[2]))):
    print(f"  - {name} {ver} :: {dist}")

removed = []
for name, ver, dist in hits:
    want = TARGET[name]
    # Remove any stale METADATA (Trivy reads METADATA)
    if ver and ver != want:
        removed.append((name, ver, str(dist)))
        for child in sorted(dist.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass
        try:
            dist.rmdir()
        except OSError:
            pass

print("TRACE: removed stale dist-info:")
for name, ver, path in removed:
    print(f"  - REMOVED {name} {ver} :: {path}")

# Re-scan after cleanup
hits2 = []
for r in roots:
    for dist in r.glob("*.dist-info"):
        name, ver = read_metadata(dist)
        if name in TARGET:
            hits2.append((name, ver, str(dist)))

print("TRACE: dist-info hits (after cleanup):")
for name, ver, path in sorted(hits2):
    print(f"  - {name} {ver} :: {path}")

bad = [(n, v, p) for (n, v, p) in hits2 if v != TARGET[n]]
if bad:
    raise SystemExit("FAIL: stale dist-info still present: " + str(bad))

print("TRACE: OK - no stale dist-info remains for wheel/jaraco.context")
PY

# 5) Sanity
RUN python -m pip check

COPY src ./src
COPY rubrics ./rubrics

USER appuser
EXPOSE 8000
CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
