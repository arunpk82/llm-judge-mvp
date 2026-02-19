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
    PYTHONPATH=/app/src

WORKDIR /app
RUN useradd -m -u 10001 appuser

COPY --from=builder /app/requirements.txt /app/requirements.txt

# --- 1) Install app deps (baseline) ---
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir -r /app/requirements.txt

# --- 2) Force patched versions + ensure required deps exist ---
RUN python -m pip install --no-cache-dir --upgrade --force-reinstall \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0"

# --- 3) TRACE + CLEANUP stale dist-info across ALL python paths (Trivy focuses on METADATA) ---
# BuildKit-safe: heredoc is its own RUN (no trailing "\" after PY terminator)
RUN python - <<'PY'
from __future__ import annotations

import os
import sys
import site
import sysconfig
from pathlib import Path
import shutil
import importlib.metadata as md

def banner(msg: str) -> None:
    print("\n" + "=" * 88)
    print(msg)
    print("=" * 88)

def existing_dirs(paths):
    seen = set()
    for p in paths:
        if not p:
            continue
        try:
            pp = Path(p)
        except Exception:
            continue
        if pp.exists() and pp.is_dir():
            rp = str(pp.resolve())
            if rp not in seen:
                seen.add(rp)
                yield Path(rp)

def all_python_dirs():
    candidates = []
    # Common packaging dirs
    try:
        candidates += site.getsitepackages()
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if usp:
            candidates.append(usp)
    except Exception:
        pass

    # sysconfig locations
    sp = sysconfig.get_paths()
    for k in ("purelib", "platlib", "stdlib", "data"):
        v = sp.get(k)
        if v:
            candidates.append(v)

    # sys.path (includes dist-packages on Debian)
    candidates += [p for p in sys.path if isinstance(p, str)]

    # Filter to directories that are likely to hold installed dists
    # but keep stdlib too (sometimes dist-info can end up there in broken images)
    return list(existing_dirs(candidates))

def print_dist(name: str):
    try:
        dist = md.distribution(name)
    except md.PackageNotFoundError:
        print(f"[dist] {name}: NOT INSTALLED (per importlib.metadata)")
        return

    loc = None
    try:
        loc = str(dist.locate_file(""))
    except Exception:
        loc = "unknown"
    print(f"[dist] {name}: version={dist.version} location={loc}")
    # dist.files can be None depending on install type
    if dist.files:
        # print a few representative entries
        sample = list(dist.files)[:10]
        print(f"[dist] {name}: sample files:")
        for f in sample:
            print(f"  - {f}")

def find_metadata_dirs():
    # Known-problem stale metadata variants (dash/underscore normalization differs)
    targets = {
        "wheel-0.45.1.dist-info",
        "jaraco.context-5.3.0.dist-info",
        "jaraco_context-5.3.0.dist-info",
    }

    hits = []
    for base in all_python_dirs():
        try:
            for t in targets:
                p = base / t
                if p.exists():
                    hits.append(p)
        except Exception:
            continue
    return hits

def deep_find(patterns):
    hits = []
    for base in all_python_dirs():
        # Avoid huge scans outside python dirs; we only iterate immediate children
        try:
            for child in base.iterdir():
                n = child.name
                if any(n == pat for pat in patterns):
                    hits.append(child)
        except Exception:
            continue
    return hits

banner("TRACE (pre-clean) - Python runtime + dist locations")
print("python:", sys.version.replace("\n"," "))
print("executable:", sys.executable)
print("sys.path:")
for p in sys.path:
    print(" -", p)

banner("TRACE (pre-clean) - importlib.metadata view")
print_dist("wheel")
print_dist("jaraco.context")
print_dist("packaging")
print_dist("backports.tarfile")

banner("TRACE (pre-clean) - filesystem metadata dirs that commonly trip Trivy")
pre_hits = find_metadata_dirs()
if not pre_hits:
    print("No known stale dist-info dirs found (wheel 0.45.1 / jaraco_context 5.3.0).")
else:
    for h in pre_hits:
        print("FOUND:", h)

# Remove stale metadata dirs wherever they exist
banner("CLEANUP - removing stale metadata dirs (if present)")
removed = 0
for h in pre_hits:
    try:
        if h.is_dir():
            shutil.rmtree(h)
        else:
            h.unlink()
        print("REMOVED:", h)
        removed += 1
    except Exception as e:
        print("FAILED to remove:", h, "error:", repr(e))

banner("TRACE (post-clean) - confirm stale metadata is gone")
post_hits = find_metadata_dirs()
if not post_hits:
    print("OK: stale dist-info dirs are gone.")
else:
    for h in post_hits:
        print("STILL PRESENT:", h)

print(f"\nCleanup summary: removed={removed} remaining={len(post_hits)}")
PY

# --- 4) Integrity check (will fail the build if deps are inconsistent) ---
RUN python -m pip check

COPY src ./src
COPY rubrics ./rubrics

USER appuser
EXPOSE 8000
CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
