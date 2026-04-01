# syntax=docker/dockerfile:1.7

############################
# Builder: export requirements
############################
FROM python:3.11-slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "poetry==2.1.1" \
  && poetry self add poetry-plugin-export

COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes


############################
# Runtime: install + audit + run
############################
FROM python:3.11-slim AS runtime
WORKDIR /app

# cache-buster for CI debugging
ARG AUDIT_SEED=0
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN useradd -m -u 10001 appuser

COPY --from=builder /app/requirements.txt /app/requirements.txt

# Install dependencies + force patched versions + *prove* filesystem state for Trivy
RUN set -eux; \
    echo "AUDIT_SEED=${AUDIT_SEED}"; \
    python -m pip install --no-cache-dir --upgrade pip; \
    python -m pip install --no-cache-dir -r /app/requirements.txt; \
    python -m pip install --no-cache-dir --upgrade --force-reinstall \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0"; \
    \
    # Remove common pip caches/temp artifacts that can carry stale wheels/sdists
    rm -rf /root/.cache/pip /tmp/pip-* /tmp/build /var/tmp/* || true; \
    \
    # === TRIVY-TRACE: scan the filesystem for what Trivy scans (dist-info/METADATA) ===
    python - <<'PY'
import sys, zipfile
from pathlib import Path

TARGET = {
    "wheel": "0.46.2",
    "jaraco.context": "6.1.0",
}

def parse_name_ver(txt: str):
    name = ver = None
    for line in txt.splitlines():
        if line.startswith("Name: "):
            name = line.split(":", 1)[1].strip()
        elif line.startswith("Version: "):
            ver = line.split(":", 1)[1].strip()
        if name and ver:
            break
    return name, ver

def safe_read(p: Path) -> str:
    try:
        return p.read_text(errors="ignore")
    except Exception:
        return ""

def remove_tree(p: Path):
    if p.is_file():
        p.unlink(missing_ok=True)
        return
    for child in sorted(p.rglob("*"), reverse=True):
        try:
            if child.is_file():
                child.unlink(missing_ok=True)
            elif child.is_dir():
                child.rmdir()
        except Exception:
            pass
    try:
        p.rmdir()
    except Exception:
        pass

print("TRIVY-TRACE: python =", sys.version.replace("\n"," "))
print("TRIVY-TRACE: executable =", sys.executable)

# 1) dist-info scan (filesystem-wide)
hits = []
for dist in Path("/").rglob("*.dist-info"):
    meta = dist / "METADATA"
    if not meta.exists():
        continue
    name, ver = parse_name_ver(safe_read(meta))
    if name in TARGET:
        hits.append((name, ver, str(dist)))

print("TRIVY-TRACE: dist-info hits (pre-clean):")
for name, ver, path in sorted(hits):
    print(f"  - {name} {ver} :: {path}")

# Remove stale dist-info dirs for our targets
removed = []
for name, ver, path in hits:
    want = TARGET[name]
    if ver and ver != want:
        removed.append((name, ver, path))
        remove_tree(Path(path))

print("TRIVY-TRACE: removed stale dist-info:")
for name, ver, path in sorted(removed):
    print(f"  - REMOVED {name} {ver} :: {path}")

# 2) wheel archive scan: remove any .whl embedding stale METADATA for targets
bad_whls = []
for whl in Path("/").rglob("*.whl"):
    try:
        with zipfile.ZipFile(whl) as z:
            metas = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
            for m in metas:
                txt = z.read(m).decode("utf-8", "ignore")
                name, ver = parse_name_ver(txt)
                if name in TARGET and ver and ver != TARGET[name]:
                    bad_whls.append((name, ver, str(whl)))
                    break
    except Exception:
        continue

print("TRIVY-TRACE: bad whl artifacts found:", len(bad_whls))
for name, ver, path in bad_whls[:200]:
    print(f"  - BAD whl {name} {ver} :: {path}")

for _, _, p in bad_whls:
    try:
        Path(p).unlink(missing_ok=True)
    except Exception:
        pass

# 3) dist-info scan again (proof)
hits2 = []
for dist in Path("/").rglob("*.dist-info"):
    meta = dist / "METADATA"
    if not meta.exists():
        continue
    name, ver = parse_name_ver(safe_read(meta))
    if name in TARGET:
        hits2.append((name, ver, str(dist)))

print("TRIVY-TRACE: dist-info hits (post-clean):")
for name, ver, path in sorted(hits2):
    print(f"  - {name} {ver} :: {path}")

bad2 = [(n,v,p) for (n,v,p) in hits2 if v and v != TARGET[n]]
if bad2:
    raise SystemExit("TRIVY-TRACE: stale dist-info still present: " + str(bad2[:20]))

print("TRIVY-TRACE: OK - no stale METADATA remains for wheel/jaraco.context")
PY

RUN python -m pip check

# Copy app code + static assets baked into image
COPY src ./src
COPY rubrics ./rubrics
COPY configs ./configs
COPY rules ./rules
COPY datasets/math_basic ./datasets/math_basic
COPY datasets/validation ./datasets/validation
COPY tools ./tools

# Create data directories (will be overlaid by volume mounts)
RUN mkdir -p /data/reports /data/baselines /data/datasets \
    && chown -R appuser:appuser /data

# D1 env-independent paths: default for Docker layout
ENV LLM_JUDGE_CONFIGS_DIR=/app/configs \
    LLM_JUDGE_DATA_DIR=/data

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import httpx; r=httpx.get('http://localhost:8000/ready', timeout=4); exit(0 if r.status_code==200 else 1)"]

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
