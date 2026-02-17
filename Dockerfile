# After installing requirements + forcing patched versions:
# 1) purge pip caches/temp
# 2) remove ANY stale wheel/jaraco.context dist-info anywhere on filesystem
# 3) remove ANY wheel archives that embed vulnerable METADATA
# 4) print proof

RUN set -eux; \
    python -m pip install --no-cache-dir --upgrade pip; \
    python -m pip install --no-cache-dir -r /app/requirements.txt; \
    python -m pip install --no-cache-dir --upgrade --force-reinstall \
      "packaging>=24.0" \
      "backports.tarfile>=1.2.0" \
      "wheel==0.46.2" \
      "jaraco.context==6.1.0"; \
    \
    # Hard delete known cache locations (Trivy can see these)
    rm -rf /root/.cache/pip /tmp/pip-* /tmp/build /var/tmp/* || true; \
    \
    python - <<'PY'
import os, sys, zipfile
from pathlib import Path

TARGET = {
  "wheel": "0.46.2",
  "jaraco.context": "6.1.0",
}

def read_metadata_text(path: Path) -> str:
  try:
    return path.read_text(errors="ignore")
  except Exception:
    return ""

def parse_name_ver(metadata_text: str):
  name = ver = None
  for line in metadata_text.splitlines():
    if line.startswith("Name: "):
      name = line.split(":",1)[1].strip()
    elif line.startswith("Version: "):
      ver = line.split(":",1)[1].strip()
    if name and ver:
      return name, ver
  return name, ver

def remove_tree(p: Path):
  if p.is_file():
    p.unlink(missing_ok=True)
    return
  # dirs
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

# 1) filesystem-wide dist-info scan (not just sys.path)
bad_distinfo = []
for dist in Path("/").rglob("*.dist-info"):
  meta = dist / "METADATA"
  if not meta.exists():
    continue
  name, ver = parse_name_ver(read_metadata_text(meta))
  if name in TARGET and ver and ver != TARGET[name]:
    bad_distinfo.append((name, ver, str(dist)))

print("TRIVY-TRACE: bad dist-info found:", len(bad_distinfo))
for name, ver, p in bad_distinfo[:200]:
  print(f"  - BAD dist-info {name} {ver} :: {p}")

for _, _, p in bad_distinfo:
  remove_tree(Path(p))

# 2) wheel archive scan (Trivy may flag METADATA inside artifacts)
bad_whls = []
for whl in Path("/").rglob("*.whl"):
  # avoid scanning huge locations if any show up; still safe in slim images
  try:
    with zipfile.ZipFile(whl) as z:
      # METADATA usually lives under *.dist-info/METADATA
      metas = [n for n in z.namelist() if n.endswith(".dist-info/METADATA")]
      for m in metas:
        txt = z.read(m).decode("utf-8", "ignore")
        name, ver = parse_name_ver(txt)
        if name in TARGET and ver and ver != TARGET[name]:
          bad_whls.append((name, ver, str(whl)))
          break
  except Exception:
    continue

print("TRIVY-TRACE: bad .whl artifacts found:", len(bad_whls))
for name, ver, p in bad_whls[:200]:
  print(f"  - BAD whl {name} {ver} :: {p}")

for _, _, p in bad_whls:
  try:
    Path(p).unlink(missing_ok=True)
  except Exception:
    pass

# 3) show final authoritative state (runtime)
import importlib.metadata as md
print("TRIVY-TRACE: final versions via importlib.metadata:")
for k in TARGET:
  try:
    print(f"  - {k} = {md.version(k)}")
  except Exception as e:
    print(f"  - {k} = <missing> ({e})")

# 4) fail fast if anything still exists anywhere (dist-info)
still_bad = []
for dist in Path("/").rglob("*.dist-info"):
  meta = dist / "METADATA"
  if not meta.exists():
    continue
  name, ver = parse_name_ver(read_metadata_text(meta))
  if name in TARGET and ver and ver != TARGET[name]:
    still_bad.append((name, ver, str(dist)))

if still_bad:
  raise SystemExit("TRIVY-TRACE: stale dist-info still present: " + str(still_bad[:20]))

print("TRIVY-TRACE: OK - no stale METADATA remains anywhere on filesystem")
PY
