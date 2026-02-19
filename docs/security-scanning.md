# Security Scanning & CVE Mitigation

## Background

CVE-2026-24049 (wheel)
CVE-2026-23949 (jaraco.context)

Trivy scans `METADATA` files, not only runtime imports.

This means even stale `.dist-info` directories can cause failures.

---

## Root Cause

- `pip uninstall` removes package
- But stale metadata may remain in:
  - site-packages
  - pip cache
  - embedded wheel archives

Trivy detects those artifacts.

---

## Mitigation Strategy

1. Force reinstall patched versions
2. Remove pip cache
3. Scan entire filesystem for:
   - `.dist-info`
   - `.whl`
4. Parse `METADATA`
5. Remove stale artifacts
6. Fail build if any remain

---

## Validation Logic

The Dockerfile:

- Performs filesystem-wide scan
- Verifies final versions via `importlib.metadata`
- Fails build on mismatch

---

## Future Maintenance

If a new CVE appears:

1. Add version pin in Dockerfile
2. Add to TARGET dictionary
3. Commit + rebuild
4. Validate Trivy output

---

This ensures zero-regression security posture.
