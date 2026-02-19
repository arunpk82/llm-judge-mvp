# Release Process

## Step 1: Versioning

Use semantic versioning:

MAJOR.MINOR.PATCH

Example:
v0.1.15

---

## Step 2: Create Tag
git tag -a v0.1.15 -m "release: v0.1.15"
git push origin v0.1.15

---

## Step 3: CI Execution

Release workflow performs:
- Docker build (no cache, pull latest base image)
- Metadata cleanup
- Trivy vulnerability scan
- Push to GHCR

---

## Step 4: Verification

Check:
- GHCR image digest
- Trivy scan result
- Image tag availability

---

## Rollback

To rollback:

docker pull ghcr.io/arunpk82/llm-judge-mvp:<previous-version>
