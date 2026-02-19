# Containerization Strategy

## 1. Purpose

This document defines the containerization approach for the LLM Judge MVP platform.

Containerization transforms the application from a development service into a deployable, immutable, production-grade artifact.

The objectives of containerization are to:

* Produce reproducible runtime environments
* Ensure environment consistency across stages
* Enable horizontal scalability
* Support CI/CD automation
* Strengthen security posture

---

## 2. Containerization Principles

The container image must:

* Be minimal and secure
* Use immutable builds
* Avoid running as root
* Externalize configuration
* Expose health endpoints
* Support horizontal scaling

Container images must be treated as immutable artifacts.

---

## 3. Target Image Architecture

### Base Image

Recommended:

```
python:3.11-slim
```

Rationale:

* Smaller attack surface
* Reduced image size
* Faster build and deploy cycles

---

## 4. Multi-Stage Build Strategy

A multi-stage Docker build is recommended to reduce final image size.

### Stage 1 – Builder

* Install Poetry
* Install dependencies
* Build wheel (optional future optimization)

### Stage 2 – Runtime

* Copy installed dependencies
* Copy application source
* Create non-root user
* Set working directory
* Expose port
* Define startup command

---

## 5. Recommended Dockerfile (Reference Implementation)

```
# ---------- Builder Stage ----------
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --only main

# ---------- Runtime Stage ----------
FROM python:3.11-slim

WORKDIR /app

# Create non-root user
RUN useradd -m appuser

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src ./src
COPY rubrics ./rubrics

USER appuser

EXPOSE 8000

CMD ["uvicorn", "llm_judge.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This configuration produces a minimal and secure runtime image.

---

## 6. Health Check Integration

The container should expose the `/health` endpoint.

Future enhancement:

```
HEALTHCHECK CMD curl --fail http://localhost:8000/health || exit 1
```

This enables orchestration-level liveness monitoring.

---

## 7. Image Versioning Strategy

Images should be tagged using semantic versioning.

Recommended pattern:

```
llm-judge-mvp:1.0.0
llm-judge-mvp:1.0.1
llm-judge-mvp:latest
```

The `latest` tag should not be relied upon for production deployments.

---

## 8. Security Considerations

The container must:

* Run as non-root user
* Avoid embedding secrets
* Use minimal base image
* Keep dependency footprint small
* Regularly scan for vulnerabilities

Recommended tools:

* Trivy
* Docker Scout
* Snyk

---

## 9. Runtime Configuration

All runtime configuration must be provided via environment variables.

Examples:

* JUDGE_ENGINE
* RUBRIC_REGISTRY_PATH
* LOG_LEVEL
* LLM_PROVIDER
* LLM_API_KEY

No secrets should be hardcoded or baked into the image.

---

## 10. Scalability Model

The service is stateless and horizontally scalable.

Container replicas may be scaled via:

* Kubernetes Deployment
* ECS Service
* Docker Swarm

Stateless design ensures safe scaling without session affinity.

---

## 11. CI/CD Integration

Containerization enables the following pipeline:

1. Code commit
2. CI validation (ruff, mypy, pytest)
3. Docker image build
4. Security scan
5. Push to container registry
6. Deploy to staging
7. Promote to production

Only CI-validated images may be deployed.

---

## 12. Future Hardening Enhancements

Planned improvements:

* Distroless base image
* Multi-architecture builds (amd64, arm64)
* SBOM generation
* Supply chain signing (cosign)
* Runtime security policies

---

## 13. Guiding Principles

* Immutable artifacts
* Minimal attack surface
* Deterministic builds
* Security by default
* Production parity across environments

Containerization formalizes the service as a deployable enterprise-grade artifact.
