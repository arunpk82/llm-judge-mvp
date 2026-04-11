from __future__ import annotations

import os
from contextlib import asynccontextmanager

import httpx
import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from llm_judge.paths import validate_paths
from llm_judge.runtime import get_judge_engine
from llm_judge.schemas import PredictRequest, PredictResponse

logger = structlog.get_logger()

judge_engine = get_judge_engine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("service.startup", service="llm-judge")
    yield
    logger.info("service.shutdown", service="llm-judge")


app = FastAPI(
    title="LLM Judge MVP",
    version="0.1.0",
    description="LLM-as-a-Judge MVP service",
    lifespan=lifespan,
)

# Global engine (initialized on startup)
# judge_engine: JudgeEngine | None = None


@app.on_event("startup")
async def on_startup() -> None:
    global judge_engine
    logger.info("service.startup", service="llm-judge")

    judge_engine = get_judge_engine()
    logger.info("judge.engine.loaded", engine=type(judge_engine).__name__)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    logger.info("service.shutdown", service="llm-judge")


# =====================================================================
# Health endpoints (EPIC-D2)
# =====================================================================


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is running."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    """Readiness probe — validates config, storage, and datasets.

    Returns 200 with per-check details when all checks pass.
    Returns 503 with per-check details when any check fails.

    Uses the same ``config_root()`` / ``state_root()`` / ``datasets_root()``
    helpers as all runtime code (EPIC-D1), so the readiness check validates
    the exact paths that production code will use.
    """
    checks = validate_paths()
    status_code = 200 if checks["ok"] else 503

    logger.info(
        "readiness.check",
        ok=checks["ok"],
        status_code=status_code,
    )

    return JSONResponse(
        content={"status": "ready" if checks["ok"] else "not_ready", **checks},
        status_code=status_code,
    )


@app.get("/health/dependencies")
async def health_dependencies() -> JSONResponse:
    """Dependency probe — checks external service reachability.

    When ``JUDGE_ENGINE`` is set to an LLM provider (not ``deterministic``),
    verifies the provider API is reachable with a lightweight HTTP request.
    Returns per-dependency detail in all cases.
    """
    engine_choice = os.getenv("JUDGE_ENGINE", "deterministic").strip().lower()
    deps: dict[str, dict] = {}
    all_ok = True

    # --- LLM provider check ---
    if engine_choice == "deterministic":
        deps["llm_provider"] = {
            "status": "not_configured",
            "engine": engine_choice,
            "ok": True,
        }
    else:
        provider_ok, detail = _check_llm_provider(engine_choice)
        deps["llm_provider"] = {
            "status": "reachable" if provider_ok else "unreachable",
            "engine": engine_choice,
            "ok": provider_ok,
            **detail,
        }
        if not provider_ok:
            all_ok = False

    status_code = 200 if all_ok else 503

    logger.info(
        "dependency.check",
        ok=all_ok,
        engine=engine_choice,
        status_code=status_code,
    )

    return JSONResponse(
        content={"status": "healthy" if all_ok else "unhealthy", "dependencies": deps},
        status_code=status_code,
    )


def _check_llm_provider(engine: str) -> tuple[bool, dict]:
    """Lightweight reachability check for an LLM provider.

    Returns (ok, detail_dict). Does NOT send a real inference request —
    just checks that the API endpoint responds (HTTP-level health).
    """
    # Map engine names to their health/reachability URLs
    provider_urls: dict[str, tuple[str, str | None]] = {
        "gemini": (
            "https://generativelanguage.googleapis.com/",
            os.getenv("GEMINI_API_KEY"),
        ),
        "openai": (
            "https://api.openai.com/v1/models",
            os.getenv("LLM_API_KEY"),
        ),
        "llm": (
            "https://api.openai.com/v1/models",
            os.getenv("LLM_API_KEY"),
        ),
        "groq": (
            "https://api.groq.com/openai/v1/models",
            os.getenv("GROQ_API_KEY"),
        ),
        "ollama": (
            os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/tags",
            None,  # Ollama doesn't need an API key
        ),
    }

    if engine not in provider_urls:
        return False, {"error": f"Unknown engine: {engine}"}

    url, api_key = provider_urls[engine]

    # Check API key presence (except ollama)
    if engine != "ollama" and not api_key:
        key_name = {
            "gemini": "GEMINI_API_KEY",
            "openai": "LLM_API_KEY",
            "llm": "LLM_API_KEY",
            "groq": "GROQ_API_KEY",
        }.get(engine, "API_KEY")
        return False, {"error": f"{key_name} not set"}

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            # Any HTTP response (even 401) means the service is reachable
            return True, {"http_status": resp.status_code}
    except httpx.ConnectError:
        return False, {"error": "Connection refused or DNS failure"}
    except httpx.TimeoutException:
        return False, {"error": "Request timed out (5s)"}
    except Exception as exc:
        return False, {"error": str(exc)}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    assert judge_engine is not None, "judge_engine not initialized"

    logger.info(
        "predict.called",
        rubric_id=request.rubric_id,
        turns=len(request.conversation),
    )

    return judge_engine.evaluate(request)
