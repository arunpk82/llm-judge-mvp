from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

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


@app.get("/health")
async def health() -> dict[str, str]:
    logger.info("health.check")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    assert judge_engine is not None, "judge_engine not initialized"

    logger.info(
        "predict.called",
        rubric_id=request.rubric_id,
        turns=len(request.conversation),
    )

    return judge_engine.evaluate(request)
