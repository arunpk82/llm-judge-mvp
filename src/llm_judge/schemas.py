from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class PredictRequest(BaseModel):
    conversation: List[Message] = Field(..., min_length=1)
    candidate_answer: str = Field(..., min_length=1)
    rubric_id: str = Field(..., min_length=1)
    source_context: Optional[List[str]] = Field(
        default=None,
        description=(
            "RAG retrieval documents — the source documentation the LLM "
            "should have drawn from. Faithfulness properties (1.1-1.5) "
            "ground against these docs. When absent and a knowledge base "
            "is configured, the retriever populates this inside Gate 2."
        ),
    )


class PredictResponse(BaseModel):
    decision: Literal["pass", "fail"]
    overall_score: float = Field(..., ge=0.0, le=5.0)
    scores: dict[str, int] = Field(..., description="Per-dimension scores (1..5)")
    confidence: float = Field(..., ge=0.0, le=1.0)
    flags: List[str] = Field(default_factory=list)
    explanations: Optional[Dict[str, str]] = None
