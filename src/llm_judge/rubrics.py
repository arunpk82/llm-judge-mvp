from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Rubric:
    id: str
    version: str
    dimensions: List[str]
    pass_threshold: float
    max_score_per_dimension: int = 5


# Versioned rubric registry
RUBRICS: Dict[str, Rubric] = {
    "chat_quality": Rubric(
        id="chat_quality",
        version="v1",
        dimensions=[
            "relevance",
            "clarity",
            "correctness",
            "tone",
        ],
        pass_threshold=3.5,
    ),
}
