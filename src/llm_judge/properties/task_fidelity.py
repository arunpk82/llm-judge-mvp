"""
Task Fidelity Properties (Category 4).

4.1 Instruction Following — detects explicit constraints in query, checks compliance
4.2 Format & Structure — validates output against expected format/schema

Almost entirely deterministic — no LLM needed.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =====================================================================
# 4.1 Instruction Following
# =====================================================================

_CONSTRAINT_PATTERNS = {
    "json_format": re.compile(
        r"\b(?:respond\s+in\s+json|return\s+json|json\s+format|as\s+json)\b",
        re.I,
    ),
    "bullet_limit": re.compile(
        r"\b(?:limit\s+to|at\s+most|no\s+more\s+than|maximum\s+of)\s+(\d+)\s+"
        r"(?:bullet|point|item|step)",
        re.I,
    ),
    "word_limit": re.compile(
        r"\b(?:limit\s+to|at\s+most|no\s+more\s+than|maximum\s+of|under)\s+"
        r"(\d+)\s+words?\b",
        re.I,
    ),
    "no_jargon": re.compile(
        r"\b(?:no\s+(?:technical\s+)?jargon|simple\s+language|plain\s+english|"
        r"avoid\s+technical\s+terms)\b",
        re.I,
    ),
    "language": re.compile(
        r"\b(?:respond|answer|reply)\s+in\s+(english|spanish|french|german|hindi)\b",
        re.I,
    ),
    "list_format": re.compile(
        r"\b(?:as\s+a\s+list|in\s+list\s+form|numbered\s+list|bullet\s+list)\b",
        re.I,
    ),
    "step_by_step": re.compile(
        r"\b(?:step\s+by\s+step|step-by-step)\b",
        re.I,
    ),
}


@dataclass
class InstructionResult:
    """Result of instruction following check."""

    case_id: str
    constraints_detected: list[str] = field(default_factory=list)
    constraints_violated: list[str] = field(default_factory=list)
    compliance_score: float = 1.0  # 1.0 = all constraints met
    flags: list[str] = field(default_factory=list)


def _count_bullets(text: str) -> int:
    """Count bullet points or numbered items in text."""
    bullet_pattern = re.compile(r"(?:^|\n)\s*(?:[-•*]|\d+[.)]\s)")
    return len(bullet_pattern.findall(text))


def check_instruction_following(
    *,
    query: str,
    response: str,
    case_id: str = "unknown",
) -> InstructionResult:
    """Check if response follows explicit constraints in the query."""
    detected: list[str] = []
    violated: list[str] = []
    flags: list[str] = []

    for constraint_type, pattern in _CONSTRAINT_PATTERNS.items():
        match = pattern.search(query)
        if match:
            detected.append(constraint_type)

            if constraint_type == "json_format":
                try:
                    json.loads(response.strip())
                except (json.JSONDecodeError, ValueError):
                    # Check if JSON is embedded in markdown fences
                    cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
                    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
                    try:
                        json.loads(cleaned)
                    except (json.JSONDecodeError, ValueError):
                        violated.append("json_format")

            elif constraint_type == "bullet_limit":
                limit = int(match.group(1))
                actual = _count_bullets(response)
                if actual > limit:
                    violated.append(f"bullet_limit:{actual}>{limit}")

            elif constraint_type == "word_limit":
                limit = int(match.group(1))
                actual = len(response.split())
                if actual > limit:
                    violated.append(f"word_limit:{actual}>{limit}")

    if violated:
        compliance = 1.0 - (len(violated) / max(len(detected), 1))
        flags.append(f"instruction_violations:{len(violated)}")
    else:
        compliance = 1.0

    return InstructionResult(
        case_id=case_id,
        constraints_detected=detected,
        constraints_violated=violated,
        compliance_score=round(compliance, 2),
        flags=flags,
    )


# =====================================================================
# 4.2 Format & Structure
# =====================================================================


@dataclass
class FormatResult:
    """Result of format & structure check."""

    case_id: str
    is_valid_json: bool = False
    has_required_fields: bool = True
    missing_fields: list[str] = field(default_factory=list)
    extra_observations: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def check_format_structure(
    *,
    response: str,
    expected_format: str | None = None,
    required_fields: list[str] | None = None,
    case_id: str = "unknown",
) -> FormatResult:
    """Check if response matches expected format/schema."""
    flags: list[str] = []
    is_json = False
    missing: list[str] = []
    observations: list[str] = []

    # Check JSON validity if expected
    if expected_format == "json":
        cleaned = re.sub(r"^```(?:json)?\s*", "", response.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
        try:
            parsed = json.loads(cleaned)
            is_json = True

            if required_fields:
                if isinstance(parsed, dict):
                    for f in required_fields:
                        if f not in parsed:
                            missing.append(f)
                else:
                    observations.append("json_not_object")
        except (json.JSONDecodeError, ValueError):
            is_json = False
            flags.append("invalid_json")

    # General structure checks
    if len(response.strip()) == 0:
        flags.append("empty_response")
        observations.append("empty_response")

    if missing:
        flags.append(f"missing_fields:{','.join(missing)}")

    return FormatResult(
        case_id=case_id,
        is_valid_json=is_json,
        has_required_fields=len(missing) == 0,
        missing_fields=missing,
        extra_observations=observations,
        flags=flags,
    )
