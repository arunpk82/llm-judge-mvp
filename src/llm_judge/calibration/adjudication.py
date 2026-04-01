"""
Human Adjudication Queue and Confidence Router (EPIC 7.3).

Routes low-confidence LLM evaluations to a human adjudication queue.
Human decisions feed back into calibration data, creating a closed loop:
  LLM scores → low confidence → human review → calibration improves.

Queue states: pending → claimed → resolved
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_judge.paths import state_root

logger = logging.getLogger(__name__)

DEFAULT_QUEUE_PATH = state_root() / "adjudication" / "queue.jsonl"

QUEUE_STATES = ("pending", "claimed", "resolved")
VALID_QUEUE_TRANSITIONS: dict[str, list[str]] = {
    "pending": ["claimed", "resolved"],
    "claimed": ["resolved", "pending"],  # can release back to pending
    "resolved": [],  # terminal
}


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class AdjudicationCase:
    """A case routed to human adjudication."""
    case_id: str
    run_id: str
    rubric_id: str
    state: str  # pending | claimed | resolved

    # From the LLM evaluation
    llm_decision: str
    llm_confidence: float
    llm_scores: dict[str, int]

    # Case content for the reviewer
    conversation: list[dict[str, str]]
    candidate_answer: str

    # Human resolution (filled when resolved)
    human_decision: str | None = None
    human_scores: dict[str, int] | None = None
    human_notes: str | None = None
    assigned_to: str | None = None

    # Timestamps
    created_at: str = ""
    resolved_at: str | None = None


# =====================================================================
# Confidence Router
# =====================================================================

def should_route_to_human(
    *,
    confidence: float,
    threshold: float = 0.6,
    flags: list[str] | None = None,
) -> tuple[bool, str]:
    """
    Decide whether a case should be routed to human adjudication.

    Routes to human when:
      1. Confidence below threshold
      2. Flags contain known disagreement indicators
    """
    if confidence < threshold:
        return True, f"Low confidence: {confidence:.2f} < {threshold:.2f}"

    if flags:
        for flag in flags:
            if "disagreement" in flag or "low_confidence" in flag:
                return True, f"Flag indicates uncertainty: {flag}"

    return False, "Confidence above threshold"


# =====================================================================
# Adjudication Queue
# =====================================================================

def enqueue_case(
    *,
    case_id: str,
    run_id: str,
    rubric_id: str,
    llm_decision: str,
    llm_confidence: float,
    llm_scores: dict[str, int],
    conversation: list[dict[str, str]],
    candidate_answer: str,
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> AdjudicationCase:
    """Add a case to the human adjudication queue."""
    case = AdjudicationCase(
        case_id=case_id,
        run_id=run_id,
        rubric_id=rubric_id,
        state="pending",
        llm_decision=llm_decision,
        llm_confidence=llm_confidence,
        llm_scores=llm_scores,
        conversation=conversation,
        candidate_answer=candidate_answer,
        created_at=_utc_now_iso(),
    )

    queue_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "case_id": case.case_id,
        "run_id": case.run_id,
        "rubric_id": case.rubric_id,
        "state": case.state,
        "llm_decision": case.llm_decision,
        "llm_confidence": case.llm_confidence,
        "llm_scores": case.llm_scores,
        "conversation": case.conversation,
        "candidate_answer": case.candidate_answer,
        "created_at": case.created_at,
    }

    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")

    return case


def load_queue(
    *,
    state: str | None = None,
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> list[dict[str, Any]]:
    """Load adjudication queue, optionally filtered by state."""
    if not queue_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    # Track latest state per case_id (append-only log)
    latest: dict[str, dict[str, Any]] = {}

    with queue_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                cid = obj.get("case_id", "")
                latest[cid] = obj
            except json.JSONDecodeError:
                continue

    for obj in latest.values():
        if state and obj.get("state") != state:
            continue
        entries.append(obj)

    return entries


def resolve_case(
    *,
    case_id: str,
    human_decision: str,
    human_scores: dict[str, int] | None = None,
    human_notes: str = "",
    adjudicator: str = "qa_lead",
    queue_path: Path = DEFAULT_QUEUE_PATH,
) -> dict[str, Any]:
    """
    Resolve an adjudication case with human decision.

    Appends the resolution to the queue log. The human decision can be
    fed back into calibration data for the LLM judge.
    """
    # Find current state
    current_entries = load_queue(queue_path=queue_path)
    matching = [e for e in current_entries if e.get("case_id") == case_id]

    if not matching:
        raise ValueError(f"Adjudication case not found: {case_id}")

    current = matching[0]
    current_state = current.get("state", "pending")

    if current_state == "resolved":
        raise ValueError(f"Case {case_id} already resolved")

    resolution = {
        **current,
        "state": "resolved",
        "human_decision": human_decision,
        "human_scores": human_scores,
        "human_notes": human_notes,
        "assigned_to": adjudicator,
        "resolved_at": _utc_now_iso(),
    }

    with queue_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(resolution, sort_keys=True) + "\n")

    # Emit event to cross-capability registry
    try:
        from llm_judge.eval.event_registry import append_event

        append_event(
            event_type="eval_run",
            source="calibration/adjudication.py",
            actor=adjudicator,
            related_ids={
                "case_id": case_id,
                "run_id": current.get("run_id", ""),
                "rubric_id": current.get("rubric_id", ""),
            },
            payload={
                "action": "human_adjudication",
                "llm_decision": current.get("llm_decision"),
                "human_decision": human_decision,
                "agreement": current.get("llm_decision") == human_decision,
            },
        )
    except Exception:
        pass  # best-effort

    return resolution


def get_queue_stats(queue_path: Path = DEFAULT_QUEUE_PATH) -> dict[str, Any]:
    """Get queue statistics."""
    entries = load_queue(queue_path=queue_path)
    state_counts: dict[str, int] = {}
    agreement_count = 0
    resolved_count = 0

    for e in entries:
        state = e.get("state", "unknown")
        state_counts[state] = state_counts.get(state, 0) + 1

        if state == "resolved":
            resolved_count += 1
            if e.get("llm_decision") == e.get("human_decision"):
                agreement_count += 1

    return {
        "total_cases": len(entries),
        "by_state": state_counts,
        "resolved": resolved_count,
        "llm_human_agreement_rate": (
            round(agreement_count / resolved_count, 4)
            if resolved_count > 0 else None
        ),
    }
