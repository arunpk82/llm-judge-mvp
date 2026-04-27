"""
Robustness Properties — Calibration Diagnostics (Category 5, Properties 5.3–5.7).

These assess the JUDGE, not the response. Run periodically against
golden datasets, not per-response.

5.3 Self-Preference Bias — judge favors LLM-style writing?
5.4 Consistency — paraphrase stability
5.5 Adversarial Resilience — quality under attack
5.6 Edge Case Handling — empty/contradictory/unusual inputs
5.7 Reproducibility — same input → same output (temperature=0)
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

from llm_judge.judge_base import JudgeEngine
from llm_judge.schemas import Message, PredictRequest

logger = logging.getLogger(__name__)


# =====================================================================
# 5.3 Self-Preference Bias
# =====================================================================


@dataclass
class SelfPreferenceResult:
    """Result of self-preference bias check."""

    tested: int
    llm_written_avg_score: float
    human_written_avg_score: float
    preference_delta: float
    status: str  # PASS / FAIL / INSUFFICIENT_DATA
    flags: list[str] = field(default_factory=list)


def check_self_preference_bias(
    *,
    judge: JudgeEngine,
    llm_written_cases: list[dict[str, Any]],
    human_written_cases: list[dict[str, Any]],
    threshold: float = 0.5,
) -> SelfPreferenceResult:
    """
    Check if judge systematically favors LLM-style writing (Property 5.3).

    Compares average scores on LLM-written vs human-written responses
    that have similar human-rated quality. A significant delta indicates
    self-preference bias.
    """
    if len(llm_written_cases) < 3 or len(human_written_cases) < 3:
        return SelfPreferenceResult(
            tested=0,
            llm_written_avg_score=0,
            human_written_avg_score=0,
            preference_delta=0,
            status="INSUFFICIENT_DATA",
        )

    llm_scores: list[float] = []
    human_scores: list[float] = []

    for case in llm_written_cases:
        try:
            resp = _evaluate_case(judge, case)
            llm_scores.append(resp.overall_score)
        except Exception:
            continue

    for case in human_written_cases:
        try:
            resp = _evaluate_case(judge, case)
            human_scores.append(resp.overall_score)
        except Exception:
            continue

    if len(llm_scores) < 3 or len(human_scores) < 3:
        return SelfPreferenceResult(
            tested=len(llm_scores) + len(human_scores),
            llm_written_avg_score=0,
            human_written_avg_score=0,
            preference_delta=0,
            status="INSUFFICIENT_DATA",
        )

    llm_avg = statistics.mean(llm_scores)
    human_avg = statistics.mean(human_scores)
    delta = llm_avg - human_avg

    flags: list[str] = []
    status = "PASS"
    if abs(delta) > threshold:
        status = "FAIL"
        flags.append(f"self_preference_bias:{delta:.2f}")

    return SelfPreferenceResult(
        tested=len(llm_scores) + len(human_scores),
        llm_written_avg_score=round(llm_avg, 2),
        human_written_avg_score=round(human_avg, 2),
        preference_delta=round(delta, 2),
        status=status,
        flags=flags,
    )


# =====================================================================
# 5.4 Consistency
# =====================================================================


@dataclass
class ConsistencyResult:
    """Result of consistency (paraphrase stability) check."""

    tested: int
    consistent: int
    inconsistent: int
    avg_score_delta: float
    status: str
    flags: list[str] = field(default_factory=list)


def check_consistency(
    *,
    judge: JudgeEngine,
    paraphrase_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    max_delta: float = 1.0,
) -> ConsistencyResult:
    """
    Check paraphrase stability (Property 5.4).

    Given pairs of semantically equivalent queries with different wording,
    the judge should produce similar scores for both.
    """
    if len(paraphrase_pairs) < 3:
        return ConsistencyResult(
            tested=0,
            consistent=0,
            inconsistent=0,
            avg_score_delta=0,
            status="INSUFFICIENT_DATA",
        )

    consistent = 0
    inconsistent = 0
    deltas: list[float] = []

    for case_a, case_b in paraphrase_pairs:
        try:
            resp_a = _evaluate_case(judge, case_a)
            resp_b = _evaluate_case(judge, case_b)
            delta = abs(resp_a.overall_score - resp_b.overall_score)
            deltas.append(delta)

            if delta <= max_delta:
                consistent += 1
            else:
                inconsistent += 1
        except Exception:
            continue

    if not deltas:
        return ConsistencyResult(
            tested=0,
            consistent=0,
            inconsistent=0,
            avg_score_delta=0,
            status="INSUFFICIENT_DATA",
        )

    avg_delta = statistics.mean(deltas)
    flags: list[str] = []
    total = consistent + inconsistent
    status = "PASS" if inconsistent / total < 0.15 else "FAIL"

    if status == "FAIL":
        flags.append(f"consistency_failures:{inconsistent}/{total}")

    return ConsistencyResult(
        tested=total,
        consistent=consistent,
        inconsistent=inconsistent,
        avg_score_delta=round(avg_delta, 2),
        status=status,
        flags=flags,
    )


# =====================================================================
# 5.5 Adversarial Resilience
# =====================================================================


@dataclass
class AdversarialResult:
    """Result of adversarial resilience check."""

    tested: int
    passed: int
    failed: int
    pass_rate: float
    failed_cases: list[str] = field(default_factory=list)
    status: str = "PASS"
    flags: list[str] = field(default_factory=list)


def check_adversarial_resilience(
    *,
    judge: JudgeEngine,
    adversarial_cases: list[dict[str, Any]],
    min_pass_rate: float = 0.85,
) -> AdversarialResult:
    """
    Check judge quality under adversarial inputs (Property 5.5).

    Each adversarial case has an expected_decision. The judge should
    produce the correct decision despite adversarial input design.
    """
    if len(adversarial_cases) < 3:
        return AdversarialResult(
            tested=0,
            passed=0,
            failed=0,
            pass_rate=0,
            status="INSUFFICIENT_DATA",
        )

    passed = 0
    failed = 0
    failed_ids: list[str] = []

    for case in adversarial_cases:
        try:
            resp = _evaluate_case(judge, case)
            expected = case.get("expected_decision", case.get("human_decision"))
            if resp.decision == expected:
                passed += 1
            else:
                failed += 1
                failed_ids.append(str(case.get("case_id", "unknown")))
        except Exception:
            failed += 1
            failed_ids.append(str(case.get("case_id", "error")))

    total = passed + failed
    rate = passed / total if total > 0 else 0
    status = "PASS" if rate >= min_pass_rate else "FAIL"
    flags: list[str] = []
    if status == "FAIL":
        flags.append(f"adversarial_failures:{failed}/{total}")

    return AdversarialResult(
        tested=total,
        passed=passed,
        failed=failed,
        pass_rate=round(rate, 4),
        failed_cases=failed_ids[:10],
        status=status,
        flags=flags,
    )


# =====================================================================
# 5.6 Edge Case Handling
# =====================================================================


@dataclass
class EdgeCaseResult:
    """Result of edge case handling check."""

    tested: int
    handled_gracefully: int
    crashed: int
    unexpected_results: int
    details: list[dict[str, str]] = field(default_factory=list)
    status: str = "PASS"
    flags: list[str] = field(default_factory=list)


def check_edge_cases(
    *,
    judge: JudgeEngine,
    rubric_id: str,
) -> EdgeCaseResult:
    """
    Check judge behavior with unusual inputs (Property 5.6).

    Tests: empty context, very short response, very long response,
    contradictory conversation, single-word response.
    """
    edge_cases: list[dict[str, Any]] = [
        {
            "name": "empty_context",
            "conversation": [{"role": "user", "content": "Help"}],
            "candidate_answer": "I can help you with that. What do you need?",
        },
        {
            "name": "single_word_response",
            "conversation": [
                {"role": "user", "content": "What is your return policy?"}
            ],
            "candidate_answer": "Yes.",
        },
        {
            "name": "very_long_response",
            "conversation": [{"role": "user", "content": "Summarize briefly."}],
            "candidate_answer": "Here is the answer. " * 200,
        },
        {
            "name": "contradictory_conversation",
            "conversation": [
                {"role": "user", "content": "I want to cancel my order."},
                {"role": "assistant", "content": "Your order has been shipped."},
                {"role": "user", "content": "I never placed an order."},
            ],
            "candidate_answer": "I understand your confusion. Let me look into this.",
        },
    ]

    handled = 0
    crashed = 0
    unexpected = 0
    details: list[dict[str, str]] = []
    flags: list[str] = []

    for ec in edge_cases:
        try:
            conv = ec["conversation"]
            messages = [Message(role=m["role"], content=m["content"]) for m in conv]
            request = PredictRequest(
                conversation=messages,
                candidate_answer=str(ec["candidate_answer"]),
                rubric_id=rubric_id,
            )
            resp = judge.evaluate(request)

            case_name = str(ec["name"])
            if resp.decision in ("pass", "fail") and 1.0 <= resp.overall_score <= 5.0:
                handled += 1
                details.append(
                    {"case": case_name, "result": "handled", "decision": resp.decision}
                )
            else:
                unexpected += 1
                details.append(
                    {
                        "case": case_name,
                        "result": "unexpected",
                        "decision": resp.decision,
                    }
                )
        except Exception as e:
            crashed += 1
            details.append(
                {
                    "case": str(ec.get("name", "unknown")),
                    "result": "crashed",
                    "error": str(e)[:60],
                }
            )

    total = handled + crashed + unexpected
    if crashed > 0:
        flags.append(f"edge_case_crashes:{crashed}")
    status = "PASS" if crashed == 0 and unexpected == 0 else "FAIL"

    return EdgeCaseResult(
        tested=total,
        handled_gracefully=handled,
        crashed=crashed,
        unexpected_results=unexpected,
        details=details,
        status=status,
        flags=flags,
    )


# =====================================================================
# 5.7 Reproducibility
# =====================================================================


@dataclass
class ReproducibilityResult:
    """Result of reproducibility check."""

    tested: int
    identical: int
    different: int
    identity_rate: float
    score_deltas: list[float] = field(default_factory=list)
    status: str = "PASS"
    flags: list[str] = field(default_factory=list)


def check_reproducibility(
    *,
    judge: JudgeEngine,
    cases: list[dict[str, Any]],
    max_cases: int = 20,
    min_identity_rate: float = 0.90,
) -> ReproducibilityResult:
    """
    Check same input → same output with temperature=0 (Property 5.7).

    Runs each case twice and compares decisions and scores.
    """
    sample = cases[:max_cases]

    if len(sample) < 3:
        return ReproducibilityResult(
            tested=0,
            identical=0,
            different=0,
            identity_rate=0,
            status="INSUFFICIENT_DATA",
        )

    identical = 0
    different = 0
    deltas: list[float] = []

    for case in sample:
        try:
            resp1 = _evaluate_case(judge, case)
            resp2 = _evaluate_case(judge, case)

            delta = abs(resp1.overall_score - resp2.overall_score)
            deltas.append(delta)

            if resp1.decision == resp2.decision and delta < 0.01:
                identical += 1
            else:
                different += 1
        except Exception:
            continue

    total = identical + different
    if total == 0:
        return ReproducibilityResult(
            tested=0,
            identical=0,
            different=0,
            identity_rate=0,
            status="INSUFFICIENT_DATA",
        )

    rate = identical / total
    flags: list[str] = []
    status = "PASS" if rate >= min_identity_rate else "FAIL"

    if status == "FAIL":
        flags.append(f"reproducibility_failures:{different}/{total}")

    return ReproducibilityResult(
        tested=total,
        identical=identical,
        different=different,
        identity_rate=round(rate, 4),
        score_deltas=deltas,
        status=status,
        flags=flags,
    )


# =====================================================================
# Helper
# =====================================================================


def _evaluate_case(judge: JudgeEngine, case: dict[str, Any]) -> Any:
    """Evaluate a single case dict through the judge."""
    messages = [Message(**m) for m in case.get("conversation", [])]
    request = PredictRequest(
        conversation=messages,
        candidate_answer=str(case.get("candidate_answer", "")),
        rubric_id=str(case.get("rubric_id", "chat_quality")),
    )
    return judge.evaluate(request)
