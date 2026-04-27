"""L1 sub-capability instrumentation coverage (CAP-7 Phase 1).

Four conceptual boundaries are emitted on every L1-enabled run:

* ``l1_input_preparation`` — SOFT (sentence segmentation runs
  unconditionally for all layers).
* ``l1_substring_matching`` — CLEAN (the per-sentence loop is
  L1-specific; ``started`` + ``completed`` events carry duration).
* ``l1_aggregation`` — SOFT (interleaved with the matching loop).
* ``l1_result_emission`` — SOFT (no discrete emission step;
  ``layer_stats`` flows through to ``HallucinationResult``).

Soft boundaries emit ``sub_capability_skipped`` with explicit
``reason`` rather than 0ms-duration events (CAP-7 Phase 1 D5).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from structlog.testing import capture_logs

from llm_judge.calibration.hallucination import check_hallucination

L1_SUB_CAPS = (
    "l1_input_preparation",
    "l1_substring_matching",
    "l1_aggregation",
    "l1_result_emission",
)
SOFT_BOUNDARIES = (
    "l1_input_preparation",
    "l1_aggregation",
    "l1_result_emission",
)
CLEAN_BOUNDARIES = ("l1_substring_matching",)


def _l1_only_kwargs(case_id: str = "test-l1") -> dict[str, Any]:
    return dict(
        case_id=case_id,
        skip_embeddings=True,
        l1_enabled=True,
        l2_enabled=False,
        l3_enabled=False,
        l4_enabled=False,
        gate2_routing="none",
    )


def _l1_events(
    logs: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        log
        for log in logs
        if log.get("event")
        in (
            "sub_capability_started",
            "sub_capability_completed",
            "sub_capability_skipped",
            "sub_capability_failed",
        )
        and log.get("capability_id") == "CAP-7"
        and isinstance(log.get("sub_capability_id"), str)
        and log["sub_capability_id"].startswith("l1_")
    ]


def test_l1_emits_sub_capability_events_for_all_four_boundaries() -> None:
    with capture_logs() as logs:
        check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France and its largest city.",
            **_l1_only_kwargs(),
        )

    seen = {e["sub_capability_id"] for e in _l1_events(logs)}
    assert seen == set(L1_SUB_CAPS), (
        f"L1 boundaries: got {sorted(seen)}, expected {sorted(L1_SUB_CAPS)}"
    )


def test_l1_substring_matching_clean_boundary_emits_started_and_completed() -> None:
    with capture_logs() as logs:
        check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France and its largest city.",
            **_l1_only_kwargs(),
        )

    started = [
        log
        for log in logs
        if log.get("event") == "sub_capability_started"
        and log.get("capability_id") == "CAP-7"
        and log.get("sub_capability_id") == "l1_substring_matching"
    ]
    completed = [
        log
        for log in logs
        if log.get("event") == "sub_capability_completed"
        and log.get("capability_id") == "CAP-7"
        and log.get("sub_capability_id") == "l1_substring_matching"
    ]
    assert len(started) == 1, f"expected 1 started, got {len(started)}"
    assert len(completed) == 1, f"expected 1 completed, got {len(completed)}"
    assert completed[0]["status"] == "success"
    assert completed[0]["duration_ms"] >= 0.0
    assert completed[0]["request_id"] == "test-l1"


def test_l1_soft_boundaries_emit_skipped_with_reason() -> None:
    with capture_logs() as logs:
        check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France and its largest city.",
            **_l1_only_kwargs(),
        )

    for sub_cap in SOFT_BOUNDARIES:
        skipped = [
            log
            for log in logs
            if log.get("event") == "sub_capability_skipped"
            and log.get("capability_id") == "CAP-7"
            and log.get("sub_capability_id") == sub_cap
        ]
        assert len(skipped) == 1, (
            f"{sub_cap}: expected 1 skipped event, got {len(skipped)}"
        )
        reason = skipped[0].get("reason")
        assert isinstance(reason, str) and reason, (
            f"{sub_cap}: reason must be a non-empty string"
        )
        assert skipped[0]["request_id"] == "test-l1"


def test_l1_input_preparation_metadata_sentence_count_in_result() -> None:
    """Sentence count is observable via ``len(sentence_results)`` plus
    L1's ``layer_stats['L1']`` count of matched sentences. Sub-capability
    events themselves carry only the contract fields (event-shape tests
    enforce no extras), so metadata lives in the result payload.
    """
    response = "Paris is the capital of France. The Eiffel Tower stands in Paris."
    context = (
        "Paris is the capital of France. The Eiffel Tower stands in Paris and "
        "is its most famous landmark."
    )
    result = check_hallucination(
        response=response,
        context=context,
        **_l1_only_kwargs("test-l1-meta-sents"),
    )
    assert "L1" in result.layer_stats
    assert isinstance(result.layer_stats["L1"], int)


def test_l1_substring_matching_metadata_match_count_in_layer_stats() -> None:
    # Ensure the response sentence appears verbatim (including trailing
    # period) in the context so the cheapest substring check fires.
    response = "Paris is the capital of France."
    context = (
        "Background information follows. Paris is the capital of France. "
        "It is also home to the Eiffel Tower."
    )
    result = check_hallucination(
        response=response,
        context=context,
        **_l1_only_kwargs("test-l1-meta-match"),
    )
    assert result.layer_stats["L1"] >= 1, (
        f"L1 should match at least 1 sentence; layer_stats={result.layer_stats}"
    )


def test_l1_disabled_emits_no_sub_capability_events() -> None:
    with capture_logs() as logs:
        check_hallucination(
            response="Paris is the capital of France.",
            context="Paris is the capital of France.",
            case_id="test-l1-disabled",
            skip_embeddings=True,
            l1_enabled=False,
            l2_enabled=False,
            l3_enabled=False,
            l4_enabled=False,
            gate2_routing="none",
        )

    assert _l1_events(logs) == [], (
        "L1 disabled must emit no sub-capability events"
    )
