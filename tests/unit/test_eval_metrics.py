from __future__ import annotations

from llm_judge.eval.metrics import compute_metrics


def test_metrics_computes_f1_for_pass_and_fail() -> None:
    judgments = [
        {"human_decision": "pass", "judge_decision": "pass"},
        {"human_decision": "pass", "judge_decision": "fail"},
        {"human_decision": "fail", "judge_decision": "fail"},
        {"human_decision": "fail", "judge_decision": "pass"},
    ]
    m = compute_metrics(judgments)
    assert m["n_cases_scored"] == 4

    # With this symmetric set, precision/recall/f1 should be 0.5 for both classes
    assert abs(m["f1_pass"] - 0.5) < 1e-9
    assert abs(m["f1_fail"] - 0.5) < 1e-9
    assert m["cohen_kappa"] is not None
