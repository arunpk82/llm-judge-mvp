from __future__ import annotations

from pathlib import Path

from llm_judge.rubric_store import get_rubric
from llm_judge.rules.engine import load_plan_for_rubric, run_rules
from llm_judge.rules.types import RuleContext
from llm_judge.schemas import Message, PredictRequest


def test_engine_loads_plan_and_runs_rules(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "rules" / "chat_quality").mkdir(parents=True, exist_ok=True)

    (tmp_path / "configs" / "rules" / "chat_quality" / "v1.yaml").write_text(
        """
rubric_id: chat_quality
version: v1
rules:
  - id: correctness.definition_sanity
    enabled: true
  - id: quality.repetition_basic
    enabled: true
    params:
      ngram_n: 2
      min_repeated_ngram_count: 2
  - id: quality.nonsense_basic
    enabled: true
""".strip(),
        encoding="utf-8",
    )

    rubric = get_rubric("chat_quality")
    plan = load_plan_for_rubric("chat_quality", "v1")

    # --- Run 1: deterministic correctness.definition_sanity ---
    req1 = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="Blockchain is blockchain.",
        rubric_id="chat_quality",
    )
    rr1 = run_rules(RuleContext(request=req1, rubric=rubric), plan)
    ids1 = {f.id for f in rr1.flags}
    assert "correctness.definition_sanity" in ids1

    # --- Run 2: deterministic quality.repetition_basic ---
    req2 = PredictRequest(
        conversation=[Message(role="user", content="Define blockchain")],
        candidate_answer="Line A\nLine A\nLine A\nLine B",
        rubric_id="chat_quality",
    )
    rr2 = run_rules(RuleContext(request=req2, rubric=rubric), plan)
    ids2 = {f.id for f in rr2.flags}
    assert "quality.repetition_basic" in ids2
