from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from llm_judge.schemas import Message, PredictRequest
from llm_judge.rules.engine import RuleEngine, load_plan_for_rubric, run_rules
from llm_judge.rules.registry import register
from llm_judge.rules.types import RuleContext


# ---- Helpers / dummy rules ----


class _Flag:
    def __init__(self, id: str, severity: str) -> None:
        self.id = id
        self.severity = severity


class _RR:
    def __init__(self, flags: list[Any]) -> None:
        self.flags = flags


@register("nonsense_basic")
class DummyNonsenseBasic:
    def apply(self, ctx: RuleContext, params: dict[str, Any]) -> _RR:
        # emit SHORT id to test prefix normalization when invoked as quality.nonsense_basic
        return _RR([_Flag("nonsense_basic", params.get("sev", "strong"))])


@register("definition_sanity")
class DummyDefinitionSanity:
    def apply(self, ctx: RuleContext, params: dict[str, Any]) -> _RR:
        return _RR([_Flag("definition_sanity", "strong")])


@register("boom_rule")
class DummyBoomRule:
    def apply(self, ctx: RuleContext, params: dict[str, Any]) -> _RR:
        raise RuntimeError("boom")


def _ctx_for(text: str, answer: str) -> RuleContext:
    req = PredictRequest(
        conversation=[Message(role="user", content=text)],
        candidate_answer=answer,
        rubric_id="chat_quality",
    )
    rubric = SimpleNamespace(rubric_id="chat_quality", version="v1", rules=None)
    return RuleContext(request=req, rubric=rubric)


# ---- Tests ----


def test_rule_engine_writes_ctx_flags_and_prefixes_fq_ids() -> None:
    ctx = _ctx_for("Define blockchain", "!!!")
    # deliberately remove ctx.flags to cover "create it" branch
    if hasattr(ctx, "flags"):
        delattr(ctx, "flags")

    engine = RuleEngine(
        rules=[
            {"id": "quality.nonsense_basic", "params": {"sev": "strong"}},
            {"id": "correctness.definition_sanity"},
        ]
    )
    engine.run(ctx)

    # engine should normalize into ctx.flags: list[str]
    assert isinstance(ctx.flags, list)
    bases = {f.split(":", 1)[0] for f in ctx.flags}
    assert "quality.nonsense_basic" in bases
    assert "correctness.definition_sanity" in bases


def test_rule_engine_unknown_rule_id_does_not_crash() -> None:
    ctx = _ctx_for("hello", "world")
    engine = RuleEngine(rules=[{"id": "does_not_exist"}])
    engine.run(ctx)
    assert ctx.flags == []


def test_rule_engine_rule_exception_does_not_crash() -> None:
    ctx = _ctx_for("hello", "world")
    engine = RuleEngine(rules=[{"id": "boom_rule"}, {"id": "quality.nonsense_basic"}])
    engine.run(ctx)
    bases = {f.split(":", 1)[0] for f in ctx.flags}
    assert "quality.nonsense_basic" in bases


def test_load_plan_for_rubric_filters_disabled_rules_and_parses_params(
    tmp_path, monkeypatch
) -> None:
    # Use cwd override because load_plan_for_rubric uses Path("configs")/...
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "rules").mkdir(parents=True)

    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
        """
rubric_id: chat_quality
version: v1
rules:
  - id: quality.nonsense_basic
    enabled: true
    params:
      sev: strong
  - id: correctness.definition_sanity
    enabled: false
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan_for_rubric("chat_quality", "v1")
    assert plan.rubric_id == "chat_quality"
    assert plan.version == "v1"
    assert len(plan.rules) == 1
    assert plan.rules[0]["id"] == "quality.nonsense_basic"
    assert plan.rules[0]["params"]["sev"] == "strong"


def test_load_plan_for_rubric_fallback_yaml_parser(tmp_path, monkeypatch) -> None:
    """
    Force the internal minimal YAML parser branch even if PyYAML is installed.
    """
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "rules").mkdir(parents=True)

    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
        """
rubric_id: chat_quality
version: v1
rules:
  - id: quality.nonsense_basic
    enabled: true
    params:
      sev: strong
""".lstrip(),
        encoding="utf-8",
    )

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any):
        if name == "yaml":
            raise ImportError("force fallback parser")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    plan = load_plan_for_rubric("chat_quality", "v1")
    assert plan.rules[0]["id"] == "quality.nonsense_basic"
    assert plan.rules[0]["params"]["sev"] == "strong"


def test_run_rules_returns_result_with_flags(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs" / "rules").mkdir(parents=True)

    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
        """
rubric_id: chat_quality
version: v1
rules:
  - id: quality.nonsense_basic
    enabled: true
    params:
      sev: strong
""".lstrip(),
        encoding="utf-8",
    )

    plan = load_plan_for_rubric("chat_quality", "v1")
    ctx = _ctx_for("Define blockchain", "!!!")

    rr = run_rules(ctx, plan)
    assert hasattr(rr, "flags")
    ids = {getattr(f, "id", "") for f in rr.flags}
    # because engine prefixes short ids when invoked as quality.xxx
    assert "quality.nonsense_basic" in ids
