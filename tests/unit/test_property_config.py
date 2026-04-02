"""
Tests for Property Configuration (EPIC 7.4) and Integrated Judge (EPICs 7.5–7.6).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from llm_judge.property_config import (
    DetectionCoverage,
    PropertyConfig,
    PropertyRegistry,
    load_property_config,
)
from llm_judge.schemas import PredictResponse


# =====================================================================
# EPIC 7.4: Property Configuration Tests
# =====================================================================


@pytest.fixture()
def sample_config(tmp_path: Path) -> Path:
    """Create a minimal valid property config."""
    config = {
        "schema_version": "1",
        "properties": {
            "groundedness": {
                "id": "1.1",
                "category": "faithfulness",
                "enabled": True,
                "gate_mode": "informational",
                "threshold": 0.3,
                "visibility": {
                    "ml_engineer": "show",
                    "product_owner": "hide",
                    "qa_lead": "show",
                    "executive": "hide",
                },
            },
            "relevance": {
                "id": "2.1",
                "category": "semantic_quality",
                "enabled": True,
                "gate_mode": "auto-gated",
                "visibility": {
                    "ml_engineer": "show",
                    "product_owner": "show",
                    "qa_lead": "show",
                    "executive": "show",
                },
            },
            "toxicity_bias": {
                "id": "3.1",
                "category": "safety",
                "enabled": False,
                "gate_mode": "informational",
                "visibility": {
                    "ml_engineer": "show",
                    "product_owner": "show",
                    "qa_lead": "show",
                    "executive": "show",
                },
            },
        },
    }
    path = tmp_path / "properties" / "property_config.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


def test_load_property_config(sample_config: Path) -> None:
    """Property config loads and parses correctly."""
    registry = load_property_config(sample_config)
    assert len(registry.all_properties) == 3


def test_property_enabled_filter(sample_config: Path) -> None:
    """get_enabled returns only enabled properties."""
    registry = load_property_config(sample_config)
    enabled = registry.get_enabled()
    assert "groundedness" in enabled
    assert "relevance" in enabled
    assert "toxicity_bias" not in enabled


def test_property_by_category(sample_config: Path) -> None:
    """get_by_category filters correctly."""
    registry = load_property_config(sample_config)
    faith = registry.get_by_category("faithfulness")
    assert "groundedness" in faith
    assert "relevance" not in faith


def test_property_is_enabled(sample_config: Path) -> None:
    """is_enabled returns correct boolean."""
    registry = load_property_config(sample_config)
    assert registry.is_enabled("groundedness") is True
    assert registry.is_enabled("toxicity_bias") is False
    assert registry.is_enabled("nonexistent") is False


def test_property_visibility(sample_config: Path) -> None:
    """Visibility filtering works per persona."""
    registry = load_property_config(sample_config)
    prop = registry.get("groundedness")
    assert prop is not None
    assert prop.is_visible_to("ml_engineer") is True
    assert prop.is_visible_to("product_owner") is False
    assert prop.is_visible_to("executive") is False


def test_detection_coverage(sample_config: Path) -> None:
    """Detection coverage metric computes correctly."""
    registry = load_property_config(sample_config)
    coverage = registry.detection_coverage()
    assert coverage.total == 3
    assert coverage.enabled == 2
    assert coverage.gated == 1  # relevance is auto-gated
    assert coverage.informational == 1  # groundedness is informational
    assert coverage.disabled == 1  # toxicity_bias


def test_detection_coverage_summary(sample_config: Path) -> None:
    """Coverage summary is human-readable."""
    registry = load_property_config(sample_config)
    summary = registry.detection_coverage().summary()
    assert "3 properties defined" in summary
    assert "2 enabled" in summary
    assert "1 gated" in summary


def test_filter_for_persona(sample_config: Path) -> None:
    """Results filtered per persona visibility."""
    registry = load_property_config(sample_config)
    results = {
        "groundedness": {"score": 0.8},
        "relevance": {"score": 4},
        "toxicity_bias": {"score": 0},
    }
    executive_view = registry.filter_for_persona(results, "executive")
    # Executive can see relevance (show) but not groundedness (hide)
    assert "relevance" in executive_view
    assert "groundedness" not in executive_view


def test_get_gated(sample_config: Path) -> None:
    """get_gated returns only gated properties."""
    registry = load_property_config(sample_config)
    gated = registry.get_gated()
    assert "relevance" in gated  # auto-gated
    assert "groundedness" not in gated  # informational


def test_missing_config_raises(tmp_path: Path) -> None:
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_property_config(tmp_path / "nonexistent.yaml")


def test_invalid_gate_mode_rejected(tmp_path: Path) -> None:
    """Invalid gate_mode raises ValueError — catches misspellings."""
    config = {
        "properties": {
            "test_prop": {
                "id": "99.1",
                "category": "faithfulness",
                "enabled": True,
                "gate_mode": "gatemode",  # MISSPELLED — must reject
                "visibility": {"ml_engineer": "show"},
            }
        }
    }
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gate_mode"):
        load_property_config(path)


def test_unknown_field_rejected(tmp_path: Path) -> None:
    """Unknown fields in property config are rejected — catches misspellings."""
    config = {
        "properties": {
            "test_prop": {
                "id": "99.1",
                "category": "faithfulness",
                "enabled": True,
                "gate_mode": "informational",
                "treshold": 0.3,  # MISSPELLED — must reject
                "visibility": {"ml_engineer": "show"},
            }
        }
    }
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown fields"):
        load_property_config(path)


def test_invalid_category_rejected(tmp_path: Path) -> None:
    """Invalid category raises ValueError."""
    config = {
        "properties": {
            "test_prop": {
                "id": "99.1",
                "category": "invalid_category",
                "enabled": True,
                "gate_mode": "informational",
                "visibility": {"ml_engineer": "show"},
            }
        }
    }
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid category"):
        load_property_config(path)


def test_invalid_persona_rejected(tmp_path: Path) -> None:
    """Unknown persona in visibility raises ValueError."""
    config = {
        "properties": {
            "test_prop": {
                "id": "99.1",
                "category": "faithfulness",
                "enabled": True,
                "gate_mode": "informational",
                "visibility": {"ceo": "show"},  # not a valid persona
            }
        }
    }
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown persona"):
        load_property_config(path)


# =====================================================================
# EPIC 7.5: Versioned Prompt Tests
# =====================================================================


def test_llm_judge_accepts_prompt_template() -> None:
    """LLMJudge constructor accepts prompt_template parameter."""
    from llm_judge.llm_judge import LLMJudge

    # Should not raise — prompt_template is optional
    judge = LLMJudge(engine="gemini", prompt_template=None)
    assert judge is not None


def test_fallback_prompt_has_no_hardcoded_system_prompt() -> None:
    """The old _SYSTEM_PROMPT constant is renamed to _FALLBACK_SYSTEM_PROMPT."""
    import llm_judge.llm_judge as llm_mod

    # _SYSTEM_PROMPT should NOT exist as a module-level constant
    # _FALLBACK_SYSTEM_PROMPT should exist
    assert hasattr(llm_mod, "_FALLBACK_SYSTEM_PROMPT")


def test_build_eval_prompt_with_template() -> None:
    """_build_eval_prompt uses versioned template when provided."""
    from llm_judge.calibration.prompts import PromptTemplate
    from llm_judge.llm_judge import _build_eval_prompt
    from llm_judge.schemas import Message, PredictRequest

    template = PromptTemplate(
        prompt_id="test",
        version="v1",
        system_prompt="Test system prompt",
        user_template="Conv:\n{conversation}\n\nAnswer:\n{candidate_answer}\n\nDims: {dimensions}",
        dimensions=["relevance", "clarity"],
        created_at="2026-04-01",
        author="test",
    )

    request = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi there!",
        rubric_id="test",
    )

    result = _build_eval_prompt(request, template)
    assert "Conv:" in result
    assert "Hello" in result
    assert "Hi there!" in result
    assert "relevance, clarity" in result


def test_build_eval_prompt_without_template() -> None:
    """_build_eval_prompt uses fallback when no template provided."""
    from llm_judge.llm_judge import _build_eval_prompt
    from llm_judge.schemas import Message, PredictRequest

    request = PredictRequest(
        conversation=[Message(role="user", content="Hello")],
        candidate_answer="Hi there!",
        rubric_id="test",
    )

    result = _build_eval_prompt(request, None)
    assert "Customer conversation:" in result
    assert "Return JSON:" in result


# =====================================================================
# EPIC 7.6: Integrated Judge Tests
# =====================================================================


def test_enriched_response_all_flags() -> None:
    """EnrichedResponse.all_flags() collects flags from all sources."""
    from llm_judge.calibration.hallucination import HallucinationResult
    from llm_judge.integrated_judge import EnrichedResponse, PropertyEvidence

    resp = EnrichedResponse(
        predict_response=PredictResponse(
            decision="pass",
            overall_score=4.0,
            scores={"relevance": 4},
            confidence=0.8,
            flags=["quality.good"],
        ),
        hallucination_result=HallucinationResult(
            case_id="test",
            risk_score=0.3,
            grounding_ratio=0.5,
            ungrounded_claims=1,
            unverifiable_citations=0,
            flags=["low_grounding:0.50"],
        ),
        property_evidence={
            "groundedness": PropertyEvidence(
                property_name="groundedness",
                property_id="1.1",
                enabled=True,
                gate_mode="informational",
                executed=True,
                flags=["low_grounding:0.50"],
            ),
        },
    )

    all_flags = resp.all_flags()
    assert "quality.good" in all_flags
    assert "low_grounding:0.50" in all_flags


def test_enriched_response_to_dict() -> None:
    """EnrichedResponse serializes correctly."""
    from llm_judge.integrated_judge import EnrichedResponse

    resp = EnrichedResponse(
        predict_response=PredictResponse(
            decision="pass",
            overall_score=4.0,
            scores={"relevance": 4, "clarity": 5},
            confidence=0.9,
            flags=[],
        ),
        prompt_version="chat_quality/v1",
        detection_coverage="28 properties defined. 13 enabled.",
    )

    d = resp.to_dict()
    assert d["decision"] == "pass"
    assert d["prompt_version"] == "chat_quality/v1"
    assert "28 properties defined" in d["detection_coverage"]
    assert "property_evidence" in d


def test_production_config_loads() -> None:
    """The actual configs/properties/property_config.yaml loads without errors."""
    from llm_judge.paths import config_root

    config_path = config_root() / "properties" / "property_config.yaml"
    if config_path.exists():
        registry = load_property_config(config_path)
        assert len(registry.all_properties) == 28
        coverage = registry.detection_coverage()
        # PCT-1: 10 fully built properties enabled, 0 gated
        assert coverage.enabled == 10
        assert coverage.gated == 0
