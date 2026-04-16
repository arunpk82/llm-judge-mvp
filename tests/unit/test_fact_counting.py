"""
Tests for fact_counting.py — ADR-0027 implementation.

Coverage:
  - FactCountResult dataclass and to_evidence_dict
  - _parse_json_safe robustness (clean JSON, markdown fence, thinking tokens, garbage)
  - check_fact_counting ratio computation and auto-clear decision
  - API error graceful degradation
  - Config-driven L3 method switch in check_hallucination
  - Legacy minicheck_deberta path still works behind flag
"""

from __future__ import annotations

import json
from unittest.mock import patch

from llm_judge.calibration.fact_counting import (
    FACT_COUNTING_PROMPT,
    FactCountResult,
    _parse_json_safe,
    check_fact_counting,
)

# =====================================================================
# FactCountResult
# =====================================================================


class TestFactCountResult:

    def test_defaults(self) -> None:
        r = FactCountResult()
        assert r.ratio == 0.0
        assert r.auto_clear is False
        assert r.error == ""
        assert r.facts == []

    def test_to_evidence_dict_flat(self) -> None:
        r = FactCountResult(
            supported=3, not_found=1, total=4, ratio=0.75,
            auto_clear=False, verdict="HALLUCINATED", model="gemini-2.5-flash",
        )
        d = r.to_evidence_dict()
        assert d["fc_supported"] == 3
        assert d["fc_ratio"] == 0.75
        assert d["fc_auto_clear"] is False
        # No nested values
        for k, v in d.items():
            assert not isinstance(v, (dict, list)), f"{k} is nested"


# =====================================================================
# JSON parser
# =====================================================================


class TestParseJsonSafe:

    def test_clean_json(self) -> None:
        raw = '{"supported": 3, "total": 4, "verdict": "GROUNDED"}'
        result = _parse_json_safe(raw)
        assert result["supported"] == 3
        assert result["verdict"] == "GROUNDED"

    def test_markdown_fence(self) -> None:
        raw = 'Here is the result:\n```json\n{"supported": 2, "total": 3}\n```\n'
        result = _parse_json_safe(raw)
        assert result["supported"] == 2

    def test_thinking_tokens_stripped(self) -> None:
        raw = '<|channel>thought\nLet me think...<channel|>{"supported": 1, "total": 1}'
        result = _parse_json_safe(raw)
        assert result["supported"] == 1

    def test_garbage_returns_parse_error(self) -> None:
        result = _parse_json_safe("This is not JSON at all.")
        assert result.get("_parse_error") is True

    def test_embedded_json_in_prose(self) -> None:
        raw = 'The analysis shows: {"supported": 5, "total": 6} which means grounded.'
        result = _parse_json_safe(raw)
        assert result["supported"] == 5


# =====================================================================
# Prompt
# =====================================================================


class TestPrompt:

    def test_prompt_has_placeholders(self) -> None:
        assert "{source}" in FACT_COUNTING_PROMPT
        assert "{claim}" in FACT_COUNTING_PROMPT

    def test_prompt_mentions_all_five_statuses(self) -> None:
        for status in ["SUPPORTED", "NOT_FOUND", "CONTRADICTED", "SHIFTED", "INFERRED"]:
            assert status in FACT_COUNTING_PROMPT


# =====================================================================
# check_fact_counting — with mocked LLM
# =====================================================================


def _mock_fc_response(supported: int, total: int, **extra: int) -> str:
    """Build a mock LLM response for fact-counting."""
    not_found = extra.get("not_found", 0)
    contradicted = extra.get("contradicted", 0)
    shifted = extra.get("shifted", 0)
    inferred = extra.get("inferred", 0)
    verdict = "GROUNDED" if supported == total else "HALLUCINATED"
    return json.dumps({
        "facts": [{"fact": f"fact_{i}", "status": "SUPPORTED", "source_ref": "S1"} for i in range(supported)],
        "supported": supported,
        "not_found": not_found,
        "contradicted": contradicted,
        "shifted": shifted,
        "inferred": inferred,
        "total": total,
        "verdict": verdict,
    })


class TestCheckFactCounting:

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_auto_clear_above_threshold(self, mock_llm: object) -> None:
        mock_llm.return_value = _mock_fc_response(5, 5)  # type: ignore[attr-defined]
        result = check_fact_counting("The sky is blue.", "Source text.", threshold=0.80)
        assert result.ratio == 1.0
        assert result.auto_clear is True
        assert result.verdict == "GROUNDED"
        assert result.error == ""

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_below_threshold_not_cleared(self, mock_llm: object) -> None:
        mock_llm.return_value = _mock_fc_response(2, 4, not_found=2)  # type: ignore[attr-defined]
        result = check_fact_counting("Some claim.", "Source text.", threshold=0.80)
        assert result.ratio == 0.5
        assert result.auto_clear is False
        assert result.supported == 2
        assert result.not_found == 2

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_exactly_at_threshold(self, mock_llm: object) -> None:
        mock_llm.return_value = _mock_fc_response(4, 5, inferred=1)  # type: ignore[attr-defined]
        result = check_fact_counting("Claim.", "Source.", threshold=0.80)
        assert result.ratio == 0.8
        assert result.auto_clear is True  # >= threshold

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_contradicted_facts_flagged(self, mock_llm: object) -> None:
        mock_llm.return_value = _mock_fc_response(2, 4, contradicted=2)  # type: ignore[attr-defined]
        result = check_fact_counting("Claim.", "Source.", threshold=0.80)
        assert result.contradicted == 2
        assert result.auto_clear is False

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_zero_total_gives_zero_ratio(self, mock_llm: object) -> None:
        mock_llm.return_value = json.dumps({"facts": [], "supported": 0, "total": 0, "verdict": "GROUNDED"})  # type: ignore[attr-defined]
        result = check_fact_counting(".", "Source.")
        assert result.ratio == 0.0
        assert result.auto_clear is False

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_recomputes_total_when_llm_wrong(self, mock_llm: object) -> None:
        # LLM says total=10 but actual counts sum to 4
        mock_llm.return_value = json.dumps({  # type: ignore[attr-defined]
            "facts": [], "supported": 3, "not_found": 1,
            "contradicted": 0, "shifted": 0, "inferred": 0,
            "total": 10, "verdict": "GROUNDED",
        })
        result = check_fact_counting("Claim.", "Source.")
        assert result.total == 4  # recomputed from actual counts
        assert result.ratio == 0.75

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_api_error_returns_error_result(self, mock_llm: object) -> None:
        mock_llm.side_effect = RuntimeError("API timeout")  # type: ignore[attr-defined]
        result = check_fact_counting("Claim.", "Source.")
        assert result.error != ""
        assert "timeout" in result.error.lower()
        assert result.ratio == 0.0
        assert result.auto_clear is False

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_json_parse_error_returns_error_result(self, mock_llm: object) -> None:
        mock_llm.return_value = "Not JSON at all"  # type: ignore[attr-defined]
        result = check_fact_counting("Claim.", "Source.")
        assert result.error != ""
        assert "json_parse_error" in result.error

    @patch("llm_judge.calibration.fact_counting._call_llm")
    def test_evidence_dict_populated_on_success(self, mock_llm: object) -> None:
        mock_llm.return_value = _mock_fc_response(3, 4, not_found=1)  # type: ignore[attr-defined]
        result = check_fact_counting("Claim.", "Source.")
        d = result.to_evidence_dict()
        assert d["fc_supported"] == 3
        assert d["fc_not_found"] == 1
        assert d["fc_total"] == 4
        assert d["fc_ratio"] == 0.75


# =====================================================================
# Config-driven L3 method switch in check_hallucination
# =====================================================================


class TestL3MethodSwitch:
    """Verify the config switch routes to the correct L3 path."""

    # Response text that L1 substring match CANNOT resolve (paraphrased, not exact)
    _RESPONSE = "The defendant entered a plea of no contest. The case was settled out of court."
    _CONTEXT = "West pleaded no contest to the charges. A settlement was reached between the parties."

    def test_fact_counting_path_used_when_configured(self) -> None:
        """When config.l3_method='fact_counting', check_hallucination uses fact-counting."""
        from llm_judge.calibration.hallucination import check_hallucination
        from llm_judge.calibration.pipeline_config import (
            LayerConfig,
            PipelineConfig,
            ThresholdConfig,
        )

        cfg = PipelineConfig(
            layers=LayerConfig(l1_enabled=False, l2_enabled=False, l3_enabled=True, l4_enabled=False),
            l3_method="fact_counting",
            thresholds=ThresholdConfig(fact_counting_clear=0.80),
        )

        # Mock the fact-counting call to return auto-clear
        with patch("llm_judge.calibration.fact_counting.check_fact_counting") as mock_fc:
            mock_fc.return_value = FactCountResult(
                supported=5, total=5, ratio=1.0, auto_clear=True,
                verdict="GROUNDED", model="test",
            )
            result = check_hallucination(
                response=self._RESPONSE,
                context=self._CONTEXT,
                gate2_routing="pass",
                skip_embeddings=True,
                config=cfg,
            )
            assert mock_fc.called
            assert result.layer_stats.get("L3_fact_counting_clear", 0) > 0

    def test_minicheck_path_used_when_configured(self) -> None:
        """When config.l3_method='minicheck_deberta', the legacy path is used."""
        from llm_judge.calibration.hallucination import check_hallucination
        from llm_judge.calibration.pipeline_config import (
            LayerConfig,
            PipelineConfig,
        )

        cfg = PipelineConfig(
            layers=LayerConfig(l1_enabled=False, l2_enabled=False, l3_enabled=True, l4_enabled=False),
            l3_method="minicheck_deberta",
        )

        with patch("llm_judge.calibration.hallucination._l3_minicheck") as mock_mc:
            mock_mc.return_value = True
            result = check_hallucination(
                response=self._RESPONSE,
                context=self._CONTEXT,
                gate2_routing="pass",
                skip_embeddings=True,
                config=cfg,
            )
            assert mock_mc.called
            assert result.layer_stats.get("L3_minicheck", 0) > 0

    def test_no_config_uses_default_minicheck(self) -> None:
        """Without config, the legacy minicheck_deberta path is used (backward compat)."""
        from llm_judge.calibration.hallucination import check_hallucination

        with patch("llm_judge.calibration.hallucination._l3_minicheck") as mock_mc:
            mock_mc.return_value = True
            check_hallucination(
                response=self._RESPONSE,
                context=self._CONTEXT,
                gate2_routing="pass",
                skip_embeddings=True,
                l1_enabled=False,
            )
            # Without config, l3_method defaults to minicheck_deberta
            assert mock_mc.called
