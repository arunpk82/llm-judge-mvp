"""
Tests for Pipeline Funnel Report & Diagnostics (funnel_report.py).

Tests cover:
  Dataclasses: LayerStats, CumulativeStats, SentenceResult, FunnelReport
  FunnelBuilder: add_layer, add_sentence, safety violation detection, catch tracking
  FunnelBuilder.build: cumulative stats, missed hallucinations, flag analysis
  print_funnel: screen output format
  save_diagnostics: JSON output structure
"""
from __future__ import annotations

import json
import os
import tempfile

# =====================================================================
# Dataclass Construction
# =====================================================================

class TestLayerStats:
    def test_defaults(self) -> None:
        from llm_judge.calibration.funnel_report import LayerStats

        stats = LayerStats(name="L1", enabled=True)
        assert stats.name == "L1"
        assert stats.enabled is True
        assert stats.input_count == 0
        assert stats.grounded == 0
        assert stats.flagged == 0
        assert stats.unknown == 0
        assert stats.tp == 0
        assert stats.fp == 0

    def test_populated(self) -> None:
        from llm_judge.calibration.funnel_report import LayerStats

        stats = LayerStats(
            name="L2", enabled=True, input_count=262,
            grounded=57, flagged=89, unknown=116, tp=9, tn=57, fp=0, fn=2,
        )
        assert stats.grounded + stats.flagged + stats.unknown == 262


class TestSentenceResult:
    def test_defaults(self) -> None:
        from llm_judge.calibration.funnel_report import SentenceResult

        sr = SentenceResult(case_id="test_1", sentence_idx=0, sentence="Hello world.")
        assert sr.verdict == "unknown"
        assert sr.gt_label == ""

    def test_populated(self) -> None:
        from llm_judge.calibration.funnel_report import SentenceResult

        sr = SentenceResult(
            case_id="ragtruth_28", sentence_idx=0, sentence="Three women...",
            gt_label="hallucinated", gt_type="Evident Conflict",
            verdict="flagged", confidence="medium", resolved_by="L2_flagged",
        )
        assert sr.gt_label == "hallucinated"
        assert sr.resolved_by == "L2_flagged"


class TestFunnelReport:
    def test_defaults(self) -> None:
        from llm_judge.calibration.funnel_report import FunnelReport

        report = FunnelReport()
        assert report.total_sentences == 0
        assert report.safety_violations == []
        assert report.layers == []


# =====================================================================
# FunnelBuilder — Core Logic
# =====================================================================

def _make_builder():
    """Create a FunnelBuilder with typical L1+L2 data."""
    from llm_judge.calibration.funnel_report import (
        FunnelBuilder,
        LayerStats,
        SentenceResult,
    )

    builder = FunnelBuilder(total_hallucinated=4)

    # L1 layer stats
    builder.add_layer(LayerStats(
        name="L1", enabled=True, input_count=20,
        grounded=5, flagged=3, unknown=12, tp=1, tn=5, fp=0, fn=0,
    ))

    # L2 layer stats
    builder.add_layer(LayerStats(
        name="L2", enabled=True, input_count=15,
        grounded=4, flagged=6, unknown=5, tp=2, tn=4, fp=0, fn=1,
    ))

    # Add clean grounded sentence (L1)
    builder.add_sentence(SentenceResult(
        case_id="case_1", sentence_idx=0, sentence="Paris is the capital of France.",
        gt_label="clean", verdict="grounded", resolved_by="L1",
    ))

    # Add hallucinated flagged sentence (L2) — should be a catch
    builder.add_sentence(SentenceResult(
        case_id="case_2", sentence_idx=1, sentence="Three women were injured.",
        gt_label="hallucinated", gt_type="Evident Conflict",
        verdict="flagged", resolved_by="L2_flagged",
        evidence=["Entity 'women' NOT in graph"],
    ))

    # Add clean flagged sentence (L2) — false positive
    builder.add_sentence(SentenceResult(
        case_id="case_3", sentence_idx=0, sentence="The event occurred downtown.",
        gt_label="clean", verdict="flagged", resolved_by="L2_flagged",
    ))

    # Add hallucinated unknown sentence — should be missed
    builder.add_sentence(SentenceResult(
        case_id="case_4", sentence_idx=2, sentence="Additionally the fact that...",
        gt_label="hallucinated", gt_type="Subtle Baseless Info",
        verdict="unknown", resolved_by="unresolved",
    ))

    return builder


class TestFunnelBuilderAddSentence:
    def test_catch_detected(self) -> None:
        builder = _make_builder()
        assert len(builder.catches) == 1
        assert builder.catches[0]["case_id"] == "case_2"

    def test_no_safety_violations(self) -> None:
        builder = _make_builder()
        assert len(builder.safety_violations) == 0

    def test_safety_violation_detected(self) -> None:
        from llm_judge.calibration.funnel_report import FunnelBuilder, SentenceResult

        builder = FunnelBuilder(total_hallucinated=1)
        builder.add_sentence(SentenceResult(
            case_id="bad", sentence_idx=0, sentence="Fabricated fact.",
            gt_label="hallucinated", verdict="grounded", resolved_by="L1",
        ))
        assert len(builder.safety_violations) == 1
        assert builder.safety_violations[0]["case_id"] == "bad"


class TestFunnelBuilderBuild:
    def test_build_returns_report(self) -> None:
        builder = _make_builder()
        report = builder.build(elapsed_s=2.5, dataset="test_data")
        assert report.total_sentences == 4
        assert report.dataset == "test_data"
        assert report.elapsed_s == 2.5

    def test_cumulative_stats_computed(self) -> None:
        builder = _make_builder()
        report = builder.build()
        assert len(report.cumulative) == 2  # L1 and L2

        after_l1 = report.cumulative[0]
        assert after_l1["total_cleared"] == 5
        assert after_l1["clearance_precision"] == 1.0

        after_l2 = report.cumulative[1]
        assert after_l2["total_cleared"] == 9  # 5 + 4

    def test_missed_hallucinations_found(self) -> None:
        builder = _make_builder()
        report = builder.build()
        assert len(report.missed_hallucinations) == 1
        assert report.missed_hallucinations[0]["case_id"] == "case_4"

    def test_flag_analysis_computed(self) -> None:
        builder = _make_builder()
        report = builder.build()
        fa = report.flag_analysis
        assert fa["total_flags"] == 2
        assert fa["true_hallucinations"] == 1
        assert fa["false_positives"] == 1
        assert fa["flag_precision"] == 0.5

    def test_safety_violations_in_report(self) -> None:
        builder = _make_builder()
        report = builder.build()
        assert len(report.safety_violations) == 0  # none in this builder

    def test_layers_serialized(self) -> None:
        builder = _make_builder()
        report = builder.build()
        assert len(report.layers) == 2
        assert report.layers[0]["name"] == "L1"
        assert report.layers[1]["name"] == "L2"

    def test_config_passed_through(self) -> None:
        builder = _make_builder()
        report = builder.build(config={"l1": True, "l2": True})
        assert report.config["l1"] is True

    def test_timestamp_populated(self) -> None:
        builder = _make_builder()
        report = builder.build()
        assert report.timestamp != ""
        assert "T" in report.timestamp  # ISO format


# =====================================================================
# print_funnel — Screen Output
# =====================================================================

class TestPrintFunnel:
    def test_prints_without_error(self, capsys) -> None:
        builder = _make_builder()
        report = builder.build()

        from llm_judge.calibration.funnel_report import print_funnel

        print_funnel(report)
        captured = capsys.readouterr()
        assert "FUNNEL REPORT" in captured.out
        assert "L1" in captured.out
        assert "L2" in captured.out
        assert "PIPELINE SUMMARY" in captured.out

    def test_shows_grounded_count(self, capsys) -> None:
        builder = _make_builder()
        report = builder.build()

        from llm_judge.calibration.funnel_report import print_funnel

        print_funnel(report)
        captured = capsys.readouterr()
        assert "Grounded:" in captured.out

    def test_shows_disabled_layers(self, capsys) -> None:
        from llm_judge.calibration.funnel_report import (
            FunnelBuilder,
            LayerStats,
            print_funnel,
        )

        builder = FunnelBuilder()
        builder.add_layer(LayerStats(name="L1", enabled=True, input_count=10, grounded=10))
        builder.add_layer(LayerStats(name="L2", enabled=False))
        report = builder.build()

        print_funnel(report)
        captured = capsys.readouterr()
        assert "DISABLED" in captured.out


# =====================================================================
# save_diagnostics — JSON Output
# =====================================================================

class TestSaveDiagnostics:
    def test_creates_valid_json(self) -> None:
        builder = _make_builder()
        report = builder.build()

        from llm_judge.calibration.funnel_report import save_diagnostics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_diagnostics(report, path)
            with open(path) as f:
                data = json.load(f)

            assert "pipeline_run" in data
            assert "funnel" in data
            assert "cumulative" in data
            assert "flag_analysis" in data
            assert "safety_violations" in data
            assert "missed_hallucinations" in data
            assert "sentences" in data
        finally:
            os.unlink(path)

    def test_pipeline_run_fields(self) -> None:
        builder = _make_builder()
        report = builder.build(elapsed_s=5.0, dataset="ragtruth_50")

        from llm_judge.calibration.funnel_report import save_diagnostics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_diagnostics(report, path)
            with open(path) as f:
                data = json.load(f)

            run = data["pipeline_run"]
            assert run["dataset"] == "ragtruth_50"
            assert run["elapsed_s"] == 5.0
            assert run["total_sentences"] == 4
        finally:
            os.unlink(path)

    def test_funnel_layer_keys(self) -> None:
        builder = _make_builder()
        report = builder.build()

        from llm_judge.calibration.funnel_report import save_diagnostics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_diagnostics(report, path)
            with open(path) as f:
                data = json.load(f)

            assert "L1" in data["funnel"]
            assert "L2" in data["funnel"]
            assert data["funnel"]["L1"]["grounded"] == 5
        finally:
            os.unlink(path)

    def test_sentences_included(self) -> None:
        builder = _make_builder()
        report = builder.build()

        from llm_judge.calibration.funnel_report import save_diagnostics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            save_diagnostics(report, path)
            with open(path) as f:
                data = json.load(f)

            assert len(data["sentences"]) == 4
            verdicts = [s["verdict"] for s in data["sentences"]]
            assert "grounded" in verdicts
            assert "flagged" in verdicts
            assert "unknown" in verdicts
        finally:
            os.unlink(path)
