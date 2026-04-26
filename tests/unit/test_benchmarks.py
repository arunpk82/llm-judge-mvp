"""Tests for benchmark framework (EPIC 7.16)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_judge.benchmarks import (
    GroundTruth,
    SpanAnnotation,
)
from llm_judge.benchmarks.metrics import (
    ClassificationMetrics,
    compute_metrics,
)
from llm_judge.benchmarks.runner import BenchmarkRunResult, PropertyResult


class TestGroundTruth:
    def test_basic_creation(self):
        gt = GroundTruth(response_level="fail")
        assert gt.response_level == "fail"
        assert gt.property_labels == {}
        assert gt.span_annotations == []

    def test_with_spans(self):
        gt = GroundTruth(
            response_level="fail",
            span_annotations=[
                SpanAnnotation(
                    start=10,
                    end=20,
                    text="Gaza Strip",
                    label_type="Evident Baseless Info",
                ),
            ],
            hallucination_types={"Evident Baseless Info": 1},
        )
        assert len(gt.span_annotations) == 1
        assert gt.hallucination_types["Evident Baseless Info"] == 1


class TestClassificationMetrics:
    def test_perfect_classifier(self):
        m = ClassificationMetrics(tp=50, fp=0, tn=50, fn=0)
        assert m.precision == 1.0
        assert m.recall == 1.0
        assert m.f1 == 1.0
        assert m.accuracy == 1.0

    def test_no_predictions(self):
        m = ClassificationMetrics(tp=0, fp=0, tn=50, fn=50)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0
        assert m.accuracy == 0.5

    def test_all_false_positives(self):
        m = ClassificationMetrics(tp=0, fp=50, tn=0, fn=50)
        assert m.precision == 0.0
        assert m.recall == 0.0
        assert m.f1 == 0.0

    def test_realistic_scores(self):
        m = ClassificationMetrics(tp=194, fp=126, tn=1606, fn=749)
        assert round(m.precision, 3) == 0.606
        assert round(m.recall, 3) == 0.206
        assert round(m.f1, 3) == 0.307

    def test_to_dict(self):
        m = ClassificationMetrics(tp=10, fp=5, tn=80, fn=5)
        d = m.to_dict()
        assert d["tp"] == 10
        assert d["total"] == 100
        assert "precision" in d
        assert "f1" in d


class TestComputeMetrics:
    def test_from_run_result(self):
        result = BenchmarkRunResult(
            benchmark_name="test",
            split="test",
            cases_evaluated=4,
            elapsed_seconds=1.0,
            response_level_results=[
                {
                    "predicted": "fail",
                    "expected": "fail",
                    "match": True,
                    "model": "m1",
                    "task_type": "QA",
                },
                {
                    "predicted": "pass",
                    "expected": "fail",
                    "match": False,
                    "model": "m1",
                    "task_type": "QA",
                },
                {
                    "predicted": "pass",
                    "expected": "pass",
                    "match": True,
                    "model": "m1",
                    "task_type": "Summary",
                },
                {
                    "predicted": "fail",
                    "expected": "pass",
                    "match": False,
                    "model": "m1",
                    "task_type": "Summary",
                },
            ],
            property_results={
                "1.1": [
                    PropertyResult(
                        case_id="c1",
                        property_id="1.1",
                        predicted="fail",
                        expected="fail",
                        match=True,
                    ),
                    PropertyResult(
                        case_id="c2",
                        property_id="1.1",
                        predicted="pass",
                        expected="fail",
                        match=False,
                    ),
                ],
            },
        )
        metrics = compute_metrics(result)
        assert metrics.cases_evaluated == 4
        assert metrics.response_level.tp == 1
        assert metrics.response_level.fn == 1
        assert "1.1" in metrics.per_property
        assert metrics.per_property["1.1"].tp == 1


class TestRAGTruthAdapter:
    def _create_test_data(self, tmpdir: Path) -> Path:
        """Create minimal RAGTruth-format test data."""
        source = {
            "source_id": "s1",
            "task_type": "QA",
            "source": "test",
            "source_info": {"question": "What is X?", "passages": "X is a thing."},
            "prompt": "Answer: What is X?",
        }
        response_clean = {
            "id": "r1",
            "source_id": "s1",
            "model": "test-model",
            "temperature": 0.0,
            "labels": [],
            "split": "test",
            "quality": "good",
            "response": "X is a thing that does stuff.",
        }
        response_hallucinated = {
            "id": "r2",
            "source_id": "s1",
            "model": "test-model",
            "temperature": 0.0,
            "split": "test",
            "quality": "good",
            "labels": [
                {
                    "start": 0,
                    "end": 10,
                    "text": "Y is fake",
                    "label_type": "Evident Baseless Info",
                    "meta": "fabricated",
                    "implicit_true": False,
                    "due_to_null": False,
                }
            ],
            "response": "Y is fake information not in the source.",
        }

        with (tmpdir / "source_info.jsonl").open("w") as f:
            f.write(json.dumps(source) + "\n")
        with (tmpdir / "response.jsonl").open("w") as f:
            f.write(json.dumps(response_clean) + "\n")
            f.write(json.dumps(response_hallucinated) + "\n")

        return tmpdir

    def test_loads_cases(self, tmp_path):
        data_dir = self._create_test_data(tmp_path)
        from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

        adapter = RAGTruthAdapter(data_dir=data_dir)

        cases = list(adapter.load_cases(split="test"))
        assert len(cases) == 2

        # First case: clean (no hallucination)
        assert cases[0].ground_truth.response_level == "pass"
        assert cases[0].ground_truth.property_labels["1.1"] == "pass"

        # Second case: hallucinated
        assert cases[1].ground_truth.response_level == "fail"
        assert cases[1].ground_truth.property_labels["1.1"] == "fail"
        assert cases[1].ground_truth.property_labels["1.5"] == "fail"
        assert len(cases[1].ground_truth.span_annotations) == 1

    def test_metadata(self):
        from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

        adapter = RAGTruthAdapter()
        meta = adapter.metadata()
        assert meta.name == "RAGTruth"
        assert "1.1" in meta.supported_properties
        assert len(meta.published_baselines) > 0

    def test_max_cases(self, tmp_path):
        data_dir = self._create_test_data(tmp_path)
        from llm_judge.benchmarks.ragtruth import RAGTruthAdapter

        adapter = RAGTruthAdapter(data_dir=data_dir)

        cases = list(adapter.load_cases(split="test", max_cases=1))
        assert len(cases) == 1


class TestRunRagtruth50Serializer:
    """Regression guard: run_ragtruth50.py cmd_benchmark must serialize all
    fields of BenchmarkRunResult that downstream consumers depend on.
    Added after #167: sentence_level_metrics was computed but silently
    dropped by the hand-built summary dict."""

    def test_sentence_level_metrics_in_output(self, tmp_path, monkeypatch):
        import argparse
        import importlib.util
        from pathlib import Path as _Path

        repo_root = _Path(__file__).resolve().parents[2]
        module_path = repo_root / "tools" / "run_ragtruth50.py"
        spec = importlib.util.spec_from_file_location("run_ragtruth50", module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        stub_metrics = {
            "overall": {"precision": 0.91, "recall": 0.42, "f1": 0.57},
            "by_layer": {"l1": {"tp": 10, "fp": 1, "fn": 14}},
        }
        stub_result = BenchmarkRunResult(
            benchmark_name="RAGTruth",
            split="test",
            cases_evaluated=1,
            elapsed_seconds=0.1,
            response_level_results=[],
            sentence_level_metrics=stub_metrics,
        )

        class _StubAdapter:
            def set_benchmark_filter(self, *_args, **_kwargs):
                pass

        monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
        from llm_judge.benchmarks import registry as _registry
        monkeypatch.setitem(
            _registry._REGISTRY, "ragtruth_50", lambda: _StubAdapter()
        )
        monkeypatch.setattr(
            "llm_judge.benchmarks.runner.run_benchmark",
            lambda *_a, **_kw: stub_result,
        )
        monkeypatch.setenv("HALLUCINATION_LAYERS", "l1")
        monkeypatch.setenv("GEMINI_API_KEY", "stub")
        monkeypatch.chdir(tmp_path)  # datasets/benchmarks/... won't exist → skip prereq

        args = argparse.Namespace(
            max_cases=1, gate2_routing="pass", with_llm=False
        )
        assert module.cmd_benchmark(args) is True

        written = json.loads(
            (tmp_path / "results" / "ragtruth50_results.json").read_text()
        )
        assert "sentence_level_metrics" in written
        assert written["sentence_level_metrics"] == stub_metrics
