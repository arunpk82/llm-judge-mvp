"""
Per-builder isolation tests for ``build_all_graphs`` (#179).

Asserts that when one of the five graph builders raises, the remaining
four still succeed and a WARNING is logged with the specific graph name
and exception type. Pre-fix, a single bad pass took the entire L2
ensemble down silently (see ``docs/sessions/session_continuity_brief_2026-04-21.md``).
"""

from __future__ import annotations

import logging

import pytest

import llm_judge.calibration.hallucination_graphs as hg


def _make_fact_tables() -> dict:
    """Minimum non-empty inputs for every builder."""
    return {
        "passes": {
            "P1_entities": {"entities": [{"name": "Alice", "type": "person"}]},
            "P2_events": {
                "events": [{"who": "Alice", "action": "visited", "target": "Paris"}]
            },
            "P3_relationships": {
                "relationships": [
                    {"entity1": "Alice", "entity2": "Bob", "relationship": "friend"}
                ]
            },
            "P4_numbers": {
                "numerical_facts": [
                    {"entity": "Alice", "number": "3", "describes": "books"}
                ],
                "temporal_facts": [],
            },
            "P5_negations": {
                "explicit_negations": [{"statement": "Alice is not a cat"}],
                "absent_information": [],
                "corrections": [],
            },
        }
    }


class TestBuildAllGraphsIsolation:
    """One builder raises → others still run + WARNING names the bad graph."""

    @pytest.mark.parametrize(
        "target",
        ["build_g1_entities", "build_g2_events", "build_g3_relationships",
         "build_g4_numbers", "build_g5_negations"],
    )
    def test_single_builder_failure_isolated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        target: str,
    ) -> None:
        expected_graph = {
            "build_g1_entities": "G1",
            "build_g2_events": "G2",
            "build_g3_relationships": "G3",
            "build_g4_numbers": "G4",
            "build_g5_negations": "G5",
        }[target]

        def _boom(*args, **kwargs):
            raise RuntimeError(f"simulated {target} failure")

        monkeypatch.setattr(hg, target, _boom)

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.hallucination_graphs"):
            graphs = hg.build_all_graphs(_make_fact_tables())

        # All five graphs still present; the bad one is empty.
        assert set(graphs.keys()) == {"G1", "G2", "G3", "G4", "G5"}
        assert graphs[expected_graph].number_of_nodes() == 0, (
            f"{expected_graph} should be empty (builder raised), got "
            f"{graphs[expected_graph].number_of_nodes()} nodes"
        )

        # Others still built their graphs (non-empty).
        for name, G in graphs.items():
            if name == expected_graph:
                continue
            # G2 can be empty if its input was empty; here we supplied data for all
            assert G.number_of_nodes() > 0, (
                f"{name} should be non-empty after {expected_graph} failed, "
                f"got 0 nodes — failure was not isolated"
            )

        # WARNING was logged with the specific graph name + exception type.
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        graph_field = [getattr(r, "graph", None) for r in warns]
        err_field = [getattr(r, "error_type", None) for r in warns]
        assert expected_graph in graph_field, (
            f"WARNING must name graph={expected_graph}; got {graph_field}"
        )
        assert "RuntimeError" in err_field, (
            f"WARNING must include error_type=RuntimeError; got {err_field}"
        )

    def test_no_failure_no_warnings(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Clean path: when all builders succeed, no WARNING fires."""
        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.hallucination_graphs"):
            graphs = hg.build_all_graphs(_make_fact_tables())

        assert set(graphs.keys()) == {"G1", "G2", "G3", "G4", "G5"}
        assert all(G.number_of_nodes() > 0 for G in graphs.values())

        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warns, (
            f"no WARNINGs expected on clean path; got {[r.getMessage() for r in warns]}"
        )


class TestP5StringsAcceptedPostNormalization:
    """Regression guard: G5 no longer raises on bare-string P5 elements.

    The *cache* normalizes shape, so at the build layer the data is dict.
    But if a caller somehow passes un-normalized P5, the per-builder
    isolation still keeps the other four alive.
    """

    def test_raw_str_p5_isolated_not_fatal(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        ft = _make_fact_tables()
        ft["passes"]["P5_negations"]["explicit_negations"] = [
            "did not survive to March 1945",  # bare str — bypasses normalizer
        ]

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.hallucination_graphs"):
            graphs = hg.build_all_graphs(ft)

        # G5 emptied (builder raised on .get() against str), but G1-G4 survive.
        assert graphs["G5"].number_of_nodes() == 0
        for name in ("G1", "G2", "G3", "G4"):
            assert graphs[name].number_of_nodes() > 0

        # WARNING names G5 + AttributeError.
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            getattr(r, "graph", None) == "G5"
            and getattr(r, "error_type", None) == "AttributeError"
            for r in warns
        ), f"expected G5/AttributeError WARNING; got {[(getattr(r,'graph',None), getattr(r,'error_type',None)) for r in warns]}"
