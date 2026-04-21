"""
Unit tests for the P5 fact-table normalizer (#179).

Three core assertions as requested:
  1. Mixed shape handling — str and dict elements coexist; both end as dicts.
  2. Corrections str case — str → {"wrong": s, "right": ""}; dropped by guard downstream.
  3. Unknown type handling — neither str nor dict → WARNING, not silent drop.

The normalizer is private; we exercise it via its public entry points
``GraphCache.get`` / ``get_by_hash`` (i.e., the contract, not the symbol).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from llm_judge.calibration.graph_cache import GraphCache, compute_source_hash
from llm_judge.calibration.hallucination_graphs import build_g5_negations


def _write(cache: GraphCache, src: str, data: dict) -> None:
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    (cache.cache_dir / f"{compute_source_hash(src)}.json").write_text(
        json.dumps(data), encoding="utf-8"
    )


# =====================================================================
# 1. Mixed shape handling
# =====================================================================


class TestMixedShape:
    """str and dict elements in the same list both survive as dicts."""

    def test_mixed_explicit_negations(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write(cache, "mix-neg", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [
                        "no lighting",
                        {"statement": "no water", "detail": "reservoir empty"},
                        "no latrine",
                    ],
                    "absent_information": [],
                    "corrections": [],
                }
            }
        })

        result = cache.get("mix-neg")
        assert result is not None
        negs = result["passes"]["P5_negations"]["explicit_negations"]

        assert len(negs) == 3
        assert negs[0] == {"statement": "no lighting"}
        assert negs[1] == {"statement": "no water", "detail": "reservoir empty"}
        assert negs[2] == {"statement": "no latrine"}

        # And the downstream builder now succeeds on all three.
        G = build_g5_negations(result["passes"]["P5_negations"])
        negation_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "negation"
        ]
        assert len(negation_nodes) == 3

    def test_mixed_absent_information(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write(cache, "mix-abs", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [],
                    "absent_information": [
                        "date unclear",
                        {"what": "method unknown"},
                    ],
                    "corrections": [],
                }
            }
        })

        result = cache.get("mix-abs")
        assert result is not None
        items = result["passes"]["P5_negations"]["absent_information"]

        assert items[0] == {"what": "date unclear"}
        assert items[1] == {"what": "method unknown"}

        G = build_g5_negations(result["passes"]["P5_negations"])
        absent_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "absent"
        ]
        assert len(absent_nodes) == 2


# =====================================================================
# 2. Corrections str case (shape preserved, semantics dropped by guard)
# =====================================================================


class TestCorrectionsStrShape:
    """str correction → {"wrong": s, "right": ""}. Guard in build_g5 drops it."""

    def test_str_correction_shaped_with_empty_right(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write(cache, "str-corr", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [],
                    "absent_information": [],
                    "corrections": [
                        "Anne died a month earlier than previously thought.",
                        {"wrong": "previously thought", "right": "in fact"},
                    ],
                }
            }
        })

        result = cache.get("str-corr")
        assert result is not None
        corrs = result["passes"]["P5_negations"]["corrections"]

        # Shape preserved: str became {"wrong": s, "right": ""}.
        assert corrs[0] == {
            "wrong": "Anne died a month earlier than previously thought.",
            "right": "",
        }
        # Clean dict unchanged.
        assert corrs[1] == {"wrong": "previously thought", "right": "in fact"}

    def test_guard_drops_str_corrections_in_builder(self, tmp_path: Path) -> None:
        """Downstream build_g5_negations' `if wrong and right` guard drops empty-right
        entries. This is the documented, acceptable loss of semantic recovery."""
        cache = GraphCache(tmp_path)
        _write(cache, "drop-check", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [],
                    "absent_information": [],
                    "corrections": [
                        "bare string correction 1",
                        "bare string correction 2",
                        {"wrong": "A", "right": "B"},
                    ],
                }
            }
        })

        result = cache.get("drop-check")
        assert result is not None
        G = build_g5_negations(result["passes"]["P5_negations"])

        correction_nodes = [
            n for n, d in G.nodes(data=True) if d.get("node_type") == "correction"
        ]
        # Only the dict survives the `if wrong and right` guard. The two str
        # cases entered as {"wrong": s, "right": ""} and dropped. No crash.
        assert len(correction_nodes) == 1


# =====================================================================
# 3. Unknown type handling — WARNING, never silent drop
# =====================================================================


class TestUnknownTypeSurfacesWarning:
    """Neither str nor dict must emit a WARNING (not a silent skip)."""

    @pytest.mark.parametrize(
        "weird_element,expected_type",
        [
            (42, "int"),
            (["nested", "list"], "list"),
            (None, "NoneType"),
            (3.14, "float"),
        ],
    )
    def test_unknown_types_log_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        weird_element,
        expected_type: str,
    ) -> None:
        cache = GraphCache(tmp_path)
        _write(cache, f"weird-{expected_type}", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [weird_element, "valid str"],
                    "absent_information": [],
                    "corrections": [],
                }
            }
        })

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.graph_cache"):
            result = cache.get(f"weird-{expected_type}")

        assert result is not None
        negs = result["passes"]["P5_negations"]["explicit_negations"]
        # The weird element was dropped; the str was still normalized.
        assert negs == [{"statement": "valid str"}]

        # And a WARNING was recorded that names the field + type.
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            getattr(r, "type", None) == expected_type
            and "explicit_negations" in str(getattr(r, "field", ""))
            for r in warns
        ), (
            f"expected WARNING with type={expected_type} and field containing "
            f"'explicit_negations'; got {[(getattr(r,'type',None), getattr(r,'field',None)) for r in warns]}"
        )

    def test_known_types_do_not_warn(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Sanity: str and dict inputs trigger no 'unknown_element' WARNING."""
        cache = GraphCache(tmp_path)
        _write(cache, "known-only", {
            "passes": {
                "P5_negations": {
                    "explicit_negations": ["a", {"statement": "b"}],
                    "absent_information": [],
                    "corrections": [],
                }
            }
        })

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.graph_cache"):
            cache.get("known-only")

        unknown_warns = [
            r for r in caplog.records
            if r.levelno == logging.WARNING and "normalize_unknown_element" in r.getMessage()
        ]
        assert not unknown_warns, (
            f"no 'normalize_unknown_element' WARNING expected on clean data; "
            f"got {[r.getMessage() for r in unknown_warns]}"
        )
