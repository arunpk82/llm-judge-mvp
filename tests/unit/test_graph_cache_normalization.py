"""
Silence-contract tests for graph_cache normalization boundary (#179).

Asserts that ``GraphCache.get()`` / ``get_by_hash()`` either return a
fully normalized fact-table (P5 str elements coerced to canonical dicts)
or surface the problem — never silently returns malformed data that
would later crash ``build_g5_negations``.

Companion: ``tests/unit/test_p5_normalization.py`` (unit-level schema
assertions); ``tests/unit/test_build_all_graphs_isolation.py``
(per-builder failure isolation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from llm_judge.calibration.graph_cache import (
    GraphCache,
    compute_source_hash,
)

# =====================================================================
# Fixtures — malformed fact tables the old G5 would crash on
# =====================================================================


def _write_entry(cache: GraphCache, source_text: str, data: dict) -> Path:
    source_hash = compute_source_hash(source_text)
    path = cache.cache_dir / f"{source_hash}.json"
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


_MIXED_P5 = {
    "passes": {
        "P1_entities": {"entities": []},
        "P2_events": {"events": []},
        "P3_relationships": {"relationships": []},
        "P4_numbers": {"numerical_facts": [], "temporal_facts": []},
        "P5_negations": {
            # Mix: strings (drift) and dicts (clean)
            "explicit_negations": [
                "did not survive to March 1945",
                {"statement": "no electricity in the camp"},
            ],
            "absent_information": [
                "The exact dates of death remain unclear.",
                {"what": "Method of execution"},
            ],
            "corrections": [
                "A said X but source says Y.",
                {"wrong": "previously thought", "right": "in fact"},
            ],
        },
    }
}


# =====================================================================
# Silence contract: get() must return normalized data, never raw drift
# =====================================================================


class TestGetNeverReturnsStrElements:
    """Post-get(), no P5 list contains a bare string. Drift is invisible downstream."""

    def test_get_normalizes_explicit_negations(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write_entry(cache, "src-neg", _MIXED_P5)

        result = cache.get("src-neg")

        assert result is not None
        p5 = result["passes"]["P5_negations"]
        for elem in p5["explicit_negations"]:
            assert isinstance(elem, dict), f"bare string leaked past get(): {elem!r}"
            assert "statement" in elem

    def test_get_normalizes_absent_information(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write_entry(cache, "src-absent", _MIXED_P5)

        result = cache.get("src-absent")

        assert result is not None
        for elem in result["passes"]["P5_negations"]["absent_information"]:
            assert isinstance(elem, dict)
            assert "what" in elem

    def test_get_normalizes_corrections(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        _write_entry(cache, "src-corr", _MIXED_P5)

        result = cache.get("src-corr")

        assert result is not None
        for elem in result["passes"]["P5_negations"]["corrections"]:
            assert isinstance(elem, dict)
            assert "wrong" in elem
            assert "right" in elem

    def test_get_by_hash_also_normalizes(self, tmp_path: Path) -> None:
        """The second public read path must honor the same contract."""
        cache = GraphCache(tmp_path)
        src = "by-hash-source"
        source_hash = compute_source_hash(src)
        _write_entry(cache, src, _MIXED_P5)

        result = cache.get_by_hash(source_hash)

        assert result is not None
        for elem in result["passes"]["P5_negations"]["explicit_negations"]:
            assert isinstance(elem, dict)


# =====================================================================
# Clean pass-through: already-dict P5 entries are unchanged
# =====================================================================


class TestGetPassesThroughCleanData:
    """When P5 is already dicts, get() must not mutate shape."""

    def test_dicts_only_unchanged(self, tmp_path: Path) -> None:
        cache = GraphCache(tmp_path)
        clean = {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [{"statement": "no X"}],
                    "absent_information": [{"what": "no info"}],
                    "corrections": [{"wrong": "A", "right": "B"}],
                }
            }
        }
        _write_entry(cache, "clean", clean)

        result = cache.get("clean")

        assert result is not None
        p5 = result["passes"]["P5_negations"]
        assert p5["explicit_negations"] == [{"statement": "no X"}]
        assert p5["absent_information"] == [{"what": "no info"}]
        assert p5["corrections"] == [{"wrong": "A", "right": "B"}]


# =====================================================================
# Unknown-type contract: WARNING, never silent
# =====================================================================


class TestUnknownElementTypeSurfacesWarning:
    """Drift beyond str/dict (e.g., int, list) must not be silently dropped."""

    def test_int_element_logs_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = GraphCache(tmp_path)
        weird = {
            "passes": {
                "P5_negations": {
                    "explicit_negations": [42, "a valid str neg"],
                    "absent_information": [],
                    "corrections": [],
                }
            }
        }
        _write_entry(cache, "weird", weird)

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.graph_cache"):
            result = cache.get("weird")

        assert result is not None
        p5 = result["passes"]["P5_negations"]
        # Int was dropped; valid string was normalized.
        assert len(p5["explicit_negations"]) == 1
        assert p5["explicit_negations"][0] == {"statement": "a valid str neg"}
        # But the drop was loud — WARNING recorded with type + field.
        warns = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert any(
            "normalize_unknown_element" in r.getMessage()
            or getattr(r, "type", None) == "int"
            for r in warns
        ), f"expected WARNING for int element; got {[r.getMessage() for r in warns]}"


# =====================================================================
# Malformed JSON path remains loud (pre-existing contract, not regressed)
# =====================================================================


class TestCorruptEntryStillWarns:
    """Normalization must not swallow the existing corrupt-file WARNING."""

    def test_corrupt_json_logs_read_error(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        cache = GraphCache(tmp_path)
        cache.cache_dir.mkdir(parents=True, exist_ok=True)
        source_hash = compute_source_hash("corrupt-source")
        (cache.cache_dir / f"{source_hash}.json").write_text("{not json")

        with caplog.at_level(logging.WARNING, logger="llm_judge.calibration.graph_cache"):
            result = cache.get("corrupt-source")

        assert result is None
        assert any("read_error" in r.getMessage() for r in caplog.records)
