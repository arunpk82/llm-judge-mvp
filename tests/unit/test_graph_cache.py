"""
Tests for graph_cache.py — L2 fact-table cache (ADR-0025).

Covers:
  - Content-addressable hashing
  - Get/put/miss/TTL/immutability
  - Hash-based API (put_by_hash / get_by_hash)
  - Pre-seeding from Exp 31
  - Stats and clear
  - Singleton lifecycle
  - Integration: hallucination.py L2 cache hit/miss paths
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_judge.calibration.graph_cache import (
    FactTableValidationError,
    GraphCache,
    compute_source_hash,
    get_graph_cache,
    preseed_from_exp31,
    reset_graph_cache,
)

# =====================================================================
# Fixtures
# =====================================================================

_SAMPLE_SOURCE = "The capital of France is Paris. Paris has a population of 2.1 million."
_SAMPLE_TABLES = {
    "passes": {
        "P1_entities": {"entities": [{"name": "Paris", "type": "city"}]},
        "P2_events": {"events": []},
        "P3_relationships": {"relationships": []},
        "P4_numbers": {"numbers": [{"value": "2.1 million", "context": "population"}]},
        "P5_negations": {"negations": []},
    }
}


@pytest.fixture()
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / "graph_cache"
    d.mkdir()
    return d


@pytest.fixture()
def cache(cache_dir: Path) -> GraphCache:
    return GraphCache(cache_dir, ttl_hours=168)


# =====================================================================
# compute_source_hash
# =====================================================================


class TestComputeSourceHash:
    def test_deterministic(self) -> None:
        h1 = compute_source_hash(_SAMPLE_SOURCE)
        h2 = compute_source_hash(_SAMPLE_SOURCE)
        assert h1 == h2

    def test_different_texts_different_hashes(self) -> None:
        h1 = compute_source_hash("text A")
        h2 = compute_source_hash("text B")
        assert h1 != h2

    def test_sha256_length(self) -> None:
        h = compute_source_hash("anything")
        assert len(h) == 64  # SHA-256 hex digest


# =====================================================================
# GraphCache — core get/put
# =====================================================================


class TestGraphCacheGetPut:
    def test_miss_returns_none(self, cache: GraphCache) -> None:
        assert cache.get("nonexistent text") is None

    def test_put_then_get(self, cache: GraphCache) -> None:
        cache.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)
        result = cache.get(_SAMPLE_SOURCE)
        assert result is not None
        assert "passes" in result
        assert "P1_entities" in result["passes"]

    def test_creates_directory_on_put(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "nested"
        c = GraphCache(new_dir)
        c.put("some text", _SAMPLE_TABLES)
        assert new_dir.exists()
        assert cache_file_count(new_dir) == 1

    def test_immutability(self, cache: GraphCache) -> None:
        """Existing entries are not overwritten."""
        cache.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)

        modified = {"passes": {"P1_entities": {"entities": [{"name": "MODIFIED"}]}}}
        cache.put(_SAMPLE_SOURCE, modified)

        result = cache.get(_SAMPLE_SOURCE)
        assert result is not None
        # Should still have original data
        assert result["passes"]["P1_entities"]["entities"][0]["name"] == "Paris"

    def test_put_returns_hash(self, cache: GraphCache) -> None:
        h = cache.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)
        assert h == compute_source_hash(_SAMPLE_SOURCE)


# =====================================================================
# Hash-based API
# =====================================================================


class TestHashBasedAPI:
    def test_put_by_hash_then_get_by_hash(self, cache: GraphCache) -> None:
        h = compute_source_hash(_SAMPLE_SOURCE)
        cache.put_by_hash(h, _SAMPLE_TABLES)
        result = cache.get_by_hash(h)
        assert result is not None
        assert "passes" in result

    def test_get_by_hash_miss(self, cache: GraphCache) -> None:
        assert cache.get_by_hash("deadbeef" * 8) is None

    def test_put_by_hash_immutable(self, cache: GraphCache) -> None:
        h = compute_source_hash("text")
        cache.put_by_hash(h, _SAMPLE_TABLES)
        cache.put_by_hash(h, {"passes": {"MODIFIED": True}})
        result = cache.get_by_hash(h)
        assert "P1_entities" in result["passes"]


# =====================================================================
# TTL
# =====================================================================


class TestTTL:
    def test_expired_entry_returns_none(self, cache_dir: Path) -> None:
        c = GraphCache(cache_dir, ttl_hours=0)  # 0 = no TTL
        c.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)
        # ttl_hours=0 means TTL disabled, should still return
        assert c.get(_SAMPLE_SOURCE) is not None

    def test_ttl_expiry(self, cache_dir: Path) -> None:
        c = GraphCache(cache_dir, ttl_hours=1)
        c.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)

        # Backdate the file
        h = compute_source_hash(_SAMPLE_SOURCE)
        path = cache_dir / f"{h}.json"
        old_time = time.time() - 7200  # 2 hours ago
        import os

        os.utime(path, (old_time, old_time))

        assert c.get(_SAMPLE_SOURCE) is None

    def test_non_expired_entry_returned(self, cache_dir: Path) -> None:
        c = GraphCache(cache_dir, ttl_hours=24)
        c.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)
        assert c.get(_SAMPLE_SOURCE) is not None


# =====================================================================
# Stats
# =====================================================================


class TestStats:
    def test_initial_stats(self, cache: GraphCache) -> None:
        s = cache.stats()
        assert s["graph_cache_hits"] == 0
        assert s["graph_cache_misses"] == 0
        assert s["graph_cache_hit_ratio"] == 0.0

    def test_stats_after_hit_and_miss(self, cache: GraphCache) -> None:
        cache.put(_SAMPLE_SOURCE, _SAMPLE_TABLES)
        cache.get(_SAMPLE_SOURCE)       # hit
        cache.get("nonexistent text")   # miss

        s = cache.stats()
        assert s["graph_cache_hits"] == 1
        assert s["graph_cache_misses"] == 1
        assert s["graph_cache_hit_ratio"] == 0.5
        assert s["graph_cache_entries"] == 1

    def test_hit_ratio_zero_on_empty(self, cache: GraphCache) -> None:
        assert cache.hit_ratio == 0.0


# =====================================================================
# Clear
# =====================================================================


class TestClear:
    def test_clear_removes_entries(self, cache: GraphCache) -> None:
        cache.put("text A", _SAMPLE_TABLES)
        cache.put("text B", _SAMPLE_TABLES)
        assert cache_file_count(cache.cache_dir) == 2

        removed = cache.clear()
        assert removed == 2
        assert cache_file_count(cache.cache_dir) == 0

    def test_clear_on_nonexistent_dir(self, tmp_path: Path) -> None:
        c = GraphCache(tmp_path / "does_not_exist")
        assert c.clear() == 0


# =====================================================================
# Preseed from Exp 31
# =====================================================================


class TestPreseedFromExp31:
    def test_preseed_basic(self, cache: GraphCache, tmp_path: Path) -> None:
        exp31_data = {
            "ragtruth_24": {
                "source_len": 100,
                "pass_times": {},
                "passes": {
                    "P1_entities": {"entities": [{"name": "Paris"}]},
                    "P2_events": {"events": []},
                },
            },
            "ragtruth_25": {
                "source_len": 100,
                "pass_times": {},
                "passes": {
                    "P1_entities": {"entities": [{"name": "Paris"}]},
                    "P2_events": {"events": []},
                },
            },
        }
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        source_texts = {
            "ragtruth_24": "Same source document.",
            "ragtruth_25": "Same source document.",
        }

        result = preseed_from_exp31(cache, exp31_path, source_texts)
        assert result["seeded"] == 1  # dedup
        assert result["dedup_savings"] == 1
        assert result["failed"] == []
        assert result["exp31_cases_not_in_source_texts"] == []

    def test_preseed_data_retrievable(
        self, cache: GraphCache, tmp_path: Path
    ) -> None:
        source_text = "Unique source document for retrieval test."
        exp31_data = {
            "case_1": {
                "passes": {"P1_entities": {"entities": [{"name": "Test"}]}},
            },
        }
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        preseed_from_exp31(cache, exp31_path, {"case_1": source_text})

        result = cache.get(source_text)
        assert result is not None
        assert result["passes"]["P1_entities"]["entities"][0]["name"] == "Test"

    # Silence contract (#180): partial-coverage misses must surface.

    def test_missing_fact_table_listed_in_failed(
        self, cache: GraphCache, tmp_path: Path
    ) -> None:
        """case_ids declared by the caller but absent from Exp 31 → failed list."""
        exp31_data = {"ragtruth_24": {"passes": {"P1_entities": {}}}}
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        # Caller declares two cases; Exp 31 has one.
        source_texts = {"ragtruth_24": "a", "ragtruth_99": "b"}

        result = preseed_from_exp31(cache, exp31_path, source_texts)
        assert result["seeded"] == 1
        assert result["failed"] == ["ragtruth_99"]

    def test_strict_raises_on_any_miss(
        self, cache: GraphCache, tmp_path: Path
    ) -> None:
        """With strict=True, any miss raises with actionable detail."""
        exp31_data = {"ragtruth_24": {"passes": {"P1_entities": {}}}}
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        source_texts = {"ragtruth_24": "a", "ragtruth_99": "b"}

        with pytest.raises(FactTableValidationError) as exc_info:
            preseed_from_exp31(cache, exp31_path, source_texts, strict=True)
        msg = str(exc_info.value)
        assert "ragtruth_99" in msg
        assert "1 case_id" in msg

    def test_strict_passes_on_full_coverage(
        self, cache: GraphCache, tmp_path: Path
    ) -> None:
        """strict=True is a no-op when every declared case has a fact table."""
        exp31_data = {
            "ragtruth_24": {"passes": {"P1_entities": {}}},
            "ragtruth_25": {"passes": {"P1_entities": {}}},
        }
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        source_texts = {"ragtruth_24": "a", "ragtruth_25": "b"}

        result = preseed_from_exp31(
            cache, exp31_path, source_texts, strict=True
        )
        assert result["seeded"] == 2
        assert result["failed"] == []

    def test_drift_cases_reported_separately(
        self, cache: GraphCache, tmp_path: Path
    ) -> None:
        """Exp 31 entries the caller didn't declare are informational drift,
        not a failure."""
        exp31_data = {
            "ragtruth_24": {"passes": {"P1_entities": {}}},
            "ragtruth_999": {"passes": {"P1_entities": {}}},  # not declared
        }
        exp31_path = tmp_path / "exp31.json"
        exp31_path.write_text(json.dumps(exp31_data))

        source_texts = {"ragtruth_24": "a"}

        result = preseed_from_exp31(cache, exp31_path, source_texts)
        assert result["seeded"] == 1
        assert result["failed"] == []
        assert result["exp31_cases_not_in_source_texts"] == ["ragtruth_999"]

    def test_preseed_file_not_found_raises(self, cache: GraphCache) -> None:
        """Missing Exp 31 file is a setup error; must be loud."""
        with pytest.raises(FileNotFoundError):
            preseed_from_exp31(cache, "/nonexistent.json", {})


# =====================================================================
# Singleton
# =====================================================================


class TestSingleton:
    def test_get_graph_cache_returns_same_instance(self, tmp_path: Path) -> None:
        reset_graph_cache()
        c1 = get_graph_cache(tmp_path / "cache1", force_new=True)
        c2 = get_graph_cache(tmp_path / "cache2")  # ignored, returns c1
        assert c1 is c2

    def test_force_new_creates_new_instance(self, tmp_path: Path) -> None:
        reset_graph_cache()
        c1 = get_graph_cache(tmp_path / "cache1", force_new=True)
        c2 = get_graph_cache(tmp_path / "cache2", force_new=True)
        assert c1 is not c2

    def test_reset_clears_singleton(self, tmp_path: Path) -> None:
        reset_graph_cache()
        c1 = get_graph_cache(tmp_path / "cache1", force_new=True)
        reset_graph_cache()
        c2 = get_graph_cache(tmp_path / "cache2", force_new=True)
        assert c1 is not c2

    def teardown_method(self) -> None:
        reset_graph_cache()


# =====================================================================
# Integration: hallucination.py L2 cache paths
# =====================================================================


class TestHallucinationCacheIntegration:
    """Test that check_hallucination uses graph cache for L2."""

    _RESPONSE = "Paris is the capital of France."
    _SOURCE = "Paris is the capital of France."

    def test_cache_hit_populates_layer_stats(self) -> None:
        """When cache has fact tables for the source, L2_cache_hit is set."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = _SAMPLE_TABLES

        with (
            patch(
                "llm_judge.calibration.graph_cache.get_graph_cache",
                return_value=mock_cache,
            ) as mock_get_cache,
            patch(
                "llm_judge.calibration.hallucination_graphs.build_all_graphs",
                return_value={},
            ),
        ):
            from llm_judge.calibration.hallucination import check_hallucination

            result = check_hallucination(
                response=self._RESPONSE,
                context=self._SOURCE,
                source_context=self._SOURCE,
                case_id="test_cache_hit",
                l2_enabled=True,
                gate2_routing="pass",
                skip_embeddings=True,
                l1_enabled=False,
            )

        mock_get_cache.assert_called_once()
        mock_cache.get.assert_called_once_with(self._SOURCE)
        assert result.layer_stats.get("L2_cache_hit") == 1

    def test_cache_miss_populates_layer_stats(self) -> None:
        """When cache misses, L2_cache_miss is set and L2 is skipped."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        with patch(
            "llm_judge.calibration.graph_cache.get_graph_cache",
            return_value=mock_cache,
        ):
            from llm_judge.calibration.hallucination import check_hallucination

            result = check_hallucination(
                response=self._RESPONSE,
                context=self._SOURCE,
                source_context=self._SOURCE,
                case_id="test_cache_miss",
                l2_enabled=True,
                gate2_routing="pass",
                skip_embeddings=True,
                l1_enabled=False,
            )

        assert result.layer_stats.get("L2_cache_miss") == 1
        # L2 should not run (no fact_tables, no knowledge_graphs)
        assert result.layer_stats.get("L2", 0) == 0

    def test_no_cache_lookup_when_fact_tables_provided(self) -> None:
        """When caller provides fact_tables, cache is not consulted."""
        with patch(
            "llm_judge.calibration.graph_cache.get_graph_cache",
        ) as mock_get_cache:
            from llm_judge.calibration.hallucination import check_hallucination

            check_hallucination(
                response=self._RESPONSE,
                context=self._SOURCE,
                source_context=self._SOURCE,
                fact_tables=_SAMPLE_TABLES,
                case_id="test_no_cache_when_provided",
                l2_enabled=True,
                gate2_routing="pass",
                skip_embeddings=True,
                l1_enabled=False,
            )

        mock_get_cache.assert_not_called()

    def test_no_cache_lookup_when_l2_disabled(self) -> None:
        """When L2 is disabled, cache is not consulted."""
        with patch(
            "llm_judge.calibration.graph_cache.get_graph_cache",
        ) as mock_get_cache:
            from llm_judge.calibration.hallucination import check_hallucination

            check_hallucination(
                response=self._RESPONSE,
                context=self._SOURCE,
                source_context=self._SOURCE,
                case_id="test_no_cache_l2_off",
                l2_enabled=False,
                gate2_routing="pass",
                skip_embeddings=True,
                l1_enabled=False,
            )

        mock_get_cache.assert_not_called()


# =====================================================================
# Helpers
# =====================================================================


def cache_file_count(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.json"))
