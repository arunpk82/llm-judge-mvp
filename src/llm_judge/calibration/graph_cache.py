"""
L2 Graph Cache — Content-Addressable Fact Table Storage (ADR-0025).

Caches pre-extracted fact tables (from Gemini multi-pass extraction)
keyed by source document SHA-256 hash. Extraction happens once at
registration time; runtime is always a cache lookup, never an API call
in the hot path.

Key properties:
  - Content-addressable: same source text → same hash → same cache entry
  - Immutable: once written, entries are not modified (Lockfile/Pinning pattern)
  - TTL support: entries expire after configurable hours (default 168h = 7 days)
  - Pre-seedable: bulk load from Exp 31 multipass fact tables
  - Filesystem-backed: one JSON file per source hash in a configurable directory

Design decisions (ADR-0025):
  - Extraction is a registration step, not a runtime step
  - Cache hit = free L2 knowledge graph ensemble
  - Cache miss = L2 skipped (no API call in hot path), logged as warning
  - Source deduplication: 50 RAGTruth cases share 9 sources = 82% cost reduction

See also: ADR-0018 (multi-pass extraction), ADR-0019 (Gemini model),
          hallucination_graphs.py (build_all_graphs consumes fact tables).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_source_hash(source_text: str) -> str:
    """Compute SHA-256 hash of source text for cache keying."""
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()


# P5 fields where Gemini multi-pass extraction sometimes emits bare strings
# where the builders expect dicts. Normalizing at the cache-load boundary
# keeps the build_g5_negations contract simple (always dict) and isolates
# shape drift to a single, observable site. See #179.
_P5_STR_SCHEMAS: dict[str, tuple[str, dict[str, str]]] = {
    # field name -> (key to fill with the string, extra fixed keys)
    "explicit_negations": ("statement", {}),
    "absent_information": ("what", {}),
    "corrections": ("wrong", {"right": ""}),
}


def _normalize_fact_table(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize fact-table element shapes at the cache-load boundary.

    Gemini multi-pass extraction sometimes emits bare strings where
    ``build_g5_negations`` expects dicts. This function coerces the
    three known shape-drift fields in P5 into canonical dicts so
    downstream builders can rely on ``.get()`` succeeding.

    Schema (P5 only — P1–P4 pass through untouched by current contract):
      * ``P5_negations.explicit_negations``:
          ``"no lighting"`` → ``{"statement": "no lighting"}``
      * ``P5_negations.absent_information``:
          ``"missing info"`` → ``{"what": "missing info"}``
      * ``P5_negations.corrections``:
          ``"correction text"`` →
          ``{"wrong": "correction text", "right": ""}``
          (The empty ``right`` causes ``build_g5_negations`` to drop
          the entry via its ``if wrong and right`` guard — acceptable:
          the normalizer preserves shape, not semantics. Semantic
          recovery of bare-string corrections is out of scope.)

    Unknown element types (neither str nor dict) are dropped with a
    WARNING log — never silently — so that drift beyond these fields
    stays observable.

    In-place, idempotent. Returns the same dict.
    """
    passes = data.get("passes", data)
    if not isinstance(passes, dict):
        return data

    p5 = passes.get("P5_negations")
    if not isinstance(p5, dict):
        return data

    for field, (primary_key, extras) in _P5_STR_SCHEMAS.items():
        raw = p5.get(field)
        if not isinstance(raw, list):
            continue
        normalized: list[dict[str, Any]] = []
        for idx, elem in enumerate(raw):
            if isinstance(elem, dict):
                normalized.append(elem)
            elif isinstance(elem, str):
                shaped: dict[str, Any] = {primary_key: elem}
                shaped.update(extras)
                normalized.append(shaped)
            else:
                logger.warning(
                    "graph_cache.normalize_unknown_element",
                    extra={
                        "field": f"P5_negations.{field}",
                        "index": idx,
                        "type": type(elem).__name__,
                    },
                )
        p5[field] = normalized

    return data


class GraphCache:
    """
    Filesystem-backed cache for L2 fact tables.

    Each entry is a JSON file named ``{sha256_hash}.json`` containing
    the multi-pass fact tables (P1-P6) extracted from the source document.

    Usage::

        cache = GraphCache(Path(".cache/hallucination_graphs"))
        tables = cache.get(source_text)
        if tables is None:
            tables = extract_fact_tables(source_text)  # Gemini API call
            cache.put(source_text, tables)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        *,
        ttl_hours: int = 168,  # 7 days
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._ttl_seconds = ttl_hours * 3600
        self._hits = 0
        self._misses = 0

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def _entry_path(self, source_hash: str) -> Path:
        return self._cache_dir / f"{source_hash}.json"

    def _is_expired(self, path: Path) -> bool:
        if self._ttl_seconds <= 0:
            return False
        try:
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            return age > self._ttl_seconds
        except OSError:
            return True

    def get(self, source_text: str) -> dict[str, Any] | None:
        """
        Look up cached fact tables for a source document.

        Args:
            source_text: The source document text (hashed for lookup).

        Returns:
            The fact tables dict (containing ``passes`` key) if cached
            and not expired, else None.
        """
        source_hash = compute_source_hash(source_text)
        path = self._entry_path(source_hash)

        if not path.exists():
            self._misses += 1
            return None

        if self._is_expired(path):
            self._misses += 1
            logger.debug(
                "graph_cache.expired",
                extra={"hash": source_hash[:16], "path": str(path)},
            )
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data = _normalize_fact_table(data)
            self._hits += 1
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "graph_cache.read_error",
                extra={"hash": source_hash[:16], "error": str(e)[:80]},
            )
            self._misses += 1
            return None

    def get_by_hash(self, source_hash: str) -> dict[str, Any] | None:
        """Look up by pre-computed hash (for callers that already have it)."""
        path = self._entry_path(source_hash)
        if not path.exists() or self._is_expired(path):
            self._misses += 1
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data = _normalize_fact_table(data)
            self._hits += 1
            return data
        except (json.JSONDecodeError, OSError):
            self._misses += 1
            return None

    def put(
        self,
        source_text: str,
        fact_tables: dict[str, Any],
    ) -> str:
        """
        Store fact tables for a source document.

        Creates the cache directory if needed. Returns the source hash.
        Entries are immutable — existing entries are not overwritten.
        """
        source_hash = compute_source_hash(source_text)
        path = self._entry_path(source_hash)

        if path.exists() and not self._is_expired(path):
            logger.debug(
                "graph_cache.already_cached",
                extra={"hash": source_hash[:16]},
            )
            return source_hash

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        try:
            path.write_text(
                json.dumps(fact_tables, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info(
                "graph_cache.stored",
                extra={
                    "hash": source_hash[:16],
                    "passes": list(fact_tables.get("passes", fact_tables).keys()),
                },
            )
        except OSError as e:
            logger.warning(
                "graph_cache.write_error",
                extra={"hash": source_hash[:16], "error": str(e)[:80]},
            )

        return source_hash

    def put_by_hash(
        self,
        source_hash: str,
        fact_tables: dict[str, Any],
    ) -> None:
        """Store by pre-computed hash (for pre-seeding without source text)."""
        path = self._entry_path(source_hash)
        if path.exists() and not self._is_expired(path):
            return

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps(fact_tables, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(
                "graph_cache.write_error",
                extra={"hash": source_hash[:16], "error": str(e)[:80]},
            )

    def stats(self) -> dict[str, Any]:
        """Cache statistics for metrics.json / funnel report."""
        cached_count = 0
        if self._cache_dir.exists():
            cached_count = sum(
                1 for f in self._cache_dir.glob("*.json")
                if not self._is_expired(f)
            )
        return {
            "graph_cache_hits": self._hits,
            "graph_cache_misses": self._misses,
            "graph_cache_hit_ratio": round(self.hit_ratio, 3),
            "graph_cache_entries": cached_count,
            "graph_cache_dir": str(self._cache_dir),
        }

    def clear(self) -> int:
        """Remove all cache entries. Returns count of files removed."""
        if not self._cache_dir.exists():
            return 0
        count = 0
        for f in self._cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        return count


# =====================================================================
# Pre-seeding from Exp 31 fact tables
# =====================================================================


def preseed_from_exp31(
    cache: GraphCache,
    exp31_path: Path | str,
    source_texts: dict[str, str],
) -> dict[str, Any]:
    """
    Pre-seed the graph cache from Exp 31 multipass fact tables.

    Args:
        cache: The GraphCache instance to seed.
        exp31_path: Path to exp31_multipass_fact_tables.json.
        source_texts: Mapping from case_id (e.g. "ragtruth_24") to
            source document text. Used to compute the content hash.
            Multiple case_ids can share the same source text — dedup
            happens automatically via the hash.

    Returns:
        Summary dict with seeded_sources, skipped (no source text),
        and dedup count.
    """
    exp31_path = Path(exp31_path)
    if not exp31_path.exists():
        logger.warning(
            "preseed.file_not_found",
            extra={"path": str(exp31_path)},
        )
        return {"error": f"file not found: {exp31_path}"}

    data = json.loads(exp31_path.read_text(encoding="utf-8"))

    seeded_hashes: set[str] = set()
    skipped: list[str] = []

    for case_id, tables in data.items():
        source_text = source_texts.get(case_id)
        if source_text is None:
            skipped.append(case_id)
            continue

        source_hash = compute_source_hash(source_text)
        if source_hash in seeded_hashes:
            continue  # Already seeded from a sibling case

        # Store the passes (not the wrapper with source_len/pass_times)
        fact_data = {"passes": tables.get("passes", tables)}
        cache.put_by_hash(source_hash, fact_data)
        seeded_hashes.add(source_hash)

    result = {
        "total_cases": len(data),
        "unique_sources_seeded": len(seeded_hashes),
        "skipped_no_source_text": len(skipped),
        "dedup_savings": len(data) - len(seeded_hashes) - len(skipped),
    }

    logger.info("preseed.complete", extra=result)
    return result


# =====================================================================
# Singleton — matches pipeline_config.py pattern
# =====================================================================

_cached_instance: GraphCache | None = None


def get_graph_cache(
    cache_dir: Path | str | None = None,
    *,
    ttl_hours: int = 168,
    force_new: bool = False,
) -> GraphCache:
    """
    Get the graph cache singleton.

    First call creates the instance. Subsequent calls return the cached
    instance. If cache_dir is None, uses the config value.
    """
    global _cached_instance
    if _cached_instance is None or force_new:
        if cache_dir is None:
            # Try to get from pipeline config
            try:
                from llm_judge.calibration.pipeline_config import get_pipeline_config

                cfg = get_pipeline_config()
                cache_dir = cfg.graph_cache.directory
                ttl_hours = cfg.graph_cache.ttl_hours
            except Exception:
                cache_dir = ".cache/hallucination_graphs"

        _cached_instance = GraphCache(Path(cache_dir), ttl_hours=ttl_hours)
    return _cached_instance


def reset_graph_cache() -> None:
    """Reset the singleton. For testing only."""
    global _cached_instance
    _cached_instance = None
