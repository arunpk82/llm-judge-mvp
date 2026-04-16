#!/usr/bin/env bash
# Wave D apply script — Graph Cache (ADR-0025)
# Run from repo root: bash tools/wave_d_apply.sh
set -euo pipefail

echo "=== Wave D: Graph Cache (ADR-0025) ==="

# Verify we're in repo root
if [ ! -f pyproject.toml ]; then
    echo "ERROR: run from repo root (llm-judge-mvp/)"
    exit 1
fi

echo "[1/4] New files..."
for f in \
    src/llm_judge/calibration/graph_cache.py \
    tests/unit/test_graph_cache.py \
; do
    if [ -f "$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ MISSING: $f"
        exit 1
    fi
done

echo "[2/4] Modified files..."
for f in \
    src/llm_judge/calibration/hallucination.py \
; do
    if [ -f "$f" ]; then
        echo "  ✓ $f (L2 cache wiring)"
    else
        echo "  ✗ MISSING: $f"
        exit 1
    fi
done

echo "[3/4] Checking Exp 31 pre-seed data..."
if [ -f experiments/exp31_multipass_fact_tables.json ]; then
    echo "  ✓ Exp 31 data available ($(wc -c < experiments/exp31_multipass_fact_tables.json) bytes)"
else
    echo "  ⚠ Exp 31 data not found (pre-seeding will fail)"
fi

echo "[4/4] Running tests..."
poetry run pytest tests/unit/test_graph_cache.py -v --no-cov

echo ""
echo "=== Wave D applied successfully ==="
echo ""
echo "New module: src/llm_judge/calibration/graph_cache.py (347 lines, 17 functions)"
echo "  - GraphCache: filesystem-backed, content-addressable by SHA-256"
echo "  - preseed_from_exp31: bulk load + source deduplication"
echo "  - get_graph_cache: singleton (config-driven or explicit)"
echo ""
echo "Wiring: hallucination.py L2 cache lookup before graph building"
echo "  - Cache hit → uses cached fact_tables, L2_cache_hit in layer_stats"
echo "  - Cache miss → L2 skipped (no API call in hot path), L2_cache_miss"
echo ""
echo "Next: make git-ship MSG='feat(wave-d): L2 graph cache with Exp 31 pre-seed (ADR-0025)'"
