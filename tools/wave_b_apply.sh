#!/usr/bin/env bash
# Wave B Apply Script — ADR-0026 Config Schema + Validator
# ============================================================
# Usage:
#   cd ~/work/llm-judge-mvp
#   make git-start BRANCH=wave-b/config-schema
#   # Copy wave_b_delivery_structured.zip to repo root
#   unzip -o wave_b_delivery_structured.zip
#   bash tools/wave_b_apply.sh
#   make git-ship MSG="feat(wave-b): pipeline config schema, validator, and caller wiring (ADR-0026)"
# ============================================================

set -euo pipefail

echo "=== Wave B Apply Script ==="
echo ""

# --- Deletions (files that the zip can't represent) ---

# B2a: Delete pytest.ini (source of three-way config disagreement)
if [ -f pytest.ini ]; then
    rm pytest.ini
    echo "✅ Deleted pytest.ini (pyproject.toml is now the single pytest config)"
else
    echo "⏭  pytest.ini already absent"
fi

# B2b: Delete .bak file
if [ -f tests/unit/test_5layer_pipeline.py.bak ]; then
    rm tests/unit/test_5layer_pipeline.py.bak
    echo "✅ Deleted test_5layer_pipeline.py.bak"
else
    echo "⏭  .bak file already absent"
fi

# B2c: Delete Zone.Identifier files (Windows metadata)
ZONE_COUNT=$(find . -name "*:Zone.Identifier" -not -path "./.git/*" -not -path "*/node_modules/*" | wc -l)
if [ "$ZONE_COUNT" -gt 0 ]; then
    find . -name "*:Zone.Identifier" -not -path "./.git/*" -not -path "*/node_modules/*" -delete
    echo "✅ Deleted $ZONE_COUNT Zone.Identifier files"
else
    echo "⏭  No Zone.Identifier files found"
fi

# B2d: Delete orphaned root-level pipeline config
if [ -f hallucination_pipeline_config.yaml ]; then
    rm hallucination_pipeline_config.yaml
    echo "✅ Deleted root-level hallucination_pipeline_config.yaml (superseded by configs/pipeline/hallucination_pipeline.yaml)"
else
    echo "⏭  Root-level config already absent"
fi

echo ""
echo "=== Verification ==="

# Verify new files exist
for f in \
    src/llm_judge/calibration/pipeline_config.py \
    configs/pipeline/hallucination_pipeline.yaml \
    tests/unit/test_pipeline_config.py; do
    if [ -f "$f" ]; then
        echo "✅ $f exists"
    else
        echo "❌ MISSING: $f"
        exit 1
    fi
done

# Verify pytest.ini is gone
if [ -f pytest.ini ]; then
    echo "❌ pytest.ini still exists — delete it manually"
    exit 1
else
    echo "✅ pytest.ini removed (no more three-way config disagreement)"
fi

# Verify production config loads
echo ""
echo "=== Config Validation ==="
python -c "
import sys
sys.path.insert(0, 'src')
from llm_judge.calibration.pipeline_config import load_pipeline_config
from pathlib import Path
cfg = load_pipeline_config(Path('configs/pipeline/hallucination_pipeline.yaml'))
print(f'  Config loaded: {cfg.source_path}')
print(f'  Hash: {cfg.config_hash[:16]}...')
print(f'  L1={cfg.layers.l1_enabled} L2={cfg.layers.l2_enabled} L3={cfg.layers.l3_enabled} L4={cfg.layers.l4_enabled}')
print(f'  L3 method: {cfg.l3_method}')
print(f'  Splitter: {cfg.sentence_splitter}')
m = cfg.to_manifest_dict()
nested = [k for k, v in m.items() if isinstance(v, (dict, list))]
if nested:
    print(f'  ❌ Manifest has nested keys: {nested}')
    sys.exit(1)
print(f'  Manifest: {len(m)} flat keys ✅')
" && echo "✅ Config validation passed" || { echo "❌ Config validation failed"; exit 1; }

echo ""
echo "=== Wave B Applied Successfully ==="
echo ""
echo "Next steps:"
echo "  1. poetry run pytest tests/unit/test_pipeline_config.py -v"
echo "  2. poetry run pytest -m 'not nightly' -q  # full suite"
echo "  3. make git-ship MSG='feat(wave-b): pipeline config schema, validator, and caller wiring (ADR-0026)'"
