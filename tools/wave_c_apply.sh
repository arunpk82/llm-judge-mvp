#!/usr/bin/env bash
# Wave C Apply Script — Fact-Counting Port (ADR-0027)
# ============================================================
# Usage:
#   cd ~/work/llm-judge-mvp
#   make git-start BRANCH=wave-c/fact-counting
#   # Copy wave_c_delivery.zip to repo root
#   unzip -o wave_c_delivery.zip
#   bash tools/wave_c_apply.sh
#   make git-ship MSG="feat(wave-c): L3 fact-counting port from Exp 43, config-driven method switch (ADR-0027)"
# ============================================================

set -euo pipefail

echo "=== Wave C Apply Script ==="
echo ""

# --- Verify prerequisites (Wave B must be merged) ---
if [ ! -f src/llm_judge/calibration/pipeline_config.py ]; then
    echo "❌ Wave B not merged — pipeline_config.py missing. Merge Wave B first."
    exit 1
fi
echo "✅ Wave B prerequisite: pipeline_config.py present"

# --- Verify new files exist ---
for f in \
    src/llm_judge/calibration/fact_counting.py \
    tests/unit/test_fact_counting.py; do
    if [ -f "$f" ]; then
        echo "✅ $f exists"
    else
        echo "❌ MISSING: $f — did you unzip the delivery?"
        exit 1
    fi
done

# --- Verify config flipped ---
if grep -q 'l3_method: fact_counting' configs/pipeline/hallucination_pipeline.yaml 2>/dev/null; then
    echo "✅ Production config: l3_method=fact_counting"
else
    echo "⚠️  Production config not yet flipped to fact_counting"
fi

# --- Validate fact-counting module loads ---
echo ""
echo "=== Module Validation ==="
python -c "
import sys
sys.path.insert(0, 'src')
from llm_judge.calibration.fact_counting import (
    FACT_COUNTING_PROMPT, FactCountResult, check_fact_counting, _parse_json_safe,
)
# Verify prompt
assert '{source}' in FACT_COUNTING_PROMPT
assert '{claim}' in FACT_COUNTING_PROMPT
for s in ['SUPPORTED', 'NOT_FOUND', 'CONTRADICTED', 'SHIFTED', 'INFERRED']:
    assert s in FACT_COUNTING_PROMPT, f'Missing status: {s}'
# Verify dataclass
r = FactCountResult(supported=3, total=4, ratio=0.75)
d = r.to_evidence_dict()
assert d['fc_ratio'] == 0.75
# Verify parser
parsed = _parse_json_safe('{\"supported\": 1, \"total\": 1}')
assert parsed['supported'] == 1
print('  fact_counting module: ✅')
print(f'  Prompt: {len(FACT_COUNTING_PROMPT)} chars, 5 statuses')
print(f'  FactCountResult: {len(d)} evidence keys')
" && echo "✅ Module validation passed" || { echo "❌ Module validation failed"; exit 1; }

# --- Verify config switch works ---
echo ""
python -c "
import sys
sys.path.insert(0, 'src')
from llm_judge.calibration.pipeline_config import load_pipeline_config, reset_pipeline_config
from pathlib import Path
reset_pipeline_config()
cfg = load_pipeline_config(Path('configs/pipeline/hallucination_pipeline.yaml'))
assert cfg.l3_method == 'fact_counting', f'Expected fact_counting, got {cfg.l3_method}'
assert cfg.thresholds.fact_counting_clear == 0.80
print(f'  Config l3_method={cfg.l3_method}, threshold={cfg.thresholds.fact_counting_clear}')
" && echo "✅ Config switch validated" || { echo "❌ Config switch failed"; exit 1; }

echo ""
echo "=== Wave C Applied Successfully ==="
echo ""
echo "Next steps:"
echo "  1. poetry run pytest tests/unit/test_fact_counting.py -v"
echo "  2. poetry run pytest -m 'not nightly' -q  # full suite"
echo "  3. make git-ship MSG='feat(wave-c): L3 fact-counting port from Exp 43, config-driven method switch (ADR-0027)'"
