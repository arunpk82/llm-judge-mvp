#!/bin/bash
# ==============================================================
# apply_l2_ensemble.sh — Apply L2 Patterns pipeline changes
# ==============================================================
# 
# What this script does:
#   1. Copies NEW hallucination_graphs.py (L2 ensemble module)
#   2. Replaces hallucination.py with updated pipeline
#   3. Updates test_5layer_pipeline.py layer references
#   4. Copies test_production_pipeline.py (RAGTruth 50 verification)
#
# Run from repo root: bash tools/apply_l2_ensemble.sh
#
# After running:
#   poetry run pytest tests/unit/test_5layer_pipeline.py -v
#   poetry run python experiments/test_production_pipeline.py
# ==============================================================

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "Repo root: $REPO_ROOT"

# Step 1: Copy new L2 ensemble module
echo "Step 1: Adding hallucination_graphs.py..."
# This file should already be in the working directory from outputs
if [ ! -f "$REPO_ROOT/hallucination_graphs.py" ]; then
    echo "ERROR: hallucination_graphs.py not found in repo root."
    echo "Copy it from outputs first:"
    echo "  cp hallucination_graphs.py src/llm_judge/calibration/hallucination_graphs.py"
    exit 1
fi
cp "$REPO_ROOT/hallucination_graphs.py" "$REPO_ROOT/src/llm_judge/calibration/hallucination_graphs.py"
echo "  ✓ Created src/llm_judge/calibration/hallucination_graphs.py"

# Step 2: Replace hallucination.py
echo "Step 2: Updating hallucination.py..."
if [ ! -f "$REPO_ROOT/hallucination_modified.py" ]; then
    echo "ERROR: hallucination_modified.py not found."
    exit 1
fi
cp "$REPO_ROOT/src/llm_judge/calibration/hallucination.py" "$REPO_ROOT/src/llm_judge/calibration/hallucination.py.bak"
cp "$REPO_ROOT/hallucination_modified.py" "$REPO_ROOT/src/llm_judge/calibration/hallucination.py"
echo "  ✓ Backed up original to hallucination.py.bak"
echo "  ✓ Updated src/llm_judge/calibration/hallucination.py"

# Step 3: Update test file layer references
echo "Step 3: Updating test_5layer_pipeline.py layer references..."
TEST_FILE="$REPO_ROOT/tests/unit/test_5layer_pipeline.py"
cp "$TEST_FILE" "$TEST_FILE.bak"

# Rename layer_stats keys in test assertions
sed -i 's/layer_stats\[\"L0\"\]/layer_stats["L1"]/g' "$TEST_FILE"
sed -i 's/layer_stats.get(\"L0\"/layer_stats.get("L1"/g' "$TEST_FILE"
sed -i 's/layer_stats\[\"L1_fail\"\]/layer_stats["L3_gate1_fail"]/g' "$TEST_FILE"
sed -i 's/layer_stats.get(\"L1_fail\"/layer_stats.get("L3_gate1_fail"/g' "$TEST_FILE"
sed -i 's/layer_stats\[\"L2a_minicheck\"\]/layer_stats["L3a_minicheck"]/g' "$TEST_FILE"
sed -i 's/layer_stats.get(\"L2a_minicheck\"/layer_stats.get("L3a_minicheck"/g' "$TEST_FILE"
sed -i 's/layer_stats\[\"L2b_nli\"\]/layer_stats["L3b_nli"]/g' "$TEST_FILE"
sed -i 's/layer_stats.get(\"L2b_nli\"/layer_stats.get("L3b_nli"/g' "$TEST_FILE"
sed -i 's/layer_stats\[\"L3\"\]/layer_stats["L2"]/g' "$TEST_FILE"
sed -i 's/layer_stats.get(\"L3\"/layer_stats.get("L2"/g' "$TEST_FILE"

# Rename resolved_by values in assertions
sed -i 's/resolved_by=\"L0\"/resolved_by="L1"/g' "$TEST_FILE"
sed -i 's/resolved_by=.L0./resolved_by="L1"/g' "$TEST_FILE"
sed -i 's/\"L0\", \"L2a_minicheck\"/"L1", "L3a_minicheck"/g' "$TEST_FILE"
sed -i 's/\"L0\"/"L1"/g' "$TEST_FILE"

# Update class names and docstrings
sed -i 's/class TestL0DeterministicMatch/class TestL1RulesMatch/g' "$TEST_FILE"
sed -i 's/L0 is the cheapest layer/L1 is the cheapest layer/g' "$TEST_FILE"
sed -i 's/class TestL1Gate1Check/class TestL3Gate1Check/g' "$TEST_FILE"
sed -i 's/class TestL2aMiniCheck/class TestL3aMiniCheck/g' "$TEST_FILE"
sed -i 's/class TestL2Fallback/class TestL3Fallback/g' "$TEST_FILE"
sed -i 's/class TestL2NLICheck/class TestL3bNLICheck/g' "$TEST_FILE"
sed -i 's/class TestL3GraphRAGCheck/class TestL2EnsembleCheck/g' "$TEST_FILE"

# Update module docstring
sed -i 's/5-layer groundedness pipeline/4-layer groundedness pipeline/g' "$TEST_FILE"
sed -i 's/L0: Deterministic text match/L1: Rules — exact text match/g' "$TEST_FILE"
sed -i 's/L1: Gate 1 MiniLM/L3: Gate 1 MiniLM/g' "$TEST_FILE"
sed -i 's/L2a: MiniCheck factual/L3a: MiniCheck factual/g' "$TEST_FILE"
sed -i 's/L2b: NLI DeBERTa/L3b: NLI DeBERTa/g' "$TEST_FILE"
sed -i 's/L3: GraphRAG spaCy exact match/L2: Knowledge Graph Ensemble/g' "$TEST_FILE"

# Update comments
sed -i 's/# L0: Deterministic/# L1: Rules/g' "$TEST_FILE"
sed -i 's/# L1: Gate 1/# L3: Gate 1/g' "$TEST_FILE"
sed -i 's/# L2a: MiniCheck/# L3a: MiniCheck/g' "$TEST_FILE"
sed -i 's/# L2b: NLI/# L3b: NLI/g' "$TEST_FILE"
sed -i 's/# L3: GraphRAG/# L2: Knowledge Graph Ensemble/g' "$TEST_FILE"
sed -i 's/# L2a.*L2b Fallback/# L3a→L3b Fallback/g' "$TEST_FILE"

echo "  ✓ Updated layer references in test file"
echo "  ✓ Backed up original to test_5layer_pipeline.py.bak"

# Step 4: Copy production pipeline test
echo "Step 4: Adding production pipeline test..."
if [ -f "$REPO_ROOT/test_production_pipeline.py" ]; then
    cp "$REPO_ROOT/test_production_pipeline.py" "$REPO_ROOT/experiments/test_production_pipeline.py"
    echo "  ✓ Created experiments/test_production_pipeline.py"
fi

echo ""
echo "============================================================"
echo "CHANGES APPLIED"
echo "============================================================"
echo ""
echo "New files:"
echo "  src/llm_judge/calibration/hallucination_graphs.py  (L2 ensemble module)"
echo "  experiments/test_production_pipeline.py             (RAGTruth 50 verification)"
echo ""
echo "Modified files:"
echo "  src/llm_judge/calibration/hallucination.py          (pipeline reorder + L2 integration)"
echo "  tests/unit/test_5layer_pipeline.py                  (layer renaming)"
echo ""
echo "Backup files (delete after verification):"
echo "  src/llm_judge/calibration/hallucination.py.bak"
echo "  tests/unit/test_5layer_pipeline.py.bak"
echo ""
echo "Next steps:"
echo "  1. poetry run pytest tests/unit/test_5layer_pipeline.py -v"
echo "  2. poetry run python experiments/test_production_pipeline.py"
echo "  3. If tests pass: make git-start BRANCH=feature/l2-ensemble-pipeline"
echo "  4. make git-ship MSG='feat: L2 knowledge graph ensemble, layer renaming L0→L1'"
echo "  5. make git-merge"
