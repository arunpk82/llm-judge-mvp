#!/usr/bin/env bash
set -euo pipefail

echo "======================================"
echo "LLM-Judge Local Preflight"
echo "======================================"

# --- 1. Install / sync environment ---
echo ""
echo "[1/7] Installing dependencies..."
poetry install --no-interaction --no-root

# --- 2. Lint ---
echo ""
echo "[2/7] Ruff lint..."
poetry run ruff check . --fix

# --- 3. Type check ---
echo ""
echo "[3/7] Mypy type checking..."
poetry run mypy .

# --- 4. Unit tests ---
echo ""
echo "[4/7] Pytest..."
poetry run pytest -q

# --- 5. Validate baseline integrity ---
echo ""
echo "[5/7] Validating baseline integrity..."
poetry run python -m llm_judge.eval.baseline validate --all

# --- 6. Run PR gate ---
echo ""
echo "[6/7] Running PR gate evaluation..."
poetry run python -m llm_judge.eval.run \
  --spec configs/runspecs/pr_gate.yaml

# Capture latest run directory
LATEST_RUN=$(ls -td reports/runs/pr-gate-* | head -n 1)

if [[ -z "${LATEST_RUN}" ]]; then
  echo "ERROR: Could not find PR gate run directory."
  exit 1
fi

echo "Latest run: ${LATEST_RUN}"

# --- 7. Eval diff against baseline ---
echo ""
echo "[7/7] Running eval diff against baseline..."

poetry run python -m llm_judge.eval.diff \
  --baseline baselines/golden/chat_quality \
  --candidate "${LATEST_RUN}"

echo ""
echo "======================================"
echo "Preflight SUCCESS — ready to push."
echo "======================================"