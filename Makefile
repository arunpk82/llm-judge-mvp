# ----------------------------------------
# LLM-Judge Makefile
# World-class local workflow wrapper
# ----------------------------------------

.PHONY: help install lint typecheck test baseline-validate pr-gate diff preflight clean

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make install             Install dependencies"
	@echo "  make lint                Run Ruff"
	@echo "  make typecheck           Run mypy"
	@echo "  make test                Run pytest"
	@echo "  make baseline-validate   Validate baseline integrity"
	@echo "  make pr-gate             Run PR gate evaluation"
	@echo "  make diff                Run eval diff against baseline"
	@echo "  make preflight           Run full local preflight"
	@echo "  make clean               Remove generated run artifacts"
	@echo ""

install:
	poetry install --no-interaction --no-root

lint:
	poetry run ruff check . --fix

typecheck:
	poetry run mypy .

test:
	poetry run pytest -q

baseline-validate:
	poetry run python -m llm_judge.eval.baseline validate --all

pr-gate:
	poetry run python -m llm_judge.eval.run \
		--spec configs/runspecs/pr_gate.yaml

diff:
	@BASELINE_ROOT=baselines/golden/chat_quality; \
	BASELINE_ID=$$(jq -r '.baseline_id' $$BASELINE_ROOT/latest.json); \
	if [ -z "$$BASELINE_ID" ] || [ "$$BASELINE_ID" = "null" ]; then \
		echo "ERROR: Could not resolve baseline_id from latest.json"; \
		exit 1; \
	fi; \
	BASELINE_PATH=$$BASELINE_ROOT/snapshots/$$BASELINE_ID; \
	if [ ! -d "$$BASELINE_PATH" ]; then \
		echo "ERROR: Snapshot directory not found: $$BASELINE_PATH"; \
		exit 1; \
	fi; \
	LATEST_RUN=$$(ls -td reports/runs/pr-gate-* | head -n 1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "ERROR: No PR gate runs found."; \
		exit 1; \
	fi; \
	echo "Baseline: $$BASELINE_PATH"; \
	echo "Candidate: $$LATEST_RUN"; \
	poetry run python -m llm_judge.eval.diff \
		--baseline $$BASELINE_PATH \
		--candidate $$LATEST_RUN

preflight: lint typecheck test baseline-validate pr-gate diff
	@echo ""
	@echo "======================================"
	@echo "Preflight SUCCESS — ready to push."
	@echo "======================================"

clean:
	rm -rf reports/runs/pr-gate-*