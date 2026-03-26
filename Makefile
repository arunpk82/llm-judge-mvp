# ----------------------------------------
# LLM-Judge Makefile
# World-class local workflow wrapper
# ----------------------------------------

.PHONY: help install lint typecheck test baseline-validate pr-gate diff baseline-dry-run baseline-promote \
        registry-list registry-show registry-trend eval preflight clean

# Defaults (override like: make SUITE=golden RUBRIC=chat_quality ...)
SUITE ?= math_basic
RUBRIC ?= math_basic
RUNSPEC ?= configs/runspecs/pr_gate.yaml
POLICY ?= configs/policies/baseline_promotion.yaml

# Run registry defaults
REGISTRY ?= reports/run_registry.jsonl
REG_LIMIT ?= 20
REG_METRIC ?= f1
REG_LAST ?= 20
# For registry-show (override: make registry-show RUN_ID=pr-gate-...)
RUN_ID ?=

help:
	@echo ""
	@echo "Available targets:"
	@echo "  make install               Install dependencies"
	@echo "  make lint                  Run Ruff"
	@echo "  make typecheck             Run mypy"
	@echo "  make test                  Run pytest"
	@echo "  make baseline-validate     Validate baseline integrity"
	@echo "  make pr-gate               Run PR gate evaluation"
	@echo "  make diff                  Run eval diff against baseline (diagnostic)"
	@echo "  make baseline-dry-run      Policy-gated baseline promotion check (NO writes)"
	@echo "  make baseline-promote      Promote latest run to baseline (writes snapshot)"
	@echo "  make registry-list         List recent runs from run registry"
	@echo "  make registry-show         Show one run entry (RUN_ID=...)"
	@echo "  make registry-trend        Print metric trend (REG_METRIC=..., REG_LAST=...)"
	@echo "  make eval                  Daily workflow: pr-gate + baseline-dry-run + registry-list"
	@echo "  make preflight             Full local preflight (safe, governance-aligned)"
	@echo "  make clean                 Remove generated PR-gate run artifacts"
	@echo ""
	@echo "Config overrides:"
	@echo "  SUITE=$(SUITE) RUBRIC=$(RUBRIC)"
	@echo "  RUNSPEC=$(RUNSPEC)"
	@echo "  POLICY=$(POLICY)"
	@echo "  REGISTRY=$(REGISTRY) REG_LIMIT=$(REG_LIMIT) REG_METRIC=$(REG_METRIC) REG_LAST=$(REG_LAST)"
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
		--spec $(RUNSPEC)

# Diagnostic diff: baseline latest vs latest PR-gate run
diff:
	@set -e; \
	BASELINE_ROOT=baselines/$(SUITE)/$(RUBRIC); \
	LATEST_JSON=$$BASELINE_ROOT/latest.json; \
	if [ ! -f "$$LATEST_JSON" ]; then \
		echo "ERROR: Missing baseline pointer: $$LATEST_JSON"; \
		exit 1; \
	fi; \
	BASELINE_ID=$$(jq -r '.baseline_id' $$LATEST_JSON); \
	if [ -z "$$BASELINE_ID" ] || [ "$$BASELINE_ID" = "null" ]; then \
		echo "ERROR: Could not resolve baseline_id from $$LATEST_JSON"; \
		exit 1; \
	fi; \
	BASELINE_PATH=$$BASELINE_ROOT/snapshots/$$BASELINE_ID; \
	if [ ! -d "$$BASELINE_PATH" ]; then \
		echo "ERROR: Snapshot directory not found: $$BASELINE_PATH"; \
		exit 1; \
	fi; \
	LATEST_RUN=$$(ls -td reports/runs/pr-gate-* 2>/dev/null | head -n 1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "ERROR: No PR gate runs found under reports/runs/pr-gate-*"; \
		exit 1; \
	fi; \
	echo "Baseline:  $$BASELINE_PATH"; \
	echo "Candidate: $$LATEST_RUN"; \
	poetry run python -m llm_judge.eval.diff \
		--baseline $$BASELINE_PATH \
		--candidate $$LATEST_RUN

# P0: Governance gate for PRs — computes policy decision but does NOT write a new baseline
baseline-dry-run:
	@set -e; \
	LATEST_RUN=$$(ls -td reports/runs/pr-gate-* 2>/dev/null | head -n 1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "ERROR: No PR gate runs found under reports/runs/pr-gate-*"; \
		exit 1; \
	fi; \
	if [ ! -f "$(POLICY)" ]; then \
		echo "ERROR: Missing policy file: $(POLICY)"; \
		exit 1; \
	fi; \
	echo "Policy:    $(POLICY)"; \
	echo "Run dir:   $$LATEST_RUN"; \
	echo "Suite:     $(SUITE)"; \
	echo "Rubric:    $(RUBRIC)"; \
	poetry run python -m llm_judge.eval.baseline promote \
		--run-dir $$LATEST_RUN \
		--policy $(POLICY) \
		--baselines-dir baselines \
		--suite $(SUITE) \
		--rubric-id $(RUBRIC) \
		--dry-run

# P1/P0 (maintainers): Actually promotes the latest run to baseline (writes snapshot + updates latest.json)
baseline-promote:
	@set -e; \
	LATEST_RUN=$$(ls -td reports/runs/pr-gate-* 2>/dev/null | head -n 1); \
	if [ -z "$$LATEST_RUN" ]; then \
		echo "ERROR: No PR gate runs found under reports/runs/pr-gate-*"; \
		exit 1; \
	fi; \
	if [ ! -f "$(POLICY)" ]; then \
		echo "ERROR: Missing policy file: $(POLICY)"; \
		exit 1; \
	fi; \
	echo "Policy:    $(POLICY)"; \
	echo "Run dir:   $$LATEST_RUN"; \
	echo "Suite:     $(SUITE)"; \
	echo "Rubric:    $(RUBRIC)"; \
	poetry run python -m llm_judge.eval.baseline promote \
		--run-dir $$LATEST_RUN \
		--policy $(POLICY) \
		--baselines-dir baselines \
		--suite $(SUITE) \
		--rubric-id $(RUBRIC)

# -----------------------------
# EPIC-3: Run Registry / Observability
# -----------------------------

registry-list:
	@set -e; \
	if [ ! -f "$(REGISTRY)" ]; then \
		echo "ERROR: Registry file not found: $(REGISTRY)"; \
		echo "Hint: run 'make pr-gate' first (run.py appends registry entries)."; \
		exit 1; \
	fi; \
	poetry run python -m llm_judge.eval.registry \
		--registry $(REGISTRY) \
		list \
		--limit $(REG_LIMIT) \
		--metric $(REG_METRIC) \
		--dataset-id $(SUITE) \
		--rubric-id $(RUBRIC)

registry-show:
	@set -e; \
	if [ -z "$(RUN_ID)" ]; then \
		echo "ERROR: RUN_ID is required. Example:"; \
		echo "  make registry-show RUN_ID=pr-gate-20260305-120312-123456"; \
		exit 1; \
	fi; \
	if [ ! -f "$(REGISTRY)" ]; then \
		echo "ERROR: Registry file not found: $(REGISTRY)"; \
		exit 1; \
	fi; \
	poetry run python -m llm_judge.eval.registry \
		--registry $(REGISTRY) \
		show $(RUN_ID)

registry-trend:
	@set -e; \
	if [ ! -f "$(REGISTRY)" ]; then \
		echo "ERROR: Registry file not found: $(REGISTRY)"; \
		echo "Hint: run 'make pr-gate' first (run.py appends registry entries)."; \
		exit 1; \
	fi; \
	poetry run python -m llm_judge.eval.registry \
		--registry $(REGISTRY) \
		trend \
		--metric $(REG_METRIC) \
		--dataset-id $(SUITE) \
		--rubric-id $(RUBRIC) \
		--last $(REG_LAST)

.PHONY: drift-check

DRIFT_POLICY ?= configs/policies/drift.yaml

drift-check:
	@set -e; \
	if [ ! -f "$(DRIFT_POLICY)" ]; then \
		echo "ERROR: Missing drift policy file: $(DRIFT_POLICY)"; \
		exit 1; \
	fi; \
	if [ ! -f "reports/run_registry.jsonl" ]; then \
		echo "ERROR: Missing run registry: reports/run_registry.jsonl"; \
		echo "Hint: run 'make pr-gate' first."; \
		exit 1; \
	fi; \
	poetry run python -m llm_judge.eval.drift check \
		--policy $(DRIFT_POLICY) \
		--registry reports/run_registry.jsonl \
		--baselines-dir baselines \
		--suite $(SUITE) \
		--rubric-id $(RUBRIC)

.PHONY: rules-list rules-validate rules-export

rules-list:
	poetry run python -m llm_judge.rules.lifecycle list

rules-validate:
	poetry run python -m llm_judge.rules.lifecycle validate

rules-export:
	poetry run python -m llm_judge.rules.lifecycle export-json \
		--out reports/rules_registry.json

# Daily workflow: produce run + enforce governance + show observability
eval: pr-gate baseline-dry-run registry-list
	@echo ""
	@echo "======================================"
	@echo "Eval SUCCESS — run, gate, registry OK."
	@echo "======================================"

# --- Automated Git Workflows ---

# Usage: make git-start FEATURE=my-new-idea
git-start:
	@if [ -z "$(FEATURE)" ]; then \
		echo "ERROR: Please provide a branch name. Example: make git-start FEATURE=my-new-feature"; \
		exit 1; \
	fi
	@echo "Syncing main and creating new branch: $(FEATURE)..."
	git checkout $(BRANCH)
	git pull origin $(BRANCH)
	git checkout -b $(FEATURE)
	@echo "Branch $(FEATURE) created and ready for development."

# Usage: make git-ship MSG="feat: updated prompt logic"
git-ship: preflight
	@if [ -z "$(MSG)" ]; then \
		echo "ERROR: Please provide a commit message. Example: make git-ship MSG=\"fix: typo in prompt\""; \
		exit 1; \
	fi
	@echo "Staging all changes..."
	git add .
	@echo "Committing with message: $(MSG)"
	git commit -m "$(MSG)"
	@echo "Pushing to remote..."
	git push -u origin HEAD
	@echo "Successfully shipped! 🚀"

# Governance-aligned preflight:
# - validates toolchain
# - runs pr-gate
# - runs policy gate (dry-run)
# - validates run registry append + listing
preflight: lint typecheck test rules-validate baseline-validate pr-gate baseline-dry-run registry-list drift-check
	@echo ""
	@echo "======================================"
	@echo "Preflight SUCCESS — ready to push."
	@echo "======================================"

clean:
	rm -rf reports/runs/pr-gate-*