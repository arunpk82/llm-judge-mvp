# LLM-Judge Developer Guide

## Getting Started (< 30 minutes)

### 1. Install

```bash
git clone https://github.com/arunpk82/llm-judge-mvp.git
cd llm-judge-mvp
poetry install --no-interaction
```

### 2. Run your first evaluation

```bash
make pr-gate
```

This runs the PR gate evaluation using `configs/runspecs/pr_gate.yaml`, producing:
- `reports/runs/pr-gate-<timestamp>/judgments.jsonl` — per-case verdicts
- `reports/runs/pr-gate-<timestamp>/metrics.json` — Kappa, F1, accuracy, rule hit-rates
- `reports/runs/pr-gate-<timestamp>/manifest.json` — run metadata
- `reports/runs/pr-gate-<timestamp>/validation_report.json` — pre-flight checks

### 3. Compare against baseline

```bash
make diff
```

Shows decision flips, score deltas, and whether the run meets the baseline quality bar.

### 4. Run the full governance check

```bash
make preflight
```

Chains: lint → typecheck → test → rules-validate → baseline-validate → pr-gate → baseline-dry-run → registry-list → drift-check. If it passes, your code is ready to ship.

## Daily Workflow

```bash
make eval          # pr-gate + baseline-dry-run + registry-list
make preflight     # full governance check before pushing
```

## Git Workflow

```bash
make git-start BRANCH=epic/1.1/111-feature-name   # create branch from master
make git-ship MSG="EPIC-1.1: description"          # preflight → commit → push → PR
make git-merge                                      # squash merge after CI passes
```

## Key Commands

### L3 — Governance

| Command | What it does |
|---------|-------------|
| `make test` | Run unit tests |
| `make lint` | Run ruff linter |
| `make typecheck` | Run mypy |
| `make pr-gate` | Run evaluation on PR gate dataset |
| `make diff` | Diff latest run vs baseline |
| `make baseline-dry-run` | Test promotion policy (no writes) |
| `make baseline-promote` | Promote latest run to baseline |
| `make registry-list` | List recent evaluation runs |
| `make registry-trend` | Show metric trends |
| `make drift-check` | Check for metric drift |
| `make rules-list` | List governed rules with aging status |
| `make rules-validate` | Validate rule manifest alignment |
| `make rules-aging` | Show rule aging report |
| `make rules-audit` | Show rule audit log |
| `make dataset-list` | List registered datasets |
| `make dataset-validate` | Validate all datasets |
| `make event-query` | Query event registry |
| `make event-trace RUN_ID=...` | Trace events for a run |
| `make event-stats` | Show event registry statistics |

### L4 — Calibration & Adjudication

| Command | What it does |
|---------|-------------|
| `make judge-list` | List registered LLM judges with calibration status |
| `make judge-status` | Show trust gate status for all judges |

## Project Structure

```
src/llm_judge/
  scorer.py                    # Deterministic scoring (relevance, clarity, correctness, tone)
  rules/engine.py              # Rule execution engine with deprecation filtering
  rules/lifecycle.py           # Rule governance (list, validate, aging, audit, export)
  eval/run.py                  # Evaluation runner with smoke test + streaming progress
  eval/diff.py                 # Diff engine (flips, deltas, policy check)
  eval/baseline.py             # Baseline management (snapshots, promotion)
  eval/drift.py                # Drift detection + causation + response lifecycle
  eval/event_registry.py       # Cross-capability event registry (6 typed events)
  eval/registry.py             # Run registry (append-only event log)
  eval/metrics.py              # Metrics computation (Kappa, F1, confusion)
  datasets/registry.py         # Dataset resolution + hash verification + security scanning
  datasets/cli.py              # Dataset CLI (list, validate, inspect)
  dataset_validator.py         # Dataset validation (schema, integrity, case_id, security)
  calibration/__init__.py      # Judge registry, calibration pipeline, trust gate
  calibration/bias.py          # Position/length bias detection, DebiasedJudge
  calibration/adjudication.py  # Confidence router, human adjudication queue
  calibration/prompts.py       # Versioned adjudication prompt management
  calibration/hallucination.py # Hallucination detection (grounding, claims, citations)
  calibration/testgen.py       # Test case generation (template, document, export)
  calibration/feedback.py      # Feedback loop (disagreement analysis, recommendations)
```

## Adding a New Rule

1. Create rule module in `src/llm_judge/rules/` (see `quality/nonsense_basic.py` for template)
2. Use `@register("category.rule_name")` decorator
3. Add entry to `rules/manifest.yaml` (version, owner, status, introduced, last_reviewed, review_period_days)
4. Add to rule plan config: `configs/rules/{rubric_id}_{version}.yaml`
5. Run `make rules-validate` to verify manifest alignment

## Registering an LLM Judge (L4)

1. Add judge config to `configs/judges/registry.yaml`
2. Create versioned prompt in `configs/prompts/{rubric_id}/v{N}.yaml`
3. Run calibration against golden dataset
4. Verify trust gate passes: `make judge-status`
5. Set `JUDGE_ENGINE=llm` to enable LLM-based evaluation

## Configuration

- **RunSpecs:** `configs/runspecs/` — evaluation run configurations
- **Rule Plans:** `configs/rules/` — per-rubric rule plan configs
- **Policies:** `configs/policies/` — baseline promotion and drift policies
- **Rubrics:** `rubrics/` — rubric definitions with dimensions and decision policy
- **Datasets:** `datasets/` — governed datasets with `dataset.yaml` metadata
- **Judges:** `configs/judges/` — LLM judge registry with calibration thresholds
- **Prompts:** `configs/prompts/` — versioned adjudication prompt templates
