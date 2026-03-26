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
- `reports/runs/pr-gate-<timestamp>/metrics.json` — Kappa, F1, accuracy
- `reports/runs/pr-gate-<timestamp>/manifest.json` — run metadata
- `reports/runs/pr-gate-<timestamp>/validation_report.json` — pre-flight checks

### 3. Compare against baseline

```bash
make diff
```

Shows decision flips, score deltas, and metric changes vs the golden baseline.

### 4. Run the full preflight

```bash
make preflight
```

Chains: lint → typecheck → test → rules-validate → baseline-validate → pr-gate → baseline-dry-run → registry-list → drift-check.

## Daily Workflow

```bash
make eval          # pr-gate + baseline-dry-run + registry-list
make preflight     # full governance check before pushing
```

## Key Commands

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
| `make rules-list` | List governed rules |
| `make rules-validate` | Validate rule manifest alignment |

## Project Structure

```
src/llm_judge/
  scorer.py              # Deterministic scoring (relevance, clarity, correctness, tone)
  rules/engine.py        # Rule execution engine
  rules/lifecycle.py     # Rule governance (list, validate, export)
  eval/run.py            # Evaluation runner
  eval/diff.py           # Diff engine (flips, deltas, policy check)
  eval/baseline.py       # Baseline management (snapshots, promotion)
  eval/drift.py          # Drift detection (point + trend)
  eval/registry.py       # Run registry (append-only event log)
  eval/metrics.py        # Metrics computation (Kappa, F1, confusion)
  datasets/registry.py   # Dataset resolution + hash verification
  dataset_validator.py   # Dataset validation (schema, integrity, case_id)
```

## Adding a New Rule

1. Create rule module in `src/llm_judge/rules/` (see `quality/nonsense_basic.py` for template)
2. Use `@register("category.rule_name")` decorator
3. Add entry to `rules/manifest.yaml` (version, owner, status, introduced)
4. Add to rule plan config: `configs/rules/{rubric_id}_{version}.yaml`
5. Run `make rules-validate` to verify manifest alignment

## Configuration

- **RunSpecs:** `configs/runspecs/` — evaluation run configurations
- **Rule Plans:** `configs/rules/` — per-rubric rule plan configs
- **Policies:** `configs/policies/` — baseline promotion and drift policies
- **Rubrics:** `rubrics/` — rubric definitions with dimensions and decision policy
- **Datasets:** `datasets/` — governed datasets with `dataset.yaml` metadata
