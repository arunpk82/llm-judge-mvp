# Getting Started with LLM-Judge

**Goal: Run your first evaluation in under 30 minutes.**

## 1. Install (5 min)

```bash
git clone https://github.com/arunpk82/llm-judge-mvp.git
cd llm-judge-mvp
poetry install --no-interaction
```

## 2. Run your first evaluation (5 min)

```bash
make pr-gate
```

This runs the PR gate evaluation. You'll see streaming progress and a summary. Output goes to `reports/runs/pr-gate-<timestamp>/`.

Check the results:

```bash
cat reports/runs/pr-gate-*/metrics.json | python -m json.tool
```

## 3. Compare against the baseline (2 min)

```bash
make diff
```

This shows decision flips, metric deltas, and whether the run meets the baseline quality bar.

## 4. Run the full governance check (10 min)

```bash
make preflight
```

This chains: lint → typecheck → test → rules-validate → baseline-validate → pr-gate → baseline-dry-run → registry-list → drift-check. If it passes, your code is ready to ship.

## 5. Explore the system

```bash
make rules-list           # See all governed rules with aging status
make rules-aging          # Check which rules need review
make registry-list        # View recent evaluation runs
make registry-trend       # See metric trends over time
make drift-check          # Check for quality drift
make dataset-list         # List registered datasets
```

## Key concepts

- **Rubric**: Defines what "good" looks like (dimensions, weights, decision policy)
- **Rule**: A deterministic check (correctness, quality, nonsense detection)
- **Baseline**: A snapshot of known-good evaluation results
- **Diff**: Comparison between a new run and the baseline
- **Drift**: Quality degradation detected over time

## Project structure

```
src/llm_judge/           # Source code
  scorer.py              # Deterministic scoring engine
  rules/                 # Rule engine, lifecycle, registry
  eval/                  # Evaluation runner, diff, baseline, drift, events
  datasets/              # Dataset registry and validation
configs/                 # Configuration files
  runspecs/              # Evaluation run configurations
  rules/                 # Per-rubric rule plans
  policies/              # Baseline promotion and drift policies
rubrics/                 # Rubric definitions
datasets/                # Governed datasets
baselines/               # Golden baselines (immutable snapshots)
reports/                 # Run outputs, registries, drift reports
tests/                   # Unit and integration tests
docs/                    # Documentation
```

## Daily workflow

```bash
make eval                # Run evaluation + governance check + registry
make preflight           # Full check before shipping code
make git-ship MSG="..."  # Ship: preflight → commit → push → PR
```

## Need help?

- `make help` — list all available commands
- `docs/DEV_GUIDE.md` — detailed developer guide
- `docs/RUNBOOK.md` — troubleshooting guide
- `docs/API.md` — API reference
