# LLM-Judge Operational Runbook

## When Something Goes Wrong

### Evaluation run fails

**Symptoms:** `make pr-gate` exits with error.

**Steps:**
1. Check the error output — pre-flight failures print the specific issue
2. Common causes:
   - Missing rubric: register in `rubrics/registry.yaml`
   - Dataset hash mismatch: dataset file was modified — re-register with correct hash
   - Missing case_id: all dataset rows require `case_id` for deterministic sampling
3. If pre-flight passes but scoring fails, check `validation_report.json` in the run directory

### Baseline diff shows regression

**Symptoms:** `make diff` exits with code 2 (policy violation).

**Steps:**
1. Read `diff/diff_summary.txt` in the run directory
2. Check `diff/policy_result.json` for specific violations
3. Decision flips: review `diff/diff_report.json` → `judgments.decision_flips`
4. Metric drops: check which metrics exceeded tolerance
5. If the regression is expected (intentional rule change), promote a new baseline:
   ```bash
   make baseline-promote
   ```

### Drift alert fires

**Symptoms:** `make drift-check` reports metric drift.

**Steps:**
1. Check `reports/drift/drift_report.json` for details
2. Compare against recent runs: `make registry-trend`
3. Investigate: did rules change? Did dataset change? Did a dependency update?
4. If drift is from a known change, promote a new baseline to reset the comparison point

### Rule lifecycle validation fails

**Symptoms:** `make rules-validate` reports misalignment.

**Steps:**
1. "present in RULE_REGISTRY but missing in manifest" — add the rule to `rules/manifest.yaml`
2. "present in manifest but missing in RULE_REGISTRY" — either implement the rule or remove from manifest
3. After fixing, re-run: `make rules-validate`

## Routine Operations

### Promoting a baseline (after verified improvement)

```bash
make pr-gate                    # Run evaluation
make baseline-dry-run           # Test promotion policy (no writes)
make baseline-promote           # Write new baseline snapshot
```

### Checking system health

```bash
make preflight                  # Full governance check
make registry-list              # Recent runs
make registry-trend             # Metric trends
make drift-check                # Drift detection
```

### Viewing run history

```bash
make registry-list                                  # Last 20 runs
make registry-show RUN_ID=pr-gate-20260305-120312   # Specific run
make registry-trend REG_METRIC=cohen_kappa          # Kappa trend
```

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Continue |
| 1 | Runtime error | Check error message, fix input |
| 2 | Policy violation | Review diff report, decide: fix regression or promote baseline |
