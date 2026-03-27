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
   - Security warning: injection patterns detected — review flagged rows
   - Smoke test failure: pass rate below threshold — check dataset/rubric compatibility
3. Check `validation_report.json` in the run directory for pre-flight details

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
2. Review `correlated_events` section — governance events near the drift
3. Check `response_classification` — which violations are block vs warn vs log
4. If a drift issue was created, track it through the lifecycle:
   - Check `reports/drift/drift_issues.jsonl` for the issue state
   - Investigate correlated events (rule changes, dataset changes)
   - Transition the issue: detected → triaged → responding → resolved
5. Compare against recent runs: `make registry-trend`

### Heartbeat alert (no recent evaluations)

**Symptoms:** Drift check reports "Heartbeat: No eval_run events found."

**Steps:**
1. Verify the evaluation pipeline is running: `make pr-gate`
2. Check `reports/event_registry.jsonl` — is it being written to?
3. If pipeline is running but events aren't recorded, check file permissions
4. Adjust `heartbeat_max_hours` in `configs/policies/drift.yaml` if the interval is too tight

### Rule lifecycle validation fails

**Symptoms:** `make rules-validate` reports misalignment.

**Steps:**
1. "Ungoverned rule" — add the rule to `rules/manifest.yaml`
2. "Stale manifest entry" — either implement the rule or remove from manifest
3. After fixing, re-run: `make rules-validate`
4. Check rule aging: `make rules-aging` — review stale rules

### Stale rule detected

**Symptoms:** `make rules-aging` shows rules exceeding review period.

**Steps:**
1. Review the stale rule's logic — is it still valid for current LLM behavior?
2. Update `last_reviewed` date in `rules/manifest.yaml`
3. If the rule is outdated, set `status: deprecated` with `deprecated_at` date
4. Deprecated rules are auto-excluded after the warning period

### LLM judge trust gate blocks evaluation

**Symptoms:** `JUDGE_ENGINE=llm` fails with "Trust gate blocked."

**Steps:**
1. Check judge status: `make judge-status`
2. If "not registered" — add judge to `configs/judges/registry.yaml`
3. If "not calibrated" — run calibration against golden dataset
4. If "blocked" — review why the judge was blocked and recalibrate
5. To temporarily bypass (development only): set `trust_gate.enforce: false` in registry

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
make rules-aging                # Stale rules
make judge-status               # LLM judge calibration
make event-stats                # Event registry health
```

### Investigating a specific run

```bash
make registry-show RUN_ID=pr-gate-20260305-120312  # Run details
make event-trace RUN_ID=pr-gate-20260305-120312     # Full event lineage
```

### Reviewing human adjudication queue

Check `reports/adjudication/queue.jsonl` for pending cases. Resolve cases to feed the calibration feedback loop.

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Continue |
| 1 | Runtime error | Check error message, fix input |
| 2 | Policy violation | Review diff/drift report, fix regression or promote baseline |
