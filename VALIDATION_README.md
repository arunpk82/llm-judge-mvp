# Layer 2 Validation — Platform vs Human Expert

## What This Is

Your platform has 109+ tests proving the **code works correctly** (Layer 1).
This validation proves the **platform's judgments match expert human judgment** (Layer 2).

The difference: Layer 1 tests that `score_candidate()` runs without errors.
Layer 2 tests that `score_candidate()` gives the RIGHT scores.

## Quick Start

```bash
# 1. Copy files to your repo
cp validate_platform.py tools/
mkdir -p datasets/validation/
cp cs_validation_scored.jsonl datasets/validation/

# 2. Run the validation
python tools/validate_platform.py --dataset datasets/validation/cs_validation_scored.jsonl

# 3. View results
cat reports/validation/validation_report.json | python -m json.tool
```

## Adding to Makefile

```makefile
validate:
	python tools/validate_platform.py --dataset datasets/validation/cs_validation_scored.jsonl
```

Then: `make validate`

## Reading the Report

The report shows:

- **Decision agreement**: how often platform and human agree on pass/fail
- **Per-dimension scores**: how close platform scores are to human scores (within ±1)
- **Property lifecycle**: which dimensions to trust (auto-gate) vs review (human-gate)
- **Worst disagreements**: specific cases where platform and human disagree by 3+ points

## Exit Codes

- `0`: All dimensions auto-gated or calibrating — platform can be trusted with monitoring
- `2`: One or more dimensions need human-gating — don't trust platform decision without review

## Improving the Platform

After running validation, you'll see exactly which dimensions and cases disagree.

To improve:
1. Look at the worst disagreements — what pattern do they share?
2. Fix the scoring heuristics or rules
3. Re-run `make validate`
4. Compare the new report to the old one — did agreement improve?

This is the improvement loop: measure → understand → fix → re-measure.

## Adding More Test Cases

1. Create new cases in the JSONL format (see existing cases for schema)
2. Score them yourself: relevance, clarity, correctness, tone (1-5), pass/fail, rationale
3. Append to `datasets/validation/cs_validation_scored.jsonl`
4. Re-run validation

More cases = more reliable agreement percentages.
Target: 100+ cases for statistically meaningful per-dimension results.

## File Locations

```
tools/validate_platform.py              # The validation runner
datasets/validation/cs_validation_scored.jsonl  # Your scored test cases (30 cases)
reports/validation/validation_results.json      # Detailed per-case comparison
reports/validation/validation_report.json       # Aggregated analysis + recommendations
```
