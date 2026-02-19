# Deterministic Judge

## Purpose
The deterministic judge is the **CI-grade quality gate** for llm-judge-mvp.

It is intentionally:
- **Stable** (same input → same output)
- **Fast** (suitable for PR checks)
- **Explainable** (dimension scores + flags + explanations)
- **Rubric-driven** (policy and dimensions come from versioned YAML rubrics)

This judge is not meant to replace LLM-based evaluation. It is the **first lane** in a multi-lane evaluation strategy:

1. **PR Gate (Deterministic)** – catches obvious regressions cheaply and deterministically.
2. **Nightly / Release (LLM Judge)** – deeper semantic evaluation, agreement, calibration, bias slicing.

## How scoring works
Implementation lives in `src/llm_judge/scorer.py`.

### Inputs
- `conversation`: user + assistant messages (used for relevance signals)
- `candidate_answer`: answer being evaluated
- `rubric_id`: rubric ref (`chat_quality` or `chat_quality@v1`)

### Rubric-driven policy
Rubrics define:
- `dimensions`: which dimensions are scored
- `scale`: score range (default 1..5)
- `decision_policy`:
  - `pass_if_overall_score_gte`
  - `fail_if_any_dimension_lte`
- optional `weights`: per-dimension weights for overall score

The deterministic scorer:
- **only emits scores for rubric dimensions**
- clamps each score to `scale.min..scale.max`
- computes a **weighted overall score**
- applies the rubric decision policy

### Deterministic signals
Current v1 signals:
- **Relevance**: token overlap (stopword removal + light synonym normalization)
- **Clarity**: length thresholds + structure detection (lists, paragraphs, headings)
- **Tone**: rude language + excessive shouting detection
- **Correctness (proxy)**: conservative heuristic (uncertainty markers)

### Flags
Flags are designed to be machine-readable for trend tracking and gating.
Examples:
- `low_relevance`
- `rude_tone`
- `time_sensitive_unverified` (time-sensitive ask + definitive facts)
- `unknown_rubric`

## Extending the deterministic judge
Add new deterministic capabilities by:
1. Adding a new dimension to a rubric YAML.
2. Adding a deterministic scorer for that dimension in `scorer.py`.
3. Adding unit tests under `tests/unit/`.

The deterministic judge should remain:
- deterministic
- low-cost
- explainable
