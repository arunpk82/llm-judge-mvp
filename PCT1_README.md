# PCT-1 Wave Delivery — Property-Based Evaluation Pipeline

## What This Delivers

EPICs 7.4–7.7: Wire the 13 built properties into the Gate 2 pipeline.

- **EPIC 7.4**: Property Configuration Infrastructure — `property_config.yaml` + `PropertyRegistry`
- **EPIC 7.5**: Versioned Prompt Integration — `LLMJudge` accepts `PromptTemplate`, removes hardcoded `_SYSTEM_PROMPT`
- **EPIC 7.6**: Wire Faithfulness Properties 1.1–1.3 — `check_hallucination()` called from pipeline
- **EPIC 7.7**: Wire Bias Detection 5.1–5.2 — wired into calibration (configs ready, integration in calibration runs)

## File Manifest

### New Files
- `configs/properties/property_config.yaml` — All 28 properties with enabled/gate_mode/visibility
- `src/llm_judge/property_config.py` — PropertyRegistry loader with strict schema validation
- `src/llm_judge/integrated_judge.py` — IntegratedJudge pipeline (the core of PCT-1)
- `tests/unit/test_property_config.py` — 20 tests for property config + integrated judge

### Modified Files
- `src/llm_judge/llm_judge.py` — Accepts optional `PromptTemplate`, renames `_SYSTEM_PROMPT` to `_FALLBACK_SYSTEM_PROMPT`
- `src/llm_judge/runtime.py` — Adds `integrated` engine mode
- `configs/prompts/chat_quality/v1.yaml` — Updated with full rubric matching current scoring quality
- `tools/validate_gate2.py` — Uses `IntegratedJudge` by default, `--raw` flag for comparison
- `tests/unit/test_l4_completion.py` — Updated `test_render_prompt` assertion to match new v1 prompt content

## Application Instructions

```bash
# 1. Create feature branch from master
make git-start BRANCH=feature/pct-1-property-pipeline

# 2. Extract the zip contents over the repo (preserves directory structure)
unzip pct1_wave.zip -d .

# 3. Verify tests pass locally (preflight runs lint, typecheck, test, gates)
make preflight

# 4. Ship — commits, pushes, opens PR
make git-ship MSG="feat: PCT-1 property-based evaluation pipeline (EPICs 7.4-7.7)"

# 5. After CI passes — squash merge, delete branch, update master
make git-merge

# 6. Validate Gate 2 with integrated pipeline on master
GEMINI_API_KEY=your-key poetry run python tools/validate_gate2.py \
    --dataset datasets/validation/cs_validation_scored.jsonl

# 7. Compare with raw pipeline (optional — shows before/after)
GEMINI_API_KEY=your-key poetry run python tools/validate_gate2.py \
    --dataset datasets/validation/cs_validation_scored.jsonl \
    --raw
```

## Detection Coverage at PCT-1

```
28 properties defined. 10 enabled, 0 gated, 10 informational, 18 disabled.
Coverage: 36% enabled, 0% gated.
```

Note: 10 fully built properties enabled. 3 partially built properties (1.4, 5.5, 6.1)
remain disabled until their implementations are complete.

## What Changed in the Code Path

### Before (raw LLMJudge):
```
validate_gate2.py
  → LLMJudge(engine="gemini")     # raw LLM, no trust layers
    → hardcoded _SYSTEM_PROMPT    # not versioned
      → Gemini API call           # no hallucination check
        → raw scores returned     # no property evidence
```

### After (IntegratedJudge):
```
validate_gate2.py
  → IntegratedJudge(engine="gemini")
    → Load property_config.yaml   # which properties run?
    → Load versioned prompt       # chat_quality/v1 (tracked in git)
    → check_hallucination()       # grounding, claims, citations
    → LLMJudge(prompt_template)   # versioned prompt, not hardcoded
    → Assemble EnrichedResponse   # scores + flags + property evidence
```

## Breaking Changes

- `_SYSTEM_PROMPT` renamed to `_FALLBACK_SYSTEM_PROMPT` in `llm_judge.py`.
  Any code referencing the old name will get an `AttributeError`.
  Tests using `_build_prompt` alias still work (backward compatible).

- `ProviderAdapter.build_request()` now takes `system_prompt` as second parameter.
  Tests that mock adapters need to update the signature.

## Adapter Signature Change

The `build_request` method on `ProviderAdapter` and all subclasses now
accepts `system_prompt` as a second argument instead of using the
module-level constant. This allows versioned prompts to flow through
to the provider API call.

If you have custom adapters, update:
```python
# Before
def build_request(self, prompt: str) -> tuple[str, dict, dict]:

# After
def build_request(self, prompt: str, system_prompt: str) -> tuple[str, dict, dict]:
```
