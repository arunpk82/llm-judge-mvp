# Full Properties Wave — All 18 Remaining Properties

## What This Delivers

All remaining properties built as independent check functions, shipping **disabled**.
Promotion from PCT-2 to PCT-6 is config-only — edit `property_config.yaml`, no code changes.

## Properties Built

| Cat | Property | Module | Type |
|-----|----------|--------|------|
| 1.4 | Attribution Accuracy | `properties/faithfulness_advanced.py` | Embedding-based |
| 1.5 | Fabrication Detection | `properties/faithfulness_advanced.py` | Embedding-based |
| 2.5 | Completeness | `configs/prompts/chat_quality/v2.yaml` | Prompt dimension |
| 2.6 | Coherence | `configs/prompts/chat_quality/v2.yaml` | Prompt dimension |
| 2.7 | Depth/Nuance | `configs/prompts/chat_quality/v2.yaml` | Prompt dimension |
| 3.1 | Toxicity & Bias | `properties/safety.py` | Deterministic |
| 3.2 | Instruction Boundary | `properties/safety.py` | Deterministic |
| 3.3 | PII & Data Leakage | `properties/safety.py` | Deterministic |
| 4.1 | Instruction Following | `properties/task_fidelity.py` | Deterministic |
| 4.2 | Format & Structure | `properties/task_fidelity.py` | Deterministic |
| 5.3 | Self-Preference Bias | `properties/robustness.py` | Calibration |
| 5.4 | Consistency | `properties/robustness.py` | Calibration |
| 5.5 | Adversarial Resilience | `properties/robustness.py` | Calibration |
| 5.6 | Edge Case Handling | `properties/robustness.py` | Calibration |
| 5.7 | Reproducibility | `properties/robustness.py` | Calibration |
| 6.1 | Latency & Cost | `properties/performance.py` | Measurement |
| 6.3 | Explainability | `properties/performance.py` | Post-eval check |
| 6.4 | Judge Reasoning Fidelity | `properties/performance.py` | Post-eval check |

## File Manifest

### New Files
- `src/llm_judge/properties/__init__.py` — EmbeddingProvider interface + fallback
- `src/llm_judge/properties/safety.py` — Cat 3 (toxicity, boundary, PII)
- `src/llm_judge/properties/task_fidelity.py` — Cat 4 (instruction following, format)
- `src/llm_judge/properties/faithfulness_advanced.py` — Cat 1 (attribution, fabrication)
- `src/llm_judge/properties/robustness.py` — Cat 5 (5.3–5.7 calibration diagnostics)
- `src/llm_judge/properties/performance.py` — Cat 6 (latency, explainability, fidelity)
- `configs/prompts/chat_quality/v2.yaml` — 7-dimension prompt (v1 preserved)
- `tests/unit/test_properties_full.py` — 35 tests

## Application Instructions

```bash
# After PCT-1 is merged to master
make git-start BRANCH=feature/pct-full-properties

# Apply
unzip pct_full_wave.zip -d .    # press A

# Verify
make preflight

# Ship
make git-ship MSG="feat: all 18 remaining properties built (disabled, config-promoted)"

# After CI passes
make git-merge
```

## Promoting Properties (Config Only)

To promote to PCT-2 (enable completeness, coherence, depth):
```yaml
# In configs/properties/property_config.yaml
completeness:
  enabled: true
coherence:
  enabled: true
depth_nuance:
  enabled: true
```

To promote to PCT-3 (enable safety + task fidelity):
```yaml
toxicity_bias:
  enabled: true
instruction_boundary:
  enabled: true
pii_data_leakage:
  enabled: true
instruction_following:
  enabled: true
format_structure:
  enabled: true
```

Each promotion is: edit YAML → `make git-ship` → done.

## Embedding Models

Properties 1.4 and 1.5 use embedding distance. When `sentence-transformers`
is installed, they use all-MiniLM-L6-v2. Without it, they fall back to
token overlap (degraded accuracy, logged as WARNING).

```bash
pip install sentence-transformers    # optional, enables embedding-based checks
```

## Dependencies

- No new required dependencies — everything uses stdlib or existing packages
- `sentence-transformers` is optional (graceful fallback)
- v2 prompt is a new YAML file — v1 preserved, immutable
