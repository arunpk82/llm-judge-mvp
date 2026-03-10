Vision: Build a Netflix-grade LLM-as-a-Judge Evaluation Platform
1️⃣ Current Platform Status (Stable & Green)
✅ Deterministic Evaluation Layer

CorrectnessBasicRule implemented

math_basic detector:

Safe AST evaluation (no eval)

Supports: + - * / % **

Deterministic numeric comparison (tolerance-based)

Returns Optional[CorrectnessSignal]

Rule emits:

RuleResult.flags

Backward-compatible mutation into ctx.flags

No false positives for correct arithmetic

Non-math prompts return None

✅ Deterministic Dataset Infrastructure

Folder:

datasets/deterministic/
  math_basic_v1.jsonl

Generator:

tools/generate_deterministic_math_dataset.py

Seeded dataset generation

1000 rows generated

CI sampling: 300 rows

Full dataset ready for nightly expansion

Regression test:

tests/unit/test_deterministic_dataset_math_basic.py

Design principles:

Rule-level validation (not scorer-level)

Supports both ctx-mutation and RuleResult return contracts

Deterministic + reproducible

✅ Rule Engine & Policy Layer

YAML plan-based rule loading

Registry-backed rule resolution

Scorer policy:

Flag-first evaluation

Hard-fail on correctness flags

Severity-aware quality handling

No network calls in deterministic tests

LLMJudge test suite uses httpx.Client mocking

✅ Coverage Status

Total coverage: ~88%

math_basic detector: ~72% (expanded logic now present)

scorer.py: ~83%

engine.py: ~87%

Target: 90%+ deterministic coverage

2️⃣ Architectural Model

Evaluation stack layers:

Detectors
  ↓
Rules (RuleResult / ctx.flags)
  ↓
Rule Engine (plan execution)
  ↓
Scorer Policy (decision logic)
  ↓
LLMJudge (network execution layer)

Separation of concerns enforced:

Dataset regression tests validate rules

Engine tests validate plan wiring

Scorer tests validate policy

LLMJudge tests validate network integration contract

3️⃣ Current Capabilities

✔ Deterministic arithmetic correctness
✔ Repetition detection
✔ Nonsense detection
✔ Definition sanity rule
✔ Plan-based rule activation
✔ Reproducible dataset testing
✔ CI sampling strategy

4️⃣ Known Gaps / Next Steps
Immediate

Raise coverage to 90%+

Expand math_basic dataset to 5K–10K

Add definition_sanity_v1.jsonl generator

Introduce dataset _index.json manifest

Mid-Term

Nightly CI full dataset runs

Mutation-based adversarial dataset generation

Baseline regression metrics snapshotting

False positive tracking

Long-Term

Golden dataset governance

Drift detection

Deterministic + LLM hybrid scoring mode

Evaluation metrics dashboard

5️⃣ Deterministic Dataset Strategy

We are moving toward:

datasets/
  deterministic/
    math_basic_v1.jsonl
    definition_sanity_v1.jsonl
    repetition_v1.jsonl
    _index.json

CI Strategy:

PR: sample 300 rows

Nightly: full dataset (1000–10K rows)

Metrics snapshot comparison

6️⃣ Guiding Principles

Deterministic first

No false positives

Reproducibility via seeds

Separation of rule vs policy validation

Scalable test runtime

Explicit contracts between layers

7️⃣ Next Objective (When Starting New Chat)

Goal: Scale deterministic judge to 10K+ cases and reach 90–92% coverage without bloating CI runtime.