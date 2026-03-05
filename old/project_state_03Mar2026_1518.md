# LLM-Judge Platform — Project State (Updated)

Vision: Build a Netflix-grade LLM-as-a-Judge Evaluation Platform

---

## 1️⃣ Current Platform Status (Stable & Green)

### ✅ Deterministic Evaluation Layer
- Deterministic judge engine in place (no network calls required for deterministic flows).
- Correctness + Quality rule stack supported via registry + plan execution.

### ✅ Deterministic Dataset Infrastructure
- Datasets are versioned and stored under:
  - `datasets/deterministic/`
- Deterministic dataset testing strategy:
  - PR lane: sampled subset
  - Nightly lane: full dataset runs (opt-in via env / marker)
- `case_id` is now a first-class schema requirement for deterministic datasets (enforced by tests).

### ✅ Rubric / Registry Model (Now Multi-Rubric Ready)
- Rubrics are versioned and resolved via registry.
- `math_basic` rubric is now explicitly registered and pinned to v1:
  - `rubric_id: math_basic`, `version: v1`
  - `rules: [correctness_basic]`
  - Rationale: math-only rubric avoids relevance/nonsense/tone gates flipping correct arithmetic answers.

Registry: `rubrics/registry.yaml` maintains latest mappings.

### ✅ Eval Governance (PR Gate + Nightly Deep Lane)
- A PR-gate runspec exists:
  - `configs/runspecs/pr_gate.yaml`
  - Runs deterministic judge on `datasets/golden/v1.jsonl` with `rubric_id: chat_quality`.
- CI workflow runs:
  1) ruff
  2) mypy
  3) pytest (non-nightly)
  4) eval run (PR gate)
  5) eval diff vs baseline (decision flip + metric drop thresholds)
  6) uploads eval artifacts

- Nightly workflow runs:
  1) pytest with `-m nightly`
  2) eval run
  3) eval diff vs baseline
  4) uploads eval artifacts

### ✅ Baselines (Regression Control Plane)
- Baselines are promoted from run directories (run output → baseline snapshot).
- Latest baseline pointer is maintained (e.g., `baselines/.../latest.json`).
- Baseline create CLI is in place to operationalize promotion (so governance isn’t manual).

---

## 2️⃣ Architectural Model (Layered, Auditable)

Evaluation stack layers:
Detectors
  ↓
Rules (RuleResult / ctx.flags)
  ↓
Rule Engine (plan execution)
  ↓
Scorer Policy (decision logic)
  ↓
Eval Runner (runspec → outputs)
  ↓
Diff + Baseline Governance

Separation of concerns is enforced:
- Dataset regression validates rules deterministically
- Engine tests validate wiring and plan execution
- Scorer tests validate policy semantics
- CI/Nightly validate governance via baselines + diff gates

---

## 3️⃣ Current Capabilities (What works today)
✔ Deterministic arithmetic correctness evaluation via math_basic rubric
✔ Repetition and nonsense quality checks (as part of chat_quality rubric)
✔ Plan-based rule activation
✔ Reproducible eval runs (seeded)
✔ Eval artifacts for auditability (run outputs + diffs)
✔ PR-fast vs Nightly-deep split for scale without slowing PR velocity

---

## 4️⃣ Why coverage “drops” when running a single test file (Local Dev Note)
- Running only `tests/deterministic/test_deterministic_datasets.py` executes a tiny subset of the suite.
- Coverage is computed over the entire codebase; fewer tests hit fewer lines → percentage drops.
- In CI, the full intended test set runs (non-nightly for PR; nightly for deep lane), so the coverage gate remains meaningful.

---

## 5️⃣ Deterministic Dataset Strategy (Scale Plan)
Target structure:
datasets/
  deterministic/
    math_basic_v1.jsonl
    definition_sanity_v1.jsonl
    repetition_v1.jsonl
    _index.json

CI Strategy:
- PR: sample subset (e.g., 300 rows per dataset)
- Nightly: full datasets (1K → 10K+ rows)
- Baseline snapshot comparison via eval diff

Logging principle:
- Always print run metadata in deterministic dataset tests:
  - dataset_id, total_rows, sample_size, seed, failing case_id
  - (so failures are triageable without reproduction pain)

---

## 6️⃣ What’s Completed vs Pending (Aligned to the Roadmap)
### Completed (Needle moved)
- Deterministic judge + rule engine stabilized
- Deterministic datasets wired with schema enforcement (case_id)
- math_basic rubric v1 introduced and registry updated
- PR gate evaluation runner wired with baseline diff checks
- Nightly workflow introduced for deep regression lane
- Baseline promotion pathway established

### Pending (Next needle-mover)
1) Scale deterministic datasets to 10K+ while keeping PR fast
2) Expand beyond math_basic:
   - definition_sanity dataset generator + rubric coverage
   - repetition dataset generator + rubric coverage
3) Governance hardening:
   - dataset _index.json as manifest-of-manifests
   - stricter drift budgets (flag hit-rate + latency budgets)
4) Developer experience:
   - standard “how to reproduce CI failures locally” playbook
   - consistent artifact naming + retention guidance

---

## 7️⃣ Next Objective (Start of Next Chat / Next PR)
Goal: Scale deterministic judge to 10K+ cases while keeping PR CI fast.

Deliverables:
1) Eval run → results.jsonl + metrics.json + manifest.json
2) Eval diff → regression report (decision flips + metric deltas)
3) CI split: PR non-nightly; nightly full dataset runs + baseline comparison