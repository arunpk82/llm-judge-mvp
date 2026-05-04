# Layer 1 · External Contract · v3.3.1 (Active)

**Prepared by architect chat · 2026-04-29 · ships with L1-Pkt-A PR per Decision 6 · revised 2026-04-30 to align with as-shipped code**

**Status:** Active (v3.3.1). Batches contract impacts from L1-Pkt-1 (CP-F2 closure), L1-Pkt-2 (CP-F14 closure), L1-Pkt-3 (absorbed into L1-Pkt-A — CP-F3 closure), L1-Pkt-A (CP-F1 closure). Per L1-Pkt-A architectural Decision 6: contract update ships with packet rather than queueing for v4. v3.3.1 is the as-shipped revision; supersedes the v3.3 architect-chat draft after Layer 1 chat review and Pre-flight 6 recon corrections to §5.5.

**Recon baseline.** All normative claims cite the gap analysis (`layer-1-control-state-and-gaps-v1.2.md` — pending publication) and its underlying Pass 4 recon at branch `master`, commit `8e574304`. Phase 1 v2 (`feat/packet-phase-1-l1-completion`) has landed; the prior baseline `e739612` is no longer current. Where v3 makes a claim, the source is named: gap analysis property ID (e.g., A1.1) or code reference (e.g., `runner.py:127`).

**Purpose.** Defines what external callers (CLI tools, future API consumers, batch input file producers) can rely on when invoking the LLM-Judge platform via Layer 1 (Control). Establishes the stable surface that Foundation Packet 1 (closing CP-F1) and subsequent packets anchor against.

**Scope.** This contract covers Layer 1's external-facing surface only. Two companion contracts (forthcoming) cover Layer 1 ↔ Layer 2-6 internal contract and Layer 1 ↔ Layer 7-8 contract. Findings that belong in those contracts are explicitly excluded here (notable exclusions: CP-F3 envelope field ownership; B1.4 authorized envelope construction; admission-time governance preflight design — see §5.3).

**Mechanism class vocabulary (used throughout).** Per the gap analysis closing observation: claims are annotated with current mechanism class and target mechanism class.

| Class | Meaning | Example |
|---|---|---|
| **convention** | Stated as a rule; enforced by code review or social discipline. Nothing structural prevents violation. | Today's "do not call `check_hallucination` directly" |
| **runtime gate** | A check executed at runtime that rejects violations. Enforcement is real but conditional on the gate being on the path. | Pydantic validation at request admission |
| **structural** | The violation is impossible by construction. Type system, module privacy, or factory-only construction makes the wrong thing unreachable. | Pydantic `frozen=True` makes mutation impossible |

The closure escalator across the contract is: convention → runtime gate → structural. Per-claim annotation states **current** class and **target** class with closure-vehicle reference.

---

## 1.0 · Caller universe

**Today: internal use only.** The platform's callers today are (a) architect chat working sessions, (b) execution chat ship cycles, and (c) demonstration runs. There are no external callers — no third-party API consumers, no multi-tenant deployments, no production traffic.

**Tomorrow: external API consumers, multi-tenant deployments.** This contract anticipates the latter while documenting current behavior. Sections 7.3 (forward-compat expectations on callers) and 7.4 (backward-compat expectations from platform) are bilateral commitments. Today they have no counterparty; the contract codifies the platform-side commitment so that when external callers exist, the contract is already stated.

**Why this matters.** Several contract clauses (e.g., §4.5 BatchRunner-specific boundary, §8.1 operational guardrails) describe gaps that are low-severity today and high-severity once external callers exist. Anchoring "today" to the actual caller population prevents the contract from over-claiming production-grade properties that haven't yet been needed.

---

## 1 · Entry surface

### 1.1 Stable entry points

Layer 1 exposes two stable entry points to external callers. Signatures below are quoted from the recon baseline (commit `8e574304`).

**Single-evaluation entry — `PlatformRunner.run_single_evaluation`** (per `runner.py:127`, gap analysis Axis A1)

```python
PlatformRunner.run_single_evaluation(
    payload: SingleEvaluationRequest,
    ...
) → SingleEvaluationResult
```

**Batch-evaluation entry — `BatchRunner.run_batch`** (per `batch_runner.py:81-89`)

```python
BatchRunner.run_batch(
    cases: Iterable[SingleEvaluationRequest],
    batch_id: str,
    output_dir: Path,
    source: str,
    *,
    layers: list[str] | None = None,
) → BatchResult
```

`BatchRunner` is a wrapper around `PlatformRunner`: per `batch_runner.py:214`, `_run_one_case` calls `run_single_evaluation` per case.

**Stability rules for both entries:** Adding optional keyword arguments is non-breaking. Removing arguments, changing return type, renaming the method, or changing the meaning of an existing argument is a breaking change requiring a major version bump.

### 1.1a Batch input file is a separate surface

Batch *input file* loading (`.yaml`/`.json`) is a separate surface from `BatchRunner.run_batch`:

- `load_batch_file()` (per `control_plane/batch_input.py:63-102`, gap analysis Axis A1) loads the file and produces a `BatchInputFile`
- `tools/run_batch_evaluation.py::_batch_case_to_request` converts each case in the `BatchInputFile` to a `SingleEvaluationRequest`
- The resulting `Iterable[SingleEvaluationRequest]` is passed to `BatchRunner.run_batch`

So the contract surface for batch invocation is two-stage: file loading produces requests, requests are passed to `BatchRunner`. External callers may use either: load the file themselves and call `BatchRunner.run_batch`, or rely on `tools/run_batch_evaluation.py` as the CLI front-end.

The error-model implication: `BatchInputSchemaError` only escapes from `load_batch_file`, not from `BatchRunner.run_batch`. See §4.2.

### 1.2 Approved invocation paths

External callers may reach Layer 1 through:

1. **Direct Python import (today)** — `from llm_judge.control_plane.runner import PlatformRunner`. **Note:** This commits the platform to in-process Python execution as part of the public surface. If a future version narrows this to CLI-and-API only, the deprecation cycle (§7.2) applies.
2. **CLI** — `tools/run_batch_evaluation.py` and equivalent governed tools
3. **Programmatic batch input loader** — `load_batch_file()` for `.yaml`/`.json` inputs (per §1.1a)

### 1.3 Disallowed invocation paths

External callers must NOT reach the platform through:

1. **Direct capability invocation** — calling `check_hallucination`, `record_evaluation_manifest`, or any wrapper-internal function directly. Per gap analysis A2.2, `check_hallucination` is publicly importable today; the contract reserves the right to make these private symbols.
   - **Mechanism class:** [convention] today → [structural] target. Closure vehicle: A2.2 (separate packet, not Foundation Packet 1).
2. **Direct envelope construction** — constructing `ProvenanceEnvelope` instances outside the canonical `new_envelope` factory. Per gap analysis B1.4 PARTIAL, `ProvenanceEnvelope` is publicly importable; hand-constructed envelopes are not guaranteed to verify.
   - **Mechanism class:** [convention] today → [structural or runtime gate] target. Closure vehicle: B1.4 (separate packet, not Foundation Packet 1).
3. **Bypass paths** — any code path that produces evaluation artifacts without routing through `PlatformRunner` or `BatchRunner`. Per L1-Pkt-A (CP-F1 closure), the benchmark adapter previously a parallel entry has been refactored to load benchmark files and route per-case through `BatchRunner`. Benchmark files themselves are now governed CAP-1 artifacts (see §5.5).
   - **Mechanism class:** [runtime gate]. Closed by L1-Pkt-A. Static repository assertion confirms no parallel-entry function produces evaluation results outside `PlatformRunner` / `BatchRunner` routing.

### 1.3.1 Enforcement state acknowledgment

The clauses in §1.3 are stated as caller obligations. Enforcement state varies by clause:

- **§1.3 #1 (direct capability invocation):** [convention] today. Closure vehicle: A2.2 (future packet).
- **§1.3 #2 (direct envelope construction):** [convention] today. Closure vehicle: B1.4 (future packet).
- **§1.3 #3 (bypass paths):** [runtime gate] post-L1-Pkt-A. Static repository assertion plus Runner-as-single-entry-point property enforced.

Closure roadmap for remaining clauses:
- **Subsequent packets** close A2.2 (no capability-direct invocation) and B1.4 (authorized envelope construction).

The §1.3 #1 and #2 clauses remain aspirational, not enforced, until their closure vehicles ship. Callers reading this contract must understand that violation of these specific clauses is currently possible; the platform asks for caller cooperation as a stopgap. The §1.3 #3 clause is enforced as of L1-Pkt-A merge.

### 1.4 Contract on adding entry points

New entry points may be added in minor versions. Existing entry points may be deprecated (with grace period per §7.2) but not removed in minor versions.

---

## 2 · Request schema contract

### 2.1 SingleEvaluationRequest schema

Per gap analysis A1, the schema is defined at `src/llm_judge/control_plane/types.py`:

```python
class SingleEvaluationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    response: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    rubric_id: str = Field(..., min_length=1, description="Governed rubric identifier")
    rubric_version: str = Field(default="latest", description="Rubric version, or 'latest' for current")
    caller_id: str | None = None
    request_id: str | None = None
    benchmark_reference: BenchmarkReference | None = None  # added in L1-Pkt-A
```

**`benchmark_reference` field (added in L1-Pkt-A):** Optional typed reference to a registered benchmark dataset. Populated by benchmark adapters that route through `BatchRunner`; unset for non-benchmark per-case construction. When populated, envelope provenance records benchmark_id, version, and content_hash (see §6.1). The `BenchmarkReference` type is defined in CAP-1's expansion specification; see §5.5 for the contract surface.

**Backward compatibility:** Optional with default `None` preserves backward compat. Existing per-case construction (without benchmark_reference) continues to work unchanged.

**Bounded sizes — not specified today.** Per gap analysis A1.4 OPEN, no `max_length` constraint exists on `response`, `source`, `caller_id`, `request_id`, or `rubric_id`. A 10MB `response` is admitted today.

- **Mechanism class for size bounds:** none today → [runtime gate] target via Pydantic `max_length` constraints. Closure vehicle: A1.4 (separate packet, **not Foundation Packet 1**).
- **Caller commitment:** Callers must not assume bounds will remain unspecified. When A1.4 closes, requests exceeding the eventual bounds will be rejected. Callers should keep request sizes within reasonable evaluation sizes (current internal usage pattern: response and source under 100KB; identifiers under 256 bytes).

### 2.2 Field stability guarantees

- **Required fields stay required.** Promoting an optional field to required is a breaking change requiring a major version.
- **Required fields are not removed.** Removing a required field is a breaking change.
- **Field types are stable.** Changing a field type (e.g., `str` to `int`) is a breaking change.
- **New optional fields may be added** in minor versions.
- **Validation constraints tighten only via deprecation cycle.** Adding `max_length` (or any other tightening) follows the deprecation policy in §7.2: warn for one full minor-version cycle, then enforce. This protects existing callers who rely on the unspecified-bounds state from silent breakage.

### 2.3 Immutability guarantee

`SingleEvaluationRequest` is a frozen Pydantic model (`model_config = ConfigDict(frozen=True)` per gap analysis A1.3 CLOSED).

- **Mechanism class:** [structural]. Pydantic enforces immutability at construction; mutation attempts raise `ValidationError`.
- **Caller commitment:** Callers must not work around immutability via `__dict__` access or similar. Doing so is a contract violation; the platform may not detect it but relies on immutability for caching, integrity, and audit.

### 2.4 rubric_version "latest" resolution

Per gap analysis A1.2 PARTIAL: `rubric_version` defaults to `"latest"`. The `"latest"` value is resolved at the wrapper boundary against the rubric registry (specifically inside `invoke_cap2` at `wrappers.py:319`, calling `_resolve_effective_version` only when `rubric_version="latest"`). The resolution is silent — no record of "what `latest` resolved to at the moment of execution" appears in the integrity trail today.

- **Mechanism class:** [convention] today (the resolution happens at runtime; nothing prevents it from differing across nearby invocations) → [structural] target via admission-time resolution and recording.
- **Closure shapes (per gap analysis A1.2 and Layer 1 chat verdict on v1):**
  - (a) Require explicit version pin from callers — breaking change opt-in
  - (b) Record the resolved version in the integrity trail — observable but caller still uses `"latest"`
  - (c) **Resolve at admission time and store the resolved version in the request itself** — request becomes self-describing (preferred per Layer 1 chat verdict)
- **Closure vehicle:** Not Foundation Packet 1 (A1.2 is not in F1's locked scope of CP-F1+F2+F3). Separate packet.
- **Caller commitment today:** Callers who require reproducibility must pass an explicit `rubric_version` rather than relying on `"latest"`. Otherwise the resolution result is observable only by inspecting the integrity trail (and even there, the resolution is implicit rather than explicit until A1.2 closes).
- **Error implication:** When `rubric_version="latest"` and `rubric_id` is not in the registry, a plain `ValueError` raises from `rubric_store._resolve_version` — see §4.3(b) and CP-F14.

### 2.5 BatchInputFile schema

Per gap analysis A1, `BatchInputFile` (defined at `control_plane/batch_input.py`) provides equivalent governance for ad-hoc batch input files (`.yaml`/`.json`) loaded via `load_batch_file()`. `BatchInputSchemaError` (defined at `batch_input.py:59`, subclass of `ValueError`) is raised on invalid input.

- **Mechanism class:** [runtime gate]. Schema validated at load time; invalid files rejected.
- Same field stability rules as `SingleEvaluationRequest` (§2.2).

---

## 3 · Response schema contract

### 3.1 SingleEvaluationResult shape

Per `types.py`:

```python
class SingleEvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    verdict: dict[str, Any]
    manifest_id: str
    envelope: ProvenanceEnvelope
    integrity: Integrity
```

- **`verdict`** — the aggregated capability outputs (per §3.4 for current shape)
- **`manifest_id`** — identifier for the CAP-5-written manifest artifact
- **`envelope`** — the ProvenanceEnvelope carrying the integrity trail (per §6)
- **`integrity`** — the Integrity object capturing per-capability outcomes (per §6.1)

`request_id`, `rubric_id`, `rubric_version`, and `schema_version` are **not** top-level fields on the result. `request_id` is carried via the envelope; `rubric_id` and `rubric_version` are carried via the verdict and the envelope's stamped fields; `schema_version` lives on the envelope.

### 3.2 Schema versioning

Schema versioning lives on the envelope, not on the result. Per gap analysis B1.6 PARTIAL: schema version field exists on envelopes, currently `schema_version=3`. A backward-compat validator handles missing schema_version.

### 3.3 Backward-compat rules

- **schema_version is monotonically increasing** on envelopes.
- **Older callers parsing newer responses** must tolerate unknown fields. (Forward-compat property on responses.)
- **Older responses parsed by newer callers** are accepted via backward-compat validators (per `_backfill_legacy_schema` in `envelope.py`).
- **Schema-evolution rules (breaking vs non-breaking)** are not yet documented per gap analysis B1.6 PARTIAL.
  - **Mechanism class:** none today (no documented rules) → [convention] target via documented policy → [runtime gate] target via CI fixture testing.
  - **Closure vehicle:** B1.6 — separate packet, **not Foundation Packet 1**.
  - **Acceptance gating:** This contract should not go Active until at least the [convention]-level closure (documented schema-evolution policy) lands. The backward-compat promise is unverifiable without it. See §10 acceptance criteria (Pre-Active Closure 1).

### 3.4 Verdict dict shape

Per gap analysis C1.4 CLOSED and `runner.py:368-373`, the verdict dict is built as follows:

- When CAP-7 succeeded: `aggregated.update(verdict_from_cap7)` — CAP-7's verdict keys are **inlined** into aggregated, not nested under a `cap7_verdict` key.
- When CAP-2 succeeded: `aggregated["rule_evidence"] = list(rule_hits)` — the key is `rule_evidence`, not `cap2_rule_hits`.

Callers must not assume specific keys are present unconditionally — keys depend on which capabilities succeeded. The integrity trail (§6.1) is the source of truth for which capabilities succeeded.

Capabilities may add new keys to the verdict dict in minor versions. Callers must tolerate unknown keys. Removing keys is a breaking change.

---

## 4 · Error model

The platform produces four categories of error/result-degradation that callers must handle. Categories below are anchored to recon at `8e574304`; per Layer 1 chat verdict on v2, three of v2's category framings were factually wrong and have been replaced verbatim with Layer 1 chat's recon-anchored patches.

### 4.1 Category 1 — Caller-side validation (raised at request construction, NOT at Runner invocation)

Raised by Pydantic when callers construct a malformed `SingleEvaluationRequest`:

- `pydantic.ValidationError` — request schema violations (missing required fields, type mismatches, value constraint violations)

This raises at the caller's `SingleEvaluationRequest(...)` construction site, **NOT** at `run_single_evaluation`. Callers must wrap construction in try/except, not the Runner call.

- **Mechanism class:** [structural] (Pydantic frozen + validation).

### 4.2 Category 2 — Batch input file validation (at load_batch_file boundary, separate from Runner)

Raised by `load_batch_file` (`control_plane/batch_input.py:63-102`):

- `BatchInputSchemaError` (subclass of `ValueError`) — defined at `batch_input.py:59`. Sole declared escape from `load_batch_file`. Wraps `OSError`, `yaml.YAMLError`, `json.JSONDecodeError`, and `pydantic.ValidationError` into one class with `from e` chaining.

This is a separate boundary from `PlatformRunner.run_single_evaluation`. Callers using `load_batch_file` catch `BatchInputSchemaError`; callers using `BatchRunner.run_batch` directly don't see it.

- **Mechanism class:** [runtime gate].

### 4.3 Category 3 — Runner boundary exceptions (CAP-5 propagation only)

Per recon: only the CAP-5 phase propagates exceptions. CAP-1, CAP-2, CAP-7 failures are caught by Runner and recorded as integrity records (see Category 4).

CAP-5 propagation surface (per `runner.py:411-421`, broad `except Exception: ... raise`):

**(a) `MissingProvenanceError`** — `control_plane/types.py:23`. Raised at `wrappers.py:421` when `envelope.integrity` is empty at CAP-5 entry. Message: `"invoke_cap5: envelope.integrity is empty; at least one upstream capability outcome must be recorded before writing a manifest."`
  - **Mechanism class:** [runtime gate]
  - **Caller commitment:** catch by class, not message text.

**(b) `RubricNotInRegistryError`** (subclass of `ValueError`) — `control_plane/types.py`. Raised from `rubric_store._resolve_version` when `rubric_version="latest"` and `rubric_id` not in registry, OR registry's `"latest"` value is empty. Messages contain the literal `rubric_id`. Per L1-Pkt-2 (CP-F14 closure), the previous plain `ValueError` raise was replaced with this domain-specific class for caller type-precision.
  - **Mechanism class:** [runtime gate]. Closed by L1-Pkt-2.
  - **Backward compatibility:** Subclass of `ValueError`. Existing callers catching `ValueError` continue to work; new callers may catch `RubricNotInRegistryError` specifically for type-precise handling.
  - **Caller commitment:** Catch by class, not message text. Migration to specific catch is opportunistic; existing `except ValueError` patterns continue to function.

**(c) `OSError` family** (`FileNotFoundError`, `PermissionError`, etc.) — from CAP-5's manifest write at `eval/cap5_entry.py` and downstream calls (`write_json`, `append_run_registry_entry`, `append_event`). Callers cannot distinguish "evaluation result couldn't be persisted" from "evaluation itself failed."
  - **Mechanism class:** [convention] (raw stdlib)
  - **Closure vehicle:** CP-F16, future packet.

**(d) `pydantic.ValidationError`** — from registry/event model construction inside `append_run_registry_entry` / `append_event`. Reachable transitively. Today's mechanism: passes through unchanged.
  - **Mechanism class:** [convention] (raw third-party)
  - **Closure vehicle:** CP-F16 closure should normalize.

### 4.4 Category 4 — Degraded results (no exception; integrity.complete=False)

**Most common failure mode from caller's perspective.** CAP-1, CAP-2, CAP-7 failures are caught by Runner (per `runner.py:246, 303, 347`) and recorded with `_failure_record`. Runner does not re-raise. Call returns `SingleEvaluationResult` normally, with `integrity.complete=False`.

**Caller commitment:** callers MUST check `integrity.complete` to detect degraded results. The absence of an exception does NOT mean the evaluation succeeded. A caller treating "no exception" as "success" will silently consume incomplete evaluations.

This category is a contract obligation today — Runner already produces these results; callers just need to know to check.

### 4.5 BatchRunner-specific boundary

`BatchRunner.run_batch` (`batch_runner.py:81-202`) catches per-case exceptions in `_run_one_case` (`batch_runner.py:234`, `except Exception`). Per-case exceptions converted to `CaseResult(had_error=True, ...)` and batch continues to next case.

What escapes `BatchRunner.run_batch` (NOT caught by `_run_one_case`):

- `OSError` from `output_dir.mkdir`, `cases_root.mkdir`, `case_dir.mkdir`, `batch_manifest.json` write, `manifest.json` write
- `pydantic.ValidationError` from `CaseResult` / `BatchResult` construction
- `KeyboardInterrupt` / `SystemExit` (intentionally not caught)
- Anything from cases iterable's `__iter__` method

Per **CP-F17**: `BatchRunner` has no protection around its own filesystem operations; mid-batch `OSError` terminates the batch and propagates.

Per **CP-F19**: `BatchRunner` has no resumability or checkpoint. Mid-batch interrupt loses not-yet-processed cases. Defer to Category 5 trigger.

### 4.6 Error class stability

The Layer 1 exception taxonomy lives in `src/llm_judge/control_plane/types.py`:

| Exception class | Subclass of | Raised when | Closure packet |
|---|---|---|---|
| `MissingProvenanceError` | `Exception` | CAP-5 envelope provenance gaps | (existing) |
| `BatchInputSchemaError` | `ValueError` | `load_batch_file` schema violations | (existing) |
| `ConfigurationError` | `Exception` | Production-mode startup without HMAC env var; misaligned layer vocabulary | L1-Pkt-1 (CP-F2 + CP-F11), L1-Pkt-2 (CP-F4) |
| `RubricNotInRegistryError` | `ValueError` | `rubric_store._resolve_version` cannot resolve `rubric_id` | L1-Pkt-2 (CP-F14) |
| `RubricSchemaError` | `ValueError` | Rubric file fails schema validation at load time | L1-Pkt-2 (CP-F15 documentation) |
| `FieldOwnershipViolationError` | `ValueError` | `envelope.stamped()` receives field outside capability's allowlist | L1-Pkt-3 (CP-F3) |
| `BenchmarkFileNotFoundError` | `ValueError` | `cap1.register_benchmark` cannot find file | L1-Pkt-A (CAP-1 expansion) |
| `BenchmarkContentCollisionError` | `ValueError` | Benchmark content hash conflicts with registered version | L1-Pkt-A (CAP-1 expansion) |

**Stability rules:**

- Error categories (Categories 1-4 above) are stable.
- Error class names are stable. None of the classes in the table will be renamed.
- Error message text is **NOT** part of the contract. Callers must not parse error messages programmatically. Use class types and structured fields where available.
- The `ValueError` subclass relationship is preserved for backward compat. Any new domain-specific exception added in the future will subclass `ValueError` if it represents an input/data validation failure, allowing existing `except ValueError` callers to continue functioning.

**What may change:**

- New exception classes may be added in minor versions (additive). Existing callers unaffected.
- `OSError` and `pydantic.ValidationError` propagation may be normalized via CP-F16 closure (future packet).
- The taxonomy may be relocated from `types.py` to a dedicated `errors.py` module via CP-F20 closure (future packet, structural refactor). All class names and import paths preserved through deprecation cycle if relocation happens.

### 4.7 Error sanitization (current state)

Per gap analysis A1.5 PARTIAL, the platform makes a partial sanitization guarantee:

- Pydantic validation errors include field names that may be implementation-internal.
- `MissingProvenanceError` includes implementation-internal field names like "envelope.dataset_registry_id is absent" (Layer 1 chat Pass 4 confirmed messages embed envelope field names verbatim).
- Capability internal errors may leak file paths, internal state names, or stack details. Truncation to 500 chars happens for integrity-trail messages, but boundary errors are not formatted for sanitization.

- **Mechanism class:** [convention] today (no formal sanitization layer) → [runtime gate] target via boundary-error formatter.
- **Closure vehicle:** A1.5 — separate packet, **not Foundation Packet 1**.
- **Caller commitment:** Callers must not assume error messages are safe to display to end users without sanitization on their side.

---

## 5 · Governance fields

### 5.1 Required governance identifiers

Every request must specify:
- `rubric_id` — the rubric to evaluate against (required, `min_length=1`)

### 5.2 Optional governance identifiers

- `rubric_version` — version pin, default `"latest"` (see §2.4)
- `caller_id` — caller identification for audit (optional)
- `request_id` — caller-supplied identifier for cross-system correlation (optional; the platform generates one if not provided)

### 5.3 Governance preflight behavior — actual current state

Per Pass 4 recon: **Layer 1 does NOT perform admission-time governance preflight.** The Runner does not invoke any rubric existence check, status check, or lifecycle validation before CAP-1 begins. The first registry touch is inside `invoke_cap2` at `wrappers.py:319`, calling `_resolve_effective_version` only when `rubric_version="latest"`.

What actually happens:

**(a) Unknown `rubric_id` with `rubric_version="latest"`:** `RubricNotInRegistryError` (subclass of `ValueError`) from `rubric_store._resolve_version`, surfaced via CAP-5 propagation (CAP-2 also raises but is caught by Runner). Per L1-Pkt-2 (CP-F14 closure); see §4.3(b).

**(b) Ungoverned rubric** (status not validated/production, missing lifecycle fields): **NOT CHECKED on Layer 1 path.** `UngovernedRubricError` exists at `rubrics/lifecycle.py:80` with 12 raise sites, but only reached from `eval/run.py:173` (legacy CLI eval driver). PlatformRunner never reaches it. (CP-F18)

**(c) F11 (prompt artifact): NOT CHECKED.** `check_rubrics_governed` enforces 7 criteria (registry latest entry, file exists, schema validates, status validated/production, last_reviewed within REVIEW_PERIOD_DAYS, dimensions non-empty, metrics_schema.required declared). No prompt artifact check.

**Caller commitment:** callers must not assume governance preflight has occurred. Submitting an ungoverned `rubric_id` does not raise `UngovernedRubricError` today — it either succeeds (rubric exists in registry.latest) or raises plain `ValueError` (it doesn't).

- **Mechanism class:** [convention] today (rubric resolution is implicit, lazy, no domain-specific exceptions, no lifecycle validation) → [runtime gate] target via admission-time preflight that calls `check_rubrics_governed` before CAP-1 begins.
- **Closure vehicle:** combination of CP-F14 (domain-specific exceptions), CP-F18 (wire `check_rubrics_governed` into Layer 1), F11 (prompt artifact check). Not in Foundation Packet 1's locked scope.

The forthcoming **Layer 1 ↔ Layer 7-8 contract** specifies what Layer 1 will eventually expect from Layer 7's governance preflight signal. **That preflight does not exist on the Runner path today.**

### 5.4 Default behavior changes

The contract reserves the right to add governance preflight in a future minor version, with the deprecation cycle (§7.2) applied:

- Adding admission-time governance preflight (per CP-F18 closure) follows the deprecation cycle: warn for one minor version cycle on rubrics that would fail preflight, then enforce.
- Adding new preflight checks (e.g., F11 prompt artifact validation, once preflight exists) follows the same cycle.
- Announcement venue: the contract document's audit trail (§12) plus a row in the platform CHANGELOG (when one exists; today, the audit trail is the only authoritative venue).

### 5.5 Benchmark dataset registration

Per L1-Pkt-A (CAP-1 expansion): benchmark datasets used for evaluation are registered with CAP-1 governance before any per-case work proceeds. The batch driver (`tools/run_batch_evaluation.py`) calls `llm_judge.datasets.benchmark_registry.register_benchmark(file_path) → BenchmarkReference` at adapter entry for benchmarks with a JSON definition (today: `ragtruth_50`, `ragtruth_5`); the returned reference flows through `SingleEvaluationRequest.benchmark_reference` (per §2.1) and is recorded in envelope provenance (per §6.1). Adapters without a JSON definition (e.g., `halueval`, `fever`, `ifeval`, `toxigen`, `faithdial`, `jigsaw`) load directly from raw data files and leave `benchmark_reference=None`.

**`BenchmarkReference` shape** (per as-shipped code at `src/llm_judge/control_plane/types.py`; supersedes the architect-draft v3.3 shape after Pre-flight 6 recon correction — `version` → `benchmark_version`, `content_hash` → `benchmark_content_hash`, `registration_timestamp` → `benchmark_registration_timestamp`, `file_path` dropped because the JSON definition file is locatable from `benchmark_id` via the per-dataset sidecar convention):

```python
class BenchmarkReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    benchmark_id: str = Field(..., min_length=1)
    benchmark_version: str = Field(..., min_length=1)
    benchmark_content_hash: str = Field(..., min_length=1)
    benchmark_registration_timestamp: datetime
```

**Storage** — registration records live next to the benchmark JSON definition as a sidecar file (`<benchmark_id>_registration.json`). Per Pre-flight 4 recon: the unit of registration is the benchmark JSON definition file (e.g., `datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json`); the sidecar records the SHA-256 of that file plus a UTC registration timestamp. There is no separate `data/benchmark-registry/` directory.

**Registration semantics:**

- Idempotent: re-registering an unchanged file returns the existing reference.
- Content-hash collision with a different version of the same `benchmark_id` raises `BenchmarkContentCollisionError` (no silent overwrite).
- Missing file raises `BenchmarkFileNotFoundError`.
- Registration happens before any `SingleEvaluationRequest` is constructed; failure prevents partial-batch execution.

**Mechanism class:** [runtime gate]. Closed by L1-Pkt-A.

**Caller commitment:** Callers using benchmark adapters need not interact with `register_benchmark` directly; the adapter handles it. Callers constructing per-case requests programmatically without benchmarks leave `benchmark_reference` unset.

---



---

## 6 · Integrity guarantees

### 6.1 Integrity trail contents

Per gap analysis B1.5 CLOSED, C2.2 CLOSED, C2.3 CLOSED, the platform produces an integrity trail on every evaluation, recorded in the envelope's `integrity` field. The trail captures:

- **Per-capability records** — for each capability that ran (or was skipped), a record with:
  - capability ID
  - outcome (succeeded / failed / skipped)
  - duration_ms
  - error type and sanitized message (truncated to 500 chars if failed)
- **Capability chain** — the sequence of capabilities that ran, in order. From the caller's perspective: which capabilities ran and in what order.
- **HMAC signature** over the envelope's canonical serialization (per §6.2).
- **Append-only structure** — no in-place mutation; transitions produce new envelopes (per gap analysis B1.5 CLOSED). Mechanism class: [structural] (Pydantic frozen + factory-only transition methods).
- **Schema versioning** — `schema_version=3` on envelopes today (per §3.2).
- **Field ownership enforcement** (per L1-Pkt-3, CP-F3 closure) — `envelope.stamped(*, capability, **fields)` validates that field keys belong to the capability's allowlist (defined in `field_ownership.py`); raises `FieldOwnershipViolationError` on violation. Mechanism class: [runtime gate]. See §6.5.
- **Benchmark provenance** (when `SingleEvaluationRequest.benchmark_reference` is populated, per L1-Pkt-A) — envelope records `benchmark_id`, `benchmark_version`, `benchmark_content_hash`, `benchmark_registration_timestamp` under CAP-1's allowlist.

### 6.2 HMAC integrity (post-L1-Pkt-1 closure)

Per L1-Pkt-1 (CP-F2 + CP-F11 closure): HMAC integrity is mode-aware. Two modes via `LLM_JUDGE_MODE` env var:

**Production mode (`LLM_JUDGE_MODE=production`):**
- `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` env var is REQUIRED.
- `validate_configuration()` runs at `PlatformRunner.__init__`; raises `ConfigurationError` if env var unset or empty.
- Production deployments fail closed without the key. Forging envelopes via the default development key is structurally impossible.

**Development mode (`LLM_JUDGE_MODE=development`, the default):**
- `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` is optional.
- If unset, falls back to `_DEFAULT_DEV_KEY` with a one-shot startup warning (per L1-Pkt-1, the warning moved from per-envelope-construction to startup-time emission).
- Development workflows continue without env-var configuration.

- **Mechanism class:** [runtime gate] in production mode (env var enforcement); [convention] in development mode (warning is signaling, not enforcement).
- **Closure vehicle:** CP-F2 (closed by L1-Pkt-1).

The contract today: HMAC signatures verify the integrity trail against tampering when production mode is set with a configured key. Development mode preserves backward compat for dev workflows; the warning provides visibility into the unconfigured state.

**Caller commitment:** Production deployments must set `LLM_JUDGE_MODE=production` and `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY` to a non-default value before invoking `PlatformRunner`. Failure to do so raises `ConfigurationError` at startup, not silently in flight.

### 6.3 Field ownership (post-L1-Pkt-3 closure)

Per L1-Pkt-3 (CP-F3 closure): envelope field ownership is enforced at runtime gate. `envelope.stamped(*, capability, **fields)` validates that every field key passed belongs to `FIELD_OWNERSHIP[capability]` (defined in `src/llm_judge/control_plane/field_ownership.py`); raises `FieldOwnershipViolationError` (subclass of `ValueError`) on violation.

External callers do not construct envelopes directly (see §6.4) and cannot observe field-ownership enforcement from outside Layer 1 — the enforcement protects internal capability boundaries, not external surface. The per-capability allowlist is the field ownership table.

- **Mechanism class:** [runtime gate]. Closed by L1-Pkt-3.
- **Caller commitment:** Callers consuming envelopes do not need to know which capability stamped which field; envelope structure is opaque from the caller's perspective. Internal capability authors stamp only fields in their allowlist.

The detailed per-capability field ownership table is documented in the forthcoming **Layer 1 ↔ Layer 2-6 internal contract**. From the External Contract's perspective: enforcement exists, mechanism class is runtime gate, exception class is `FieldOwnershipViolationError`.

### 6.4 Authorized envelope construction

Per gap analysis B1.4 PARTIAL:
- `new_envelope` (per `envelope.py:207-229`) is the canonical factory.
- `ProvenanceEnvelope` is publicly importable; direct construction with hand-set fields is currently possible.
- The `_backfill_legacy_schema` validator does not validate signatures on input — a hand-constructed envelope with a fabricated signature string will be accepted by the model; downstream `verify_signature` would catch the issue but only if it is called.

- **Mechanism class:** [convention] today → [structural] target via private `__init__` plus factory-only construction, OR [runtime gate] target via mandatory signature verification at every read site.
- **Closure vehicle:** B1.4 — separate packet, **not Foundation Packet 1**.
- **Caller commitment:** Callers must use `new_envelope` (or methods that produce new envelopes from existing ones) for envelope construction. Direct construction is not guaranteed to produce verifying envelopes and is not part of the contract.
- **Verification responsibility:** Callers consuming envelopes from external sources must call `verify_signature` at every read site. The platform itself does not yet do this universally.

---

## 7 · Lifecycle and stability

### 7.1 Versioning model

The Layer 1 external contract is versioned. Each version captures **current state at time of drafting**, not target state.

- **v1 (2026-04-28, Proposed)** — Initial draft. Received CONCERNS verdict from Layer 1 chat.
- **v2 (2026-04-28, Proposed)** — Addressed v1 verdict findings; recon-anchored to `e739612`. Pass 4 recon by Layer 1 chat revealed §4 and §5.3 fundamentally wrong (not patchable).
- **v3 (2026-04-28, superseded by v3.1)** — Recon-anchored to `8e574304` on master after Phase 1 v2 landed. Incorporated Layer 1 chat's §4 and §5.3 replacement patches verbatim. Received PASS verdict from Layer 1 chat with three small items.
- **v3.1 (2026-04-28, superseded by v3.2)** — Applied Layer 1 chat's three small items: §9.2 dedup, §8.4 expansion, §10 ownership clarification. Layer 1 chat second-pass review caught three v3 → v3.1 inconsistencies in §10 #4, #5, #6.
- **v3.2 (2026-04-28, Active)** — Applied the three §10 inconsistency fixes from Layer 1 chat's second-pass review. Pre-Active Closures 1 and 2 satisfied; v3.2 went Active.
- **v3.3 (2026-04-29, Proposed)** — This version. Batches contract impacts from L1-Pkt-1 (CP-F2 closure: HMAC mode-aware in §6.2), L1-Pkt-2 (CP-F14 closure: `RubricNotInRegistryError` in §4.3(b), §4.6, §5.3(a); CP-F15 documentation in §4.6), L1-Pkt-3 (CP-F3 closure: field ownership enforcement in §6.3), L1-Pkt-A (CP-F1 closure: bypass eliminated in §1.3 #3, §1.3.1; new `benchmark_reference` field in §2.1; benchmark provenance in §6.1; CAP-1 expansion contract surface in §5.5). Per L1-Pkt-A architectural Decision 6: contract update ships with packet rather than queueing for v4.
- **v4 (forthcoming)** — Post-Internal-Contract revision. Captures the Internal Contract (Layer 1 ↔ Layer 2-6) as the source of truth for fields that v3.3 documents at the External surface. May relocate §6.3 detail to Internal Contract reference; consolidate cross-references.
- **Subsequent versions** — Each major packet that changes the external surface produces a new contract version.

**Version-number rules:**
- **Major version bumps** — breaking changes (removing required fields, removing entry points, changing field types, changing error class names).
- **Minor version bumps** — non-breaking additions (new optional fields, new entry points, new error categories).
- **Patch version bumps** — documentation-only changes (clarifications, examples, mechanism-class annotation refinements).

### 7.2 Deprecation policy

When a contract feature is deprecated:
- Deprecation announcement appears in the contract document's audit trail (§12) with a target removal version.
- **Grace period:** at least one full minor-version cycle, with a minimum wall-clock duration of **90 days** from the announcement date. Whichever is longer applies.
- Use of deprecated features may emit deprecation warnings (structured, not message-text-parseable).
- Removal occurs in the announced version.

### 7.3 Forward-compat expectations on callers

Callers should:
- Tolerate unknown response fields (forward-compat by ignoring extras).
- Use error class types, not message text.
- Pin `rubric_version` explicitly when reproducibility matters; do not rely on `"latest"` (per §2.4).
- Verify envelope signatures at every read site that consumes envelopes from external sources (per §6.4).
- **Check `integrity.complete` to detect degraded results** (per §4.4).
- **Catch `MissingProvenanceError`, `BatchInputSchemaError`, plain `ValueError` (CP-F14), `OSError` family, and `pydantic.ValidationError` at the appropriate boundary** (per §4 categories).

### 7.4 Backward-compat expectations from platform

The platform will:
- Accept requests in older schema shapes (via backward-compat validators) for at least one minor version cycle after schema evolution.
- Surface backward-compat handling explicitly in the integrity trail (e.g., "request normalized from schema_version=2 to schema_version=3").

---

## 8 · Out-of-scope (caller-side commitments about what the contract does NOT cover)

This section is framed as caller-side commitments rather than platform-side disclosures. The contract claim is "callers must not assume X"; the underlying state (we don't have X) follows.

### 8.1 Operational guardrails (Category 5 — deferred)

Per the locked decision (2026-04-28): operational guardrails defer until production deployment trigger fires (first external caller, first multi-tenant deployment, or first guardrails-related incident, whichever comes first).

**Caller commitments about operational guardrails:**

- **Callers must not assume the platform enforces per-caller rate limits.** Submitting at high rate may not be rejected but may produce degraded behavior under load. Per gap analysis D1.1 MISSING (CP-F8).
- **Callers must not assume per-request timeouts are enforced.** Slow capabilities may block indefinitely. Per gap analysis D1.3 MISSING (CP-F12).
- **Callers must not assume circuit breakers protect against capability failure storms.** Per gap analysis D1.2 MISSING (CP-F8).
- **Callers must not assume a kill switch or degraded-mode operation is available.** Per gap analysis D1.4 MISSING (CP-F8).
- **Callers must not assume resource accounting beyond `duration_ms` per capability.** Per gap analysis D1.5 PARTIAL.
- **Callers must not assume `BatchRunner` resumability or checkpoint.** Mid-batch interrupt loses not-yet-processed cases. Per gap analysis CP-F19.

### 8.2 Internal contracts

This contract covers external-facing surface only. Callers must not rely on:
- The Layer 1 ↔ Layer 2-6 internal contract (capability wrapper interface, envelope field ownership rules, event vocabulary, integrity record schema) — this is a forthcoming separate contract.
- The Layer 1 ↔ Layer 7-8 contract (governance preflight signal interface, lifecycle event publication for top-level surfaces) — this is a forthcoming separate contract.

### 8.3 Capability-level guarantees

This contract specifies what Layer 1 (Control) guarantees. Callers must not assume from this contract:
- What CAP-7 guarantees about evaluation accuracy
- What CAP-1 guarantees about dataset governance
- What CAP-5 guarantees about manifest schema
- These belong to capability-level contracts (within their respective layers).

### 8.4 Admission-time governance preflight

Callers must not assume admission-time governance preflight occurs. Per §5.3, today's reality:

- **Rubric existence is verified lazily** (inside CAP-2 wrapper, after CAP-1 has run); failure produces a degraded result (CAP-2 fails, integrity records it) or — for `rubric_version="latest"` with unknown `rubric_id` — a plain `ValueError` propagated via CAP-5.
- **Rubric governance status** (validated/production) is **NOT** checked.
- **Rubric lifecycle fields** (`last_reviewed`, `dimensions`, `metrics_schema`) are **NOT** validated.
- **Rubric prompt artifact (F11)** is **NOT** checked.

Callers requiring governance assurance must validate rubrics themselves before invocation, or wait for CP-F18 closure (which adds admission-time preflight) before relying on the platform.

Closure vehicle: CP-F18.

---

## 9 · Open questions and known gaps

### 9.1 Open contract questions

- **§2.4 (rubric_version "latest" default):** Will the eventual closure shape be option (a) require explicit pin, (b) record resolved version, or (c) admission-time resolution into self-describing request? Decision pending the closure packet for A1.2.
- **§3.3 (schema-evolution rules):** Specific rules for breaking vs non-breaking schema changes are not yet documented (B1.6 PARTIAL). The contract should not go Active until at least the [convention]-level closure of B1.6 lands (§10 acceptance criteria).
- **§4.7 (error sanitization):** Sanitization is partial (A1.5 PARTIAL). Closure in a separate packet, **not Foundation Packet 1**.
- **§5.3 (governance preflight):** When admission-time preflight is added (CP-F18), will it be implemented as a Layer 1 internal call to `check_rubrics_governed`, or via the Layer 1 ↔ Layer 7-8 contract surface? Decision pending the relevant contract drafting.

### 9.2 Known gaps from Layer 1 chat findings

The contract references the following findings from `layer-1-control-state-and-gaps-v1.2.md` (pending publication; v1.2 adds CP-F14 through CP-F19):

| Finding | Status | Severity | Referenced in |
|---|---|---|---|
| CP-F1 | OPEN (Foundation Packet 1) | High | §1.3, §1.3.1 |
| CP-F2 | OPEN (Foundation Packet 1) | High | §6.2 |
| CP-F14 | New (future packet) | Low | §2.4, §4.3(b), §4.6, §5.3, §7.3 |
| CP-F15 | New (documentation) | Low | §4.6 (RubricSchemaError exists; subclass of ValueError) |
| CP-F16 | New (future packet) | Medium | §4.3(c)(d), §4.6 |
| CP-F17 | New (future packet) | Medium | §4.5 |
| CP-F18 | New (future packet) | Low | §5.3, §5.4, §8.4 |
| CP-F19 | New (Category 5 deferred) | Low | §4.5, §8.1 |
| F11 | OPEN (future packet) | Medium | §5.3, §5.4 |
| A1.2 | PARTIAL (future packet) | — | §2.4 |
| A1.4 | OPEN (future packet) | — | §2.1 |
| A1.5 | PARTIAL (future packet) | — | §4.7 |
| A2.2 | OPEN (future packet) | — | §1.3 |
| B1.4 | PARTIAL (future packet) | — | §1.3, §6.4 |
| B1.6 | PARTIAL (future packet, gating §10 acceptance) | — | §3.3 |

**Findings outside this contract** (referenced for completeness):
- CP-F3 (field ownership) → Layer 1 ↔ Layer 2-6 Internal Contract
- CP-F8, CP-F12 (operational guardrails) → §8.1 caller commitments only; full coverage in Category 5 work post-trigger

---

## 10 · Acceptance criteria

This contract is **Active (v3.3.1)** at L1-Pkt-A merge. Acceptance is structurally tied to the packet that introduces the contract changes:

1. v3.3 draft complete (architect chat, 2026-04-29).
2. Layer 1 chat reviewed v3.3 against L1-Pkt-A's architectural decisions and current code at the L1-Pkt-A pre-flight master HEAD (`994eaba`).
3. Architect chat applied surgical revisions to produce v3.3.1: §5.5 BenchmarkReference shape and storage convention aligned with as-shipped code per Pre-flight 4 + Pre-flight 6 recon.
4. L1-Pkt-A PR opens with v3.3.1 included as a docs-only commit alongside the code changes.
5. Layer 1 chat verification of L1-Pkt-A confirms code matches v3.3.1's claims.
6. L1-Pkt-A merges; v3.3.1 becomes Active at the merge moment.

This pattern (contract ships with the packet that changes it) supersedes the queue-for-batch-revision pattern used for L1-Pkt-1 and L1-Pkt-2 contract impacts. Per L1-Pkt-A architectural Decision 6: changes substantial enough to materially affect the contract surface ship in lock-step with the packet that introduces them.

After v3.3 Active:

7. Subsequent packets that don't materially change the contract surface continue to use the queue pattern (batched into next version).
8. v4 drafts after Internal Contract drafts. v4 captures the Internal Contract as source-of-truth for fields that v3.3 documents at the External surface; consolidates cross-references.

### 10.5 Contract evolution discipline

When the contract and code disagree:

- **Layer 1 chat surfaces the drift** during gap analysis updates or contract reviews.
- **Architect chat decides** whether the contract is wrong (refine document — produce a v.x.1 patch or v.x+1 minor version) or the code is wrong (close the gap via packet — refer to the relevant CP-F or property finding).
- **The decision is recorded in the audit trail** (§12) with: date, drift description, resolution direction (document or code), packet reference (if code).

This discipline prevents drift from accumulating silently. Every time it activates, it becomes evidence that the verification chat structure is doing its job.

---

## 11 · Document references

- `docs/adr/adr-platform-8-layer-architecture.md` — 8-layer architecture ADR
- `docs/adr/architecture-map-8-layer-specification.md` — full layer specification
- `layer-1-control-state-and-gaps-v1.4.md` — Layer 1 chat's gap analysis post-L1-Pkt-2 (current as of v3.3 drafting)
- `layer-1-control-state-and-gaps-v1.2.md` — gap analysis at Pass 4 recon
- `layer-1-control-state-and-gaps-v1.1.md` — prior gap analysis at Pass 2 recon
- `layer-1-external-contract-v3.2.md` — superseded predecessor (Active before v3.3)
- `cap1-expansion-spec.md` — CAP-1 benchmark dataset governance specification (referenced by §5.5; consumed by L1-Pkt-A)
- `f19-disposition.md` — multi-commit packet workflow disposition (referenced by Brief Template v1.2+)
- `brief-template-v1.3.md` — current packet brief template
- `three-lens-coordination-model-v1.md` — coordination model for layer/architect/execution chats
- `schema-evolution-policy-v1.2.md` — Active schema evolution policy
- Forthcoming: Layer 1 ↔ Layer 2-6 Internal Contract (drafts after L1-Pkt-3 + L1-Pkt-A merge)
- Forthcoming: Layer 1 ↔ Layer 7-8 Contract

**Recon baseline commit at v3.3 drafting:** to be set at L1-Pkt-A pre-flight master HEAD. v3.3 ships as docs-only commit alongside L1-Pkt-A code commits.

---

## 12 · Audit trail

- **2026-04-28 v1 Proposed** — Drafted by previous architect chat from Layer 1 chat's gap analysis. Drafted partly from memory; received CONCERNS verdict.
- **2026-04-28 v2 Proposed** — Drafted by architect chat. Recon-anchored against `e739612`. Addressed v1 verdict findings. Layer 1 chat Pass 4 recon revealed §4 and §5.3 fundamentally wrong; v2 superseded.
- **2026-04-28 v3 Proposed** — Drafted by architect chat after Layer 1 chat Pass 4 recon at `8e574304`. Incorporates §4 and §5.3 replacement patches verbatim. Six new findings (CP-F14 through CP-F19) referenced.
- **2026-04-28 v3.1 Proposed** — Layer 1 chat returned PASS verdict on v3 with three small items. Architect chat applied: §9.2 dedup (CP-F19 removed from "outside this contract" footnote), §8.4 expansion (four-bullet caller checklist), §10 ownership clarification (Pre-Active Closure 1 = architect chat; Pre-Active Closure 2 = Layer 1 chat). Layer 1 chat verdict on v3.1: PASS with one §10 typo ("Proposed (v3)" → "Proposed (v3.1)") — applied within v3.1.
- **2026-04-28 v3.2 Proposed** — Layer 1 chat second-pass review caught three v3 → v3.1 inconsistencies in §10 #4, #5, #6 (acceptance criteria sentences referenced "v3" rather than "v3.1"). v3.2 applies those fixes. All §10 acceptance criteria now consistently reference v3.2.
- **2026-04-28 v3.2 Active** — Pre-Active Closures 1 and 2 satisfied (Schema Evolution Policy v1.2 published; Gap Analysis v1.2 published). v3.2 went Active and remained Active through L1-Pkt-1 and L1-Pkt-2 ship cycles. Contract impacts from those packets (CP-F2, CP-F11, CP-F4-F6, CP-F14-F15) queued for batch revision rather than per-packet contract bumps.
- **2026-04-29 v3.3 Proposed** — Drafted by architect chat following L1-Pkt-A Decision 6 (contract update ships with packet rather than queueing). Batches four packets' impacts: L1-Pkt-1 (§6.2 HMAC mode-aware), L1-Pkt-2 (§4.3(b), §4.6, §5.3(a) RubricNotInRegistryError), L1-Pkt-3 (§6.3 field ownership runtime gate), L1-Pkt-A (§1.3 #3, §1.3.1 bypass closure; §2.1 benchmark_reference field; §5.5 benchmark dataset registration; §6.1 benchmark provenance recording; §4.6 expanded exception class taxonomy). Acceptance pattern shifts from Pre-Active Closures to "ships with the packet that introduces the changes." Held for L1-Pkt-A as-shipped revision; superseded by v3.3.1.
- **2026-04-30 v3.3.1 Active** — Ships with L1-Pkt-A PR. Surgical revision over v3.3: §5.5 BenchmarkReference shape aligned with as-shipped code per Pre-flight 6 recon (`version` → `benchmark_version`, `content_hash` → `benchmark_content_hash`, `registration_timestamp` → `benchmark_registration_timestamp`, `file_path` dropped); §5.5 sidecar storage convention added per Pre-flight 4 recon; §5.5 API reference updated to `llm_judge.datasets.benchmark_registry.register_benchmark` (as-shipped path, not `cap1.register_benchmark`). L1-Pkt-3 absorbed into L1-Pkt-A — the CP-F3 + CP-F1 closure ships in a single merge so the §6.3 drift window does not materialize. Status promoted to Active at L1-Pkt-A merge.
- **[Future] v4** — Post-Internal-Contract revision. Drafts after L1-Pkt-A ships and Internal Contract drafts. Captures Internal Contract as source of truth for fields v3.3.1 documents at External surface.

---

## 13 · Changes from v3.2 (for Layer 1 chat re-review)

This section maps each contract change in v3.3 to the packet that introduced it. Per L1-Pkt-A architectural Decision 6 (contract update ships with the packet that materially changes the surface), four packets' impacts are batched into v3.3.

### L1-Pkt-1 impacts (CP-F2 + CP-F11 closures)

| Section | v3.2 state | v3.3 state |
|---|---|---|
| §6.2 HMAC integrity | OPEN gap; default-key fallback at convention class; closure deferred to Foundation Packet 1 | Closed at runtime gate. Mode-aware enforcement: production mode requires `LLM_JUDGE_CONTROL_PLANE_HMAC_KEY`, development mode preserves default with one-shot startup warning. `validate_configuration()` runs at `PlatformRunner.__init__`. |
| §4.6 Error class stability | (no `ConfigurationError` listed) | Added `ConfigurationError` to taxonomy table. |

### L1-Pkt-2 impacts (CP-F14 + CP-F15 closures, plus other cleanup)

| Section | v3.2 state | v3.3 state |
|---|---|---|
| §4.3(b) | Plain `ValueError` from `rubric_store._resolve_version`; closure deferred to CP-F14 future packet | `RubricNotInRegistryError` (subclass of `ValueError`) at runtime gate. Backward compat preserved. |
| §5.3(a) | Plain `ValueError` for unknown rubric_id | Updated to `RubricNotInRegistryError` reference per §4.3(b). |
| §4.6 Error class stability | (RubricSchemaError undocumented; CP-F15) | Added `RubricSchemaError` to taxonomy table per L1-Pkt-2 documentation hygiene. |

### L1-Pkt-3 impacts (CP-F3 closure)

| Section | v3.2 state | v3.3 state |
|---|---|---|
| §6.3 Field ownership | "Moved to Internal Contract"; CP-F3 closure deferred to Foundation Packet 1 | Closed at runtime gate. `envelope.stamped()` validates fields against `FIELD_OWNERSHIP[capability]` per `field_ownership.py`; raises `FieldOwnershipViolationError` (subclass of `ValueError`). Detailed per-capability table remains in forthcoming Internal Contract. |
| §6.1 Integrity trail contents | (no field ownership enforcement listed) | Added field ownership enforcement note. |
| §4.6 Error class stability | (no `FieldOwnershipViolationError` listed) | Added `FieldOwnershipViolationError` to taxonomy table. |

### L1-Pkt-A impacts (CP-F1 closure + CAP-1 expansion)

| Section | v3.2 state | v3.3 state |
|---|---|---|
| §1.3 #3 | Bypass at convention; benchmark adapter listed as known violation; closure deferred to Foundation Packet 1 | Closed at runtime gate. Benchmark adapter refactored to load files and route through `BatchRunner`. Static repository assertion confirms no parallel-entry function. |
| §1.3.1 | All §1.3 clauses noted as "aspirational, not enforced" | Per-clause enforcement state distinguished. §1.3 #3 enforced post-L1-Pkt-A. §1.3 #1, #2 remain aspirational. |
| §2.1 SingleEvaluationRequest schema | (no `benchmark_reference` field) | Added optional `benchmark_reference: BenchmarkReference \| None` field. Backward compat preserved via `None` default. |
| §5.5 Benchmark dataset registration | (section did not exist) | New section. Documents `cap1.register_benchmark()` API surface, `BenchmarkReference` shape, idempotent registration, fail-closed semantics. References `cap1-expansion-spec.md`. |
| §6.1 Integrity trail contents | (no benchmark provenance recording) | Added benchmark provenance recording when `benchmark_reference` populated. |
| §4.6 Error class stability | (no `BenchmarkFileNotFoundError`, `BenchmarkContentCollisionError` listed) | Added both to taxonomy table per CAP-1 expansion. |

### Acceptance pattern shift

v3.2 used Pre-Active Closures (architect-chat- and Layer-1-chat-owned gates that had to land before contract went Active). v3.3 ships with the packet (L1-Pkt-A) that introduces the bulk of the changes. This pattern (per L1-Pkt-A Decision 6) supersedes the queue-for-batch pattern used through L1-Pkt-2.

Going forward:
- Substantial contract impact → ship contract revision with the packet (v3.3 pattern).
- Minor contract impact → queue for next batch revision (v3.2-and-earlier pattern).

### What v3.3 does NOT change

- Sections 1.0, 1.1, 1.1a, 1.2, 1.4, 2.2-2.5, 3.x, 4.1, 4.2, 4.4, 4.5, 4.7, 5.1, 5.2, 5.3 (b)(c), 5.4, 6.4, 7.x, 8.x, 9.x, 10.5, 12 (other than new entries) — unchanged from v3.2.
- The Foundation Packet 1 framing in v3.2's §13 is superseded; v3.3 reflects the actual sequence (L1-Pkt-1, L1-Pkt-2, L1-Pkt-3, L1-Pkt-A) that replaced the original Foundation Packet 1 scope-collapse.

### What v3.3 defers to v4

- Internal Contract drafts after L1-Pkt-3 + L1-Pkt-A both merge. v4 captures the Internal Contract as source of truth for fields v3.3 documents at External surface.
- §6.3 detail (per-capability field ownership table) may relocate to Internal Contract reference.
- Cross-references between External and Internal Contract may consolidate.

---

*Layer 1 External Contract v3.3.1 · prepared by architect chat · 2026-04-29 (drafted) / 2026-04-30 (revised to as-shipped) · Active at L1-Pkt-A merge · batches contract impacts from L1-Pkt-1, L1-Pkt-2, L1-Pkt-3 (absorbed), L1-Pkt-A per Decision 6*
