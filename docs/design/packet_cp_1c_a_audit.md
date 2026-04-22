# Packet CP-1c-a — Audit Turn

Date: 2026-04-22
Branch: feat/packet-cp-1c-a-rubric-governance
Commit: 5d42d4f7646d6ba4182c2898e002d1a89007add4 (stacked on CP-1 +
CP-1b; 7 commits ahead of master)

Read-only audit. No file renamed, deleted, created, or edited
except this report.

## 1. Rename blast-radius audit

### Raw grep output

**GREP 1 — `configs/rules/` in src/tests/docs/.github/Makefile** (6 matches)

```
src/llm_judge/scorer.py:406:      1. Config-driven plan: configs/rules/{rubric_id}_{version}.yaml
src/llm_judge/scorer.py:457:            "hint": "Create configs/rules/{rubric_id}_{version}.yaml",
src/llm_judge/rules/engine.py:210:    Load plan YAML from: configs/rules/{rubric_id}_{version}.yaml
docs/DEV_GUIDE.md:120:4. Add to rule plan config: `configs/rules/{rubric_id}_{version}.yaml`
docs/DEV_GUIDE.md:134:- **Rule Plans:** `configs/rules/` — per-rubric rule plan configs
docs/design/packet_cp_1_recon.md:125:  `configs/rules/<rubric_id>_<version>.yaml` (engine.py:215) and
```

**GREP 2 — `configs.rules` dotted-path in src/tests/** (0 matches)

**GREP 3 — `rules_plan|rule_plan` near yaml/path/load** (0 matches)

**GREP 4 — loader functions (`load_plan|load_rules|load_rule_plan`)** (29 matches)

Informative but not all are rename-blast-radius — the function name
`load_plan_for_rubric` is unchanged by the rename. Only fixture
writes and the loader body care about file path:

```
src/llm_judge/rules/engine.py:208:def load_plan_for_rubric(rubric_id: str, version: str) -> RulePlan:
src/llm_judge/rules/engine.py:215:    path = config_root() / "rules" / f"{rubric_id}_{version}.yaml"
src/llm_judge/scorer.py:9:   from llm_judge.rules.engine import RuleEngine, load_plan_for_rubric
src/llm_judge/scorer.py:419: plan = load_plan_for_rubric(rubric_id, rubric_version)
src/llm_judge/eval/run.py:200: from llm_judge.rules.engine import load_plan_for_rubric
src/llm_judge/eval/run.py:202: plan = load_plan_for_rubric(rubric.rubric_id, rubric.version)
src/llm_judge/control_plane/wrappers.py:32: from ... import load_plan_for_rubric, run_rules
src/llm_judge/control_plane/wrappers.py:122: plan = load_plan_for_rubric(...)
... plus 21 test-side call sites that use the function API only
(no path literal).
```

**GREP 5 — Markdown doc references** (2 files)

```
./docs/DEV_GUIDE.md
./docs/design/packet_cp_1_recon.md
```

**GREP 6 — non-configs-rules YAMLs referencing the path** (1 file)

```
./old/epics_v2.yaml:310: "Given a RunSpec, when scorer loads rules,
  then it uses configs/rules/{rubric}_{version}.yaml — not hardcoded
  defaults"
```

(Historical artifact under `old/`; not executed.)

**GREP 7 — Makefile/shell** (0 matches)

**GREP 8 — CI config (.github/)** (0 matches)

**GREP 9 — pyproject/setup** (0 matches)

**GREP 10 — path-joined constructions in tests** (supplementary, surfaced
call sites Grep 1 missed because they use `Path` concatenation rather
than the literal string `"configs/rules/"`):

```
tests/unit/test_rules_engine.py:15:
    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
tests/unit/test_rules_engine_coverage.py:101:
    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
tests/unit/test_rules_engine_coverage.py:131:
    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
tests/unit/test_rules_engine_coverage.py:162:
    (tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(
tests/unit/test_epic_d1_paths.py:218:
    (cfg / "rules" / "test_v1.yaml").write_text(
```

### Classified matches

| # | file:line | bucket | quoted line | requires update? |
|---|-----------|--------|-------------|------------------|
| 1 | src/llm_judge/rules/engine.py:215 | LOADER | `path = config_root() / "rules" / f"{rubric_id}_{version}.yaml"` | YES |
| 2 | src/llm_judge/rules/engine.py:210 | DOC (in-code docstring) | `Load plan YAML from: configs/rules/{rubric_id}_{version}.yaml` | YES (same change as row 1) |
| 3 | src/llm_judge/scorer.py:406 | DOC (in-code comment) | `# 1. Config-driven plan: configs/rules/{rubric_id}_{version}.yaml` | YES (cosmetic; guides future readers) |
| 4 | src/llm_judge/scorer.py:457 | STRING (user-facing log hint) | `"hint": "Create configs/rules/{rubric_id}_{version}.yaml"` | YES (user-visible error message points to wrong path if not updated) |
| 5 | docs/DEV_GUIDE.md:120 | DOC | ``4. Add to rule plan config: `configs/rules/{rubric_id}_{version}.yaml` `` | YES |
| 6 | docs/DEV_GUIDE.md:134 | DOC | ``- **Rule Plans:** `configs/rules/` — per-rubric rule plan configs`` | NO (refers to the directory prefix, which is unchanged) |
| 7 | docs/design/packet_cp_1_recon.md:125 | DOC (historical audit) | `configs/rules/<rubric_id>_<version>.yaml (engine.py:215)` | NO (audits are historical artifacts; don't retroactively edit) |
| 8 | tests/unit/test_rules_engine.py:15 | TEST (fixture write) | `(tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(` | YES |
| 9 | tests/unit/test_rules_engine_coverage.py:101 | TEST (fixture write) | `(tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(` | YES |
| 10 | tests/unit/test_rules_engine_coverage.py:131 | TEST (fixture write) | `(tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(` | YES |
| 11 | tests/unit/test_rules_engine_coverage.py:162 | TEST (fixture write) | `(tmp_path / "configs" / "rules" / "chat_quality_v1.yaml").write_text(` | YES |
| 12 | tests/unit/test_epic_d1_paths.py:218 | TEST (fixture write) | `(cfg / "rules" / "test_v1.yaml").write_text(` | YES |
| 13 | tests/unit/test_rules_engine_coverage.py:97 | TEST (comment) | `# Use cwd override because load_plan_for_rubric uses Path("configs")/...` | NO (generic, doesn't name flat filename) |
| 14 | tests/unit/test_wave1_integration.py:202 | TEST (docstring) | `"""Given math_basic_v1.yaml exists, load_plan_for_rubric returns it."""` | NO (cosmetic; test exercises function API) |
| 15 | tests/unit/test_wave1_integration.py:213 | TEST (docstring) | `"""Given chat_quality_v1.yaml exists, load_plan_for_rubric returns it."""` | NO (cosmetic) |
| 16 | tests/unit/test_wave1_integration.py:233 | TEST (comment) | `# All rules in chat_quality_v1.yaml have enabled: true` | NO (cosmetic) |
| 17 | old/epics_v2.yaml:310 | CONFIG (historical) | `Given a RunSpec, when scorer loads rules, then it uses configs/rules/{rubric}_{version}.yaml` | NO (under `old/`, not executed) |

### Final blast-radius count

Total unique call sites (deduplicated): **17**
Of which REQUIRES UPDATE: **10**

Breakdown of the 10 that must change:
- 1 LOADER (engine.py:215 — the real path construction)
- 1 STRING (scorer.py:457 — user-visible error hint)
- 2 DOC comments in code (engine.py:210 docstring; scorer.py:406 comment)
- 1 DOC external (DEV_GUIDE.md:120)
- 5 TEST fixtures (test_rules_engine.py:15; test_rules_engine_coverage.py:101/131/162; test_epic_d1_paths.py:218)

Threshold: **>5 triggers stop-and-report.**
Status: **OVER THRESHOLD (10 > 5).**

The rename is still mechanically tractable — one loader change,
one user-facing log message, two in-code comments, one doc page,
and five test fixtures. But the count is double what the packet
anticipated, and the architect may want to reconsider whether to
bundle the rename with CP-1c-a or defer it to a separate
housekeeping packet so the rubric governance changes land
cleanly first.

## 2. Rubric count audit

### Files found

| # | path | status? | owner? | created_at? | last_reviewed? |
|---|------|---------|--------|-------------|----------------|
| 1 | rubrics/chat_quality/v1.yaml | N | Y | Y | N |
| 2 | rubrics/math_basic/v1.yaml | N | Y | Y | N |
| 3 | rubrics/math_basic/math_basic_v1.yaml | N | Y | Y | N |
| 4 | rubrics/math_basic/v1_old.yaml | N | N | N | N |

### Final count

Total rubric files: **4**
Files needing full migration (missing ≥2 required fields): **1**
  (v1_old.yaml — zero governance fields; but it's a graveyard file
  named v1_old, likely safe to delete rather than migrate)
Files needing partial migration (missing 1 field): **0**
Files needing partial migration (missing 2 fields — status + last_reviewed): **3**
  (chat_quality/v1.yaml, math_basic/v1.yaml, math_basic/math_basic_v1.yaml)
Files already compliant (all 4 fields present, need audit_log only): **0**

Threshold: **>15 total triggers stop-and-report.**
Status: **UNDER THRESHOLD (4 ≤ 15).**

Caveats for the architect:
- `rubrics/math_basic/math_basic_v1.yaml` and `rubrics/math_basic/v1.yaml`
  appear to be duplicates of the same version — the CP-1c-a packet
  will need to decide which is canonical and what to do with the
  other.
- `rubrics/math_basic/v1_old.yaml` is an explicit graveyard name;
  cleanest outcome is deletion rather than migration.
- So "files actually needing migration" is effectively **2**
  (chat_quality/v1.yaml + math_basic/v1.yaml) once duplicates /
  graveyard are resolved.

## 3. Auxiliary diagnostics

- **`rubrics.py` importers found: 1** — *unexpected; the recon
  note said 0.*
  - `tests/unit/test_api_contracts.py:59` — `from llm_judge.rubrics import RUBRICS`
    inside `test_chat_quality_rubric_exists()`. This test asserts
    the legacy module's RUBRICS dict contains `chat_quality` with
    `version == "v1"` and `"clarity"` among dimensions. It is an
    API-contract smoke test. Any rubric governance design that
    plans to delete `rubrics.py` must either update or delete this
    test (the test locks the legacy surface in place).
- **`rubrics/latest.yaml` readers found: 0** (expected). The file
  exists but nothing reads it — drift risk flagged in the earlier
  rubrics recon stands.
- **`v1_old / math_basic_v1.yaml` references found: 2**
  - `old/epics_v2.yaml:158` — historical note. `old/` is
    non-executing.
  - `tests/unit/test_wave1_integration.py:202` — docstring.
  Neither forces a code path today; safe to ignore during
  migration.
- **`rules/lifecycle.py` line count: 554.** The reference module
  exists and is substantial (manifest load, status enum, aging,
  audit-log, deprecation-enforcement, CLI). A parallel
  `rubrics/lifecycle.py` has a concrete pattern to mirror.

Unexpected findings:
- `tests/unit/test_api_contracts.py:59` importing `llm_judge.rubrics`
  directly. This contradicts the earlier rubric recon's "no
  importers" claim. The architect should know that the legacy
  `rubrics.py` surface is not safely removable without first
  handling this test.

## 4. Headline recommendation

**GO-WITH-ADJUSTMENTS.**

One threshold cleanly passes (4 rubric files ≤ 15), the other is
breached (10 rename call sites > 5). The overage is mechanical
rather than architectural — five test fixtures, one log-message
string, two in-code comments, one doc page, one loader line — so
the rename is still tractable. Adjustments the architect should
weigh before Commit 2 lands:

1. **Consider splitting the rename into its own packet** (or its
   own commit pair inside CP-1c-a) — loader + fixtures in one
   commit, docs + user-facing strings in another — so rubric-
   governance commits are not entangled with the rename's blast
   radius if one half needs to be reverted.
2. **Settle duplicates in `rubrics/math_basic/`** before the
   migration commit: `math_basic_v1.yaml` and `v1.yaml` appear to
   be the same version under two names; `v1_old.yaml` is a
   graveyard. Decide canonical vs. delete.
3. **Account for `test_api_contracts.py:59`** — legacy
   `rubrics.py` is not import-free and cannot be removed without
   handling that test.
