# Session Continuity Brief — LLM-Judge MVP

**Session date:** 2026-04-21 (~24 hour session, continuing from 2026-04-19/20)
**Previous brief:** `docs/sessions/session_continuity_brief (4).md` (Apr 19–20, 2026) — closed pending the clean L1-only measurement read.
**Repo:** `github.com/arunpk82/llm-judge-mvp`
**Master HEAD at session close:** `9b81f6f` — `docs: ADR-0031 L2 graph cache miss contract (#175)`

---

## 1. What This Session Was Supposed to Do

Starting state: the 2026-04-20 session closed with the clean L1-only run completed but unread — its parity numbers were sitting in `results/ragtruth50_results.json` measured but never gated, and L2's cache-miss behavior was undocumented, unmonitored, and invisible to the operator. Ending state: L1 parity is now a `<1s` CI gate (`tests/smoke/test_l1_parity.py` in PR #170), the L2 cache-miss contract is codified as ADR-0031 (PR #175), and three issues (#171, #172, #173) are filed that fully specify the next session's implementation work. This was a *clean-execution* session applying L65–L67 from the previous brief — findings were captured, contracts were written, issues were filed. No new crisis was discovered; no 23-minute mystery consumed the night.

---

## 2. What Landed on Master Tonight

| # | PR   | SHA       | Issue | Description |
|---|------|-----------|-------|-------------|
| 1 | #168 | `783b13e` | #167  | `run_ragtruth50.py` serializer dropped `sentence_level_metrics` before writing results JSON — one-line fix + regression test in `tests/unit/test_benchmarks.py::TestRunRagtruth50Serializer`. |
| 2 | #170 | `24c6a95` | #169  | `tests/smoke/test_l1_parity.py` — six-case parametrized fixture locking L1's Exp 29B clearance parity. Bypasses spaCy cold-load by hand-building `SOURCE_SENTENCES`. All six cases green in 0.22s. |
| 3 | #175 | `9b81f6f` | #174  | `docs/adr/0031-l2-cache-miss-contract.md` — codifies the *miss* companion to ADR-0025's *hit* contract. MADR 4.0; status `Proposed`; docs-only. |

All three PRs cleared the seven-check governance gate on the first push. No close/rename/reopen cycles tonight.

---

## 3. What's Filed and Pending for Next Session

Three new issues filed, with a deliberate dependency chain:

| Issue | Title | Next-session role |
|-------|-------|-------------------|
| #171 | `obs: L2 cache miss is silent — upgrade to warning + surface metric` | **START HERE.** Implements clauses 1–3 of ADR-0031: WARNING log with source-hash prefix + reason (`not_exist`/`expired`/`corrupt`), `layer_stats["L2_cache_miss"]` with reason, run-summary aggregation. Widens `GraphCache.get()` signature. |
| #172 | `test: Path A — L2 ensemble flag-wins smoke (monkeypatched extraction)` | **Parallel with #171.** Unit-level smoke; monkeypatches extraction and asserts flag-wins aggregation (ADR-0020) at the ensemble level. Independent of the cache contract. |
| #173 | `test: Path B — L2 cache contract integration (cache-miss must emit detectable signal)` | **BLOCKED on #171.** Integration-level CI gate that asserts the exact signal #171 emits. Must fail against pre-#171 master and go green post-#171. |

Open issue count at session close: **68**.

---

## 4. The Deep Finding

Verbatim from `src/llm_judge/calibration/hallucination.py:674-702` (master HEAD `9b81f6f`):

```python
    l2_grounded = set()  # Only grounded (100% precision) stops here
    l2_flagged = set()  # Flagged cascades to L3 with evidence (cascade rule)
    l2_graphs = knowledge_graphs

    # Graph cache lookup (ADR-0025): if no fact_tables provided,
    # check the cache using source document hash. Cache hit = free L2.
    if l2_enabled and fact_tables is None and knowledge_graphs is None and source_doc:
        try:
            from llm_judge.calibration.graph_cache import get_graph_cache

            cache = get_graph_cache()
            cached = cache.get(source_doc)
            if cached is not None:
                fact_tables = cached
                layer_stats["L2_cache_hit"] = 1
                logger.debug("l2.cache_hit")
            else:
                layer_stats["L2_cache_miss"] = 1
                logger.debug("l2.cache_miss")
        except Exception as e:
            logger.debug(f"l2.cache_error: {str(e)[:60]}")

    if l2_enabled and (fact_tables or knowledge_graphs):
        try:
            from llm_judge.calibration.hallucination_graphs import (
                build_all_graphs,
                l2_ensemble_check,
            )
```

Verbatim from `src/llm_judge/calibration/graph_cache.py:87-136`:

```python
    def _entry_path(self, source_hash: str) -> Path:
        return self._cache_dir / f"{source_hash}.json"

    def _is_expired(self, path: Path) -> bool:
        if self._ttl_seconds <= 0:
            return False
        try:
            mtime = path.stat().st_mtime
            age = time.time() - mtime
            return age > self._ttl_seconds
        except OSError:
            return True

    def get(self, source_text: str) -> dict[str, Any] | None:
        """
        Look up cached fact tables for a source document.

        Args:
            source_text: The source document text (hashed for lookup).

        Returns:
            The fact tables dict (containing ``passes`` key) if cached
            and not expired, else None.
        """
        source_hash = compute_source_hash(source_text)
        path = self._entry_path(source_hash)

        if not path.exists():
            self._misses += 1
            return None

        if self._is_expired(path):
            self._misses += 1
            logger.debug(
                "graph_cache.expired",
                extra={"hash": source_hash[:16], "path": str(path)},
            )
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._hits += 1
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "graph_cache.read_error",
                extra={"hash": source_hash[:16], "error": str(e)[:80]},
            )
            self._misses += 1
            return None
```

Grep result confirming no live-extraction fallback site in the runtime cascade (re-run at session close against `9b81f6f`):

```
$ grep -nE "extract_fact_tables|kg_extraction" src/llm_judge/calibration/hallucination.py
(no matches)
```

Three scenarios follow from these two code blocks:

1. **Hash match, entry valid** — `path.exists()` is True, `_is_expired` is False, `json.loads` succeeds → `_hits += 1`, returns data dict → caller sets `L2_cache_hit`, L2 runs. Correct.
2. **Hash changed (or never registered)** — `path.exists()` is False → `_misses += 1`, returns `None` silently (no log at all) → caller sets `L2_cache_miss` and `logger.debug("l2.cache_miss")`. L2 silently abstains. **Invisible to operator.**
3. **No file, or expired, or corrupt** — all three collapse to the same `None` return. The caller cannot tell them apart. Expired logs at DEBUG inside `graph_cache`; corrupt logs at WARNING inside `graph_cache` but the caller discards the distinction. **Operator cannot diagnose the miss reason without re-running under DEBUG.**

The grep confirms no runtime path calls Gemini extraction on miss — the decision space is bounded to *observable abstention* vs *silent abstention*, not *cache* vs *fallback*. ADR-0031 picks observable abstention.

**Next session must not re-discover this.**

---

## 5. L68 Candidate for Playbook

Drop-in paragraph for the next version bump of the AI Architect's Playbook:

> **L68 — Silent skip is the worst failure mode.** A loud failure is handled; an abstain is logged; a silent skip degrades the system invisibly. Before shipping any layered pipeline component that can be bypassed — by cache miss, feature flag, timeout, or fallback — specify what the bypass signal is and verify it reaches the observability surface. "It didn't run" must be as visible as "it ran and failed." (Surfaced 2026-04-21 from the L2 cache-miss investigation; codified in ADR-0031; enforced by issues #171 and #173.)

---

## 6. Remaining Tech Debt

Carried forward from the 2026-04-20 brief (Section 5) — *none* of these landed tonight:

1. **Delete three stale pipeline yamls** — `hallucination.yaml`, `hallucination_l2_only.yaml`, `hallucination_legacy.yaml` use a dead schema; silently load as all-defaults. Separate cleanup PR. (Verified still unfiled: listed in `docs/adr/README.md` "Known tech debt".)
2. **Add `results/` to `.gitignore` + `git rm --cached results/benchmark_checkpoint.json`.** Verified unaddressed: `grep "^results" .gitignore` is empty; `git ls-files results/` still tracks `benchmark_checkpoint.json`.
3. **Fix `tools/run_ragtruth50.py:71` preseed path** — still passes raw `item.get("source_info", "")` (may be dict) straight to `compute_source_hash`. Crashes on QA/Data2txt task types. Verified line 71 unchanged on `9b81f6f`.
4. **`make calibrate-l1` should skip preseed** — with L2 disabled, preseed provisions a cache that is never consulted.
5. **`make git-start` should validate branch name against governance regex** — shift-left on `feat/` vs `feature/` mistakes.
6. **`make git-ship` should refuse commits without `Closes #<N>`** — same shift-left principle.

Surfaced tonight:

7. **`GraphCache.get()` signature widening required by ADR-0031.** Today returns `dict | None`; must return the miss reason on miss so the caller can emit the WARNING in clause 1. One caller today (`hallucination.py:674-702`), so the blast radius is small. Implementation work tracked under #171.

---

## 7. Environment / Tools

- **Repo root:** `~/work/llm-judge-mvp` (WSL, native filesystem)
- **Python env:** Poetry
- **Run environment:** WSL, `GEMINI_API_KEY` is set
- **Claude Code:** version **2.1.116** (up from 2.1.114 in the prior brief — minor bump, no behavior change observed)
- **Model used this session:** Opus 4.7 (the planned switch from the prior brief)
- **Master HEAD at session close:** `9b81f6f`
- **Working tree at close:** clean apart from the untracked `results/ragtruth50_results.json` and `.claude/` (both are covered by tech debt item #2; ignorable).

### Key files for next session to reference

- `src/llm_judge/calibration/hallucination.py:674-702` — L2 cache-consult site. Primary target of #171.
- `src/llm_judge/calibration/graph_cache.py:87-136` — `_is_expired` + `get`. Signature widens here.
- `docs/adr/0031-l2-cache-miss-contract.md` — the contract #171 and #173 enforce.
- `docs/adr/0025-fact-table-extraction-as-registration-step.md` — the companion *hit* contract.
- `tests/smoke/test_l1_parity.py` — template for the L2 smoke in #172.

---

## 8. Immediate Next Actions (In Order)

1. **Implement #171 (observability upgrade).** Must land first — #173 asserts the contract that #171 creates. Widen `GraphCache.get()` to return a miss-reason, update `hallucination.py:674-702` to emit WARNING with reason + source-hash prefix, bump `layer_stats["L2_cache_miss"]` with reason, aggregate into run summary.
2. **Write #172 (Path A smoke).** Can be done in parallel with #171 or immediately after — it does not depend on the cache contract. Monkeypatches extraction and asserts flag-wins aggregation at the ensemble level.
3. **Write #173 (Path B integration).** Last. The test should fail against pre-#171 master (proving the silent-skip problem is real) and go green post-#171 (proving the fix is load-bearing).

---

## 9. What Arun Said at the End

> "Let's do what a world-class AI company would do."

And we did. Stopped at hour 24+ with findings captured, issues filed, contract codified, and handoff written. The tests themselves wait for a fresh session.

---

**End of brief.** Open this file at the start of the next session and begin with Section 8, item 1.
