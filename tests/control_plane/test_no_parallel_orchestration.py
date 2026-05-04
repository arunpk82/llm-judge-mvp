"""CP-F1 gap-absence test (AST-walking).

Per L1-Pkt-A v2.2 brief Section 3 Scenario 7 + Decision 13: no
production code function in ``src/llm_judge/`` outside
``control_plane/`` may produce evaluation outputs without
transitively calling ``run_single_evaluation`` (PlatformRunner) or
``run_batch`` (BatchRunner).

Heuristic per Pre-flight 7 refinement + Option (c) intra-module
exclusion:

  Detection target types
    ``SingleEvaluationResult``, ``BatchResult``,
    ``BenchmarkRunResult``, ``list[SingleEvaluationResult]``.
    Envelope types are excluded — they are too tied to control-
    plane plumbing to be a useful parallel-entry signal.

  Scope
    ``src/llm_judge/``.

  Structural exclusions
    * Functions defined in ``src/llm_judge/control_plane/`` —
      orchestration substrate, allowed to return these types.
    * Functions called only intra-module (no callers outside
      their own module) — private library helpers, not parallel
      entry points. ``_evaluate_case_all_properties`` survives
      Option (c) here: post-Commit 6 all its callers live in
      ``benchmarks/runner.py``.

  Assertion
    Any function whose return-type annotation is in the target set
    AND that has any caller outside its own module must transitively
    reach ``PlatformRunner.run_single_evaluation`` or
    ``BatchRunner.run_batch``.

If Commit 6's removal of ``run_benchmark`` were reverted, this test
fails: ``run_benchmark`` returns ``BenchmarkRunResult``, lives
outside ``control_plane/``, has cross-module callers (``run_all``,
``tools/run_ragtruth50.py``), and does NOT transitively reach the
sanctioned entries — so the assertion fires loudly.
"""

from __future__ import annotations

import ast
from pathlib import Path

# --------------------------------------------------------------------
# Heuristic configuration
# --------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src" / "llm_judge"
EXCLUDED_SUBTREE = SRC_ROOT / "control_plane"

TARGET_RETURN_TYPES: frozenset[str] = frozenset(
    {
        "SingleEvaluationResult",
        "BatchResult",
        "BenchmarkRunResult",
        # ``list[SingleEvaluationResult]`` — annotation rendering
        # varies; the AST-side detector below also accepts subscripted
        # forms whose value is "list" and whose slice is one of the
        # target names.
    }
)

# These are the sanctioned entry points. A parallel-entry candidate
# must transitively reach one of them via its own module (call graph
# analysis goes one hop into intra-module callees by name).
SANCTIONED_ENTRY_NAMES: frozenset[str] = frozenset(
    {"run_single_evaluation", "run_batch"}
)


# --------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------


def _iter_python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if p.is_file())


def _annotation_returns_target(node: ast.AST | None) -> bool:
    """Accept ``Name(target)``, ``Attribute(...).target``, or
    subscripted forms like ``list[target]`` / ``Optional[target]``."""
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in TARGET_RETURN_TYPES
    if isinstance(node, ast.Attribute):
        return node.attr in TARGET_RETURN_TYPES
    if isinstance(node, ast.Subscript):
        # list[SingleEvaluationResult], Optional[BatchResult], etc.
        return _annotation_returns_target(node.slice)
    if isinstance(node, ast.Tuple):
        return any(_annotation_returns_target(e) for e in node.elts)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        # String annotations: ``"BenchmarkRunResult"``.
        return node.value.strip("\"'") in TARGET_RETURN_TYPES
    return False


def _function_callees(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names called inside a function body. Catches direct ``foo(...)``
    and ``obj.foo(...)`` forms — enough to walk one hop."""
    names: set[str] = set()
    for child in ast.walk(func):
        if isinstance(child, ast.Call):
            f = child.func
            if isinstance(f, ast.Name):
                names.add(f.id)
            elif isinstance(f, ast.Attribute):
                names.add(f.attr)
    return names


# --------------------------------------------------------------------
# Build per-file index of candidate functions and the cross-module
# caller index in a single pass.
# --------------------------------------------------------------------


def _build_indexes() -> tuple[
    dict[Path, dict[str, set[str]]], dict[str, set[Path]]
]:
    """Return (per_file_function_callees, cross_module_caller_index).

    per_file_function_callees: file_path -> {func_name: {callee_names}}
    cross_module_caller_index: callee_name -> {set of files that mention
                               it AS A CALL or AS AN IMPORT}
    """
    per_file_funcs: dict[Path, dict[str, set[str]]] = {}
    callsite_index: dict[str, set[Path]] = {}

    for path in _iter_python_files(SRC_ROOT):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        per_file_funcs[path] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                per_file_funcs[path][node.name] = _function_callees(node)
            if isinstance(node, ast.Call):
                f = node.func
                name: str | None = None
                if isinstance(f, ast.Name):
                    name = f.id
                elif isinstance(f, ast.Attribute):
                    name = f.attr
                if name:
                    callsite_index.setdefault(name, set()).add(path)
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    callsite_index.setdefault(alias.name, set()).add(path)

    return per_file_funcs, callsite_index


# --------------------------------------------------------------------
# The gap-absence assertion
# --------------------------------------------------------------------


def _has_cross_module_callers(
    func_name: str,
    defining_file: Path,
    caller_index: dict[str, set[Path]],
) -> bool:
    callers = caller_index.get(func_name, set())
    return any(c != defining_file for c in callers)


def _transitively_reaches_sanctioned_entry(
    func_name: str,
    defining_file: Path,
    per_file_funcs: dict[Path, dict[str, set[str]]],
    max_depth: int = 6,
) -> bool:
    """Walk callees within the defining file, then one hop into any
    callee that's a sanctioned name. Conservative — does not chase
    cross-file call edges. The platform-level convention is that
    parallel-entry candidates either call the sanctioned name
    directly or via a thin wrapper inside the same module."""
    file_funcs = per_file_funcs.get(defining_file, {})
    seen: set[str] = set()
    frontier: list[tuple[str, int]] = [(func_name, 0)]
    while frontier:
        name, depth = frontier.pop()
        if name in seen or depth > max_depth:
            continue
        seen.add(name)
        callees = file_funcs.get(name, set())
        if callees & SANCTIONED_ENTRY_NAMES:
            return True
        for c in callees:
            if c in file_funcs:
                frontier.append((c, depth + 1))
    return False


def _collect_parallel_entry_candidates(
    per_file_funcs: dict[Path, dict[str, set[str]]],
) -> list[tuple[Path, str]]:
    """Return (file_path, func_name) for every function in scope
    whose return-type annotation matches the target set, before
    structural exclusions are applied."""
    candidates: list[tuple[Path, str]] = []
    for path in _iter_python_files(SRC_ROOT):
        if EXCLUDED_SUBTREE in path.parents or path == EXCLUDED_SUBTREE:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if _annotation_returns_target(node.returns):
                    candidates.append((path, node.name))
    return candidates


# --------------------------------------------------------------------
# The actual test
# --------------------------------------------------------------------


def test_no_parallel_orchestration_entry_points() -> None:
    """CP-F1 gap-absence: any function in src/llm_judge/ outside
    control_plane/ that returns one of the target evaluation result
    types AND has cross-module callers must transitively call a
    sanctioned entry. Library helpers (intra-module callees only)
    are exempt — Option (c)."""
    per_file_funcs, caller_index = _build_indexes()
    candidates = _collect_parallel_entry_candidates(per_file_funcs)

    violations: list[str] = []
    for defining_file, func_name in candidates:
        if not _has_cross_module_callers(
            func_name, defining_file, caller_index
        ):
            # Intra-module library helper — exempt by Option (c).
            continue
        if _transitively_reaches_sanctioned_entry(
            func_name, defining_file, per_file_funcs
        ):
            continue
        violations.append(
            f"{defining_file.relative_to(REPO_ROOT)}::{func_name} "
            f"returns a target evaluation type, has cross-module "
            f"callers, and does not transitively reach "
            f"{sorted(SANCTIONED_ENTRY_NAMES)} — likely parallel "
            f"orchestration entry."
        )

    assert not violations, (
        "CP-F1 gap-absence violated — parallel orchestration entry "
        "point(s) found:\n  " + "\n  ".join(violations)
    )


def test_evaluate_case_all_properties_preserved_as_intra_module_helper() -> None:
    """Option (c) explicit guard: ``_evaluate_case_all_properties``
    must survive at ``benchmarks/runner.py`` AND have no callers
    outside its own module. If a future refactor moves it or surfaces
    a cross-module caller, the brief Decision 13 / Pre-flight 3
    invariant breaks and this guard surfaces the change."""
    runner_file = SRC_ROOT / "benchmarks" / "runner.py"
    assert runner_file.is_file(), "benchmarks/runner.py missing"
    tree = ast.parse(runner_file.read_text(encoding="utf-8"))
    defined_here = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_evaluate_case_all_properties" in defined_here, (
        "_evaluate_case_all_properties missing from benchmarks/runner.py "
        "— Option (c) library-helper invariant broken."
    )

    _, caller_index = _build_indexes()
    callers = caller_index.get("_evaluate_case_all_properties", set())
    cross_module = sorted(
        str(c.relative_to(REPO_ROOT)) for c in callers if c != runner_file
    )
    assert not cross_module, (
        "_evaluate_case_all_properties acquired cross-module callers: "
        f"{cross_module}. Option (c)'s intra-module-helper invariant "
        "is violated; rethink the architectural home before adding "
        "external callers."
    )


def test_run_benchmark_does_not_exist_in_runner() -> None:
    """Direct guard against the specific re-introduction risk: the
    name ``run_benchmark`` must not be defined in
    ``benchmarks/runner.py`` post-Commit 6."""
    runner_file = SRC_ROOT / "benchmarks" / "runner.py"
    tree = ast.parse(runner_file.read_text(encoding="utf-8"))
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "run_benchmark" not in names, (
        "run_benchmark was reintroduced into benchmarks/runner.py — "
        "this is the parallel-orchestration entry point CP-F1 closed."
    )
