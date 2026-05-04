"""CP-F9 gap-absence test (AST-walking).

Per Brief Template v1.3: a gap-absence test exercises the closure
surface and would fail if the runtime gate introduced by L1-Pkt-B
Commit 2 were reverted.

End-State property D2.1 + D2.2 transition the capability registry
from convention to runtime gate. This test asserts:

  * No production module under ``src/llm_judge/control_plane/``
    (other than ``capability_registry.py`` itself) contains a
    list/tuple literal with all four capability ids — that is, a
    hard-coded sequence that bypasses the registry.

  * Any function that calls three or more of the four
    ``invoke_capN`` wrappers must iterate :data:`CAPABILITY_REGISTRY`
    inside its body. Calling the wrappers in sequence without the
    registry-driven loop is the exact pre-L1-Pkt-B shape; this
    catches a regression there.

If L1-Pkt-B Commit 2 were reverted (orchestrator goes back to the
hard-coded CAP-1 → CAP-2 → CAP-7 → CAP-5 inline sequence), the
``run_single_evaluation`` body would call all four ``invoke_capN``
without a ``for ... in CAPABILITY_REGISTRY`` loop and the second
assertion below would fire.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTROL_PLANE_ROOT = REPO_ROOT / "src" / "llm_judge" / "control_plane"

# Modules excluded from the literal-sequence check because they are
# the registry definitions themselves. ``capability_registry.py``
# defines CAPABILITY_REGISTRY; tests for that file live elsewhere.
LITERAL_CHECK_EXCLUDED_FILES: frozenset[str] = frozenset(
    {"capability_registry.py"}
)

CAP_IDS: frozenset[str] = frozenset({"CAP-1", "CAP-2", "CAP-7", "CAP-5"})

INVOKE_NAMES: frozenset[str] = frozenset(
    {"invoke_cap1", "invoke_cap2", "invoke_cap5", "invoke_cap7"}
)

REGISTRY_NAME = "CAPABILITY_REGISTRY"


def _iter_module_files() -> list[Path]:
    return sorted(p for p in CONTROL_PLANE_ROOT.rglob("*.py") if p.is_file())


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _list_or_tuple_string_values(node: ast.AST) -> set[str] | None:
    """Return the string-literal members of ``node`` if it is a list
    or tuple literal whose elements are all string constants. Else
    ``None``."""
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    members: set[str] = set()
    for elt in node.elts:
        s = _string_literal(elt)
        if s is None:
            return None
        members.add(s)
    return members


def _function_invoke_call_names(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    """Names of ``invoke_capN`` wrappers called inside ``func``."""
    called: set[str] = set()
    for child in ast.walk(func):
        if isinstance(child, ast.Call):
            f = child.func
            name: str | None = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name is not None and name in INVOKE_NAMES:
                called.add(name)
    return called


def _function_iterates_registry(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
) -> bool:
    """True iff the function body contains ``for ... in CAPABILITY_REGISTRY``
    (the registry-driven iteration the runner uses post-Commit 2)."""
    for child in ast.walk(func):
        if isinstance(child, ast.For) and isinstance(child.iter, ast.Name):
            if child.iter.id == REGISTRY_NAME:
                return True
    return False


# ---------------------------------------------------------------------
# Heuristic 1 — no list/tuple literal with all four capability ids
# outside the registry definition file.
# ---------------------------------------------------------------------


def test_no_hard_coded_capability_sequence_literal() -> None:
    """A list/tuple containing all four capability id strings is a
    hard-coded orchestration sequence that bypasses the registry.
    Dict keys are not considered a sequence (registries are dicts/
    tuples-of-models)."""
    offenders: list[tuple[Path, int]] = []
    for module_path in _iter_module_files():
        if module_path.name in LITERAL_CHECK_EXCLUDED_FILES:
            continue
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            members = _list_or_tuple_string_values(node)
            if members is None:
                continue
            if CAP_IDS.issubset(members):
                lineno = getattr(node, "lineno", -1)
                offenders.append((module_path, lineno))
    assert not offenders, (
        "Found list/tuple literals containing all four capability ids "
        "outside capability_registry.py — that is a hard-coded "
        "capability sequence that bypasses CAPABILITY_REGISTRY. "
        f"Offenders: {offenders}"
    )


# ---------------------------------------------------------------------
# Heuristic 2 — any function calling 3+ invoke_capN wrappers must
# iterate CAPABILITY_REGISTRY in its body.
# ---------------------------------------------------------------------


def test_multi_capability_callers_iterate_registry() -> None:
    """Calling three or more ``invoke_capN`` wrappers without a
    ``for ... in CAPABILITY_REGISTRY`` loop is exactly the shape of
    the pre-L1-Pkt-B hard-coded orchestrator. Any reversion of
    Commit 2 makes ``run_single_evaluation`` fail this assertion."""
    offenders: list[tuple[Path, str, int, set[str]]] = []
    for module_path in _iter_module_files():
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            called = _function_invoke_call_names(node)
            if len(called) < 3:
                continue
            if _function_iterates_registry(node):
                continue
            offenders.append((module_path, node.name, node.lineno, called))
    assert not offenders, (
        "Functions calling 3+ invoke_capN wrappers must iterate "
        "CAPABILITY_REGISTRY (the registry-driven orchestrator shape). "
        f"Offenders: {offenders}"
    )
