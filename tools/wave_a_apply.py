#!/usr/bin/env python3
"""
Wave A applier — implements EPIC D1.2 (spaCy splitter) and ADR-0029 (rename).

This script is idempotent: it can be run multiple times. It will refuse to apply
a patch if the source state does not match what's expected (so a partial apply
will not silently double-edit).

Run from repo root:
    python3 scripts/wave_a_apply.py

What it does:
1. Patches src/llm_judge/calibration/hallucination.py:
   - Replaces the regex sentence splitter with a spaCy-primary, regex-fallback
     splitter (ADR-0014, EPIC D1.2).
   - Removes the dead _l3_graphrag_check function (ADR-0029).
   - Removes the section comment for that function.
2. Patches src/llm_judge/calibration/hallucination_graphs.py:
   - Updates two doc references to _l3_graphrag_check.
3. Renames code symbols across src/, tests/, experiments/ (ADR-0029):
     _l0_deterministic_match  -> _l1_substring_match
     _l1_gate1_check          -> _l3_minilm_gate_check
     _l2a_minicheck_score     -> _l3_minicheck_score
     _l2a_minicheck           -> _l3_minicheck
     _l2_nli_check            -> _l3_deberta_nli
   And layer_stats key renames:
     "L3a_minicheck"          -> "L3_minicheck"
     "L3b_nli"                -> "L3_deberta"
4. Removes tests of the deleted _l3_graphrag_check.

After the script runs, follow with:
    pytest tests/unit/test_split_sentences.py -v
    pytest tests/unit/ -x
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()


# ---- Helpers ----

def fail(msg: str) -> None:
    print(f"\n[FAIL] {msg}", file=sys.stderr)
    sys.exit(1)


def info(msg: str) -> None:
    print(f"[ok]   {msg}")


def warn(msg: str) -> None:
    print(f"[warn] {msg}")


def read(path: Path) -> str:
    if not path.exists():
        fail(f"Expected file not found: {path}")
    return path.read_text(encoding="utf-8")


def write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def assert_repo_root() -> None:
    if not (REPO_ROOT / "src" / "llm_judge").is_dir():
        fail(
            "This does not look like the llm-judge-mvp repo root. "
            "Run from the repo root (cd ~/work/llm-judge-mvp first)."
        )


# ---- Patch 1: spaCy splitter in hallucination.py ----

OLD_SPLITTER_BLOCK = """# Sentence splitter
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\\s+")


def _split_sentences(text: str) -> list[str]:
    \"\"\"Split text into sentences, filtering short fragments.\"\"\"
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]
"""

NEW_SPLITTER_BLOCK = """# Sentence splitter — spaCy primary, regex fallback (loud).
# spaCy doc.sents handles abbreviations ("J. Paul Getty"), titles
# ("Dr. Smith"), and initials ("D. C.") that the regex breaks on.
# See ADR-0014 (spaCy sentence splitting) and EPIC D1.2.
_SENTENCE_SPLIT_REGEX_FALLBACK = re.compile(r"(?<=[.!?])\\s+")
_SPACY_FAIL_LOGGED: bool = False


def _split_sentences(text: str) -> list[str]:
    \"\"\"Split text into sentences using spaCy; fall back to regex on failure.

    spaCy ``doc.sents`` correctly handles abbreviations ("J. Paul Getty"),
    titles ("Dr. Smith"), and initials ("D. C."). The legacy regex
    splitter breaks on these and produces false positive hallucination
    flags downstream (e.g., RAGTruth case 6).

    On spaCy load failure, falls back to regex with a one-time WARNING
    log so the degraded mode is visible in CI and production logs.
    Filters fragments shorter than 11 characters (legacy behavior).
    \"\"\"
    global _SPACY_FAIL_LOGGED

    text = text.strip()
    if not text:
        return []

    nlp = None
    try:
        _load_spacy()
        nlp = _spacy_nlp
    except Exception as e:
        if not _SPACY_FAIL_LOGGED:
            logger.warning(
                "spaCy unavailable for sentence splitting (%s: %s). "
                "Falling back to regex splitter; abbreviations like "
                "'J. Paul Getty' will be mis-split, producing false "
                "positives. Install: python -m spacy download en_core_web_sm",
                type(e).__name__,
                str(e)[:120],
            )
            _SPACY_FAIL_LOGGED = True

    if nlp is not None:
        doc = nlp(text)
        return [s.text.strip() for s in doc.sents if len(s.text.strip()) > 10]

    # Fallback path — abbreviation false positives expected.
    sentences = _SENTENCE_SPLIT_REGEX_FALLBACK.split(text)
    return [s.strip() for s in sentences if len(s.strip()) > 10]
"""


def patch_splitter(text: str) -> str:
    if "_SENTENCE_SPLIT_REGEX_FALLBACK" in text:
        warn("Splitter patch already applied; skipping.")
        return text
    if OLD_SPLITTER_BLOCK not in text:
        fail(
            "Could not find the original splitter block in hallucination.py. "
            "The file may have been edited since this script was written. "
            "Hand-apply the splitter change instead."
        )
    info("Patching sentence splitter (regex -> spaCy with regex fallback)")
    return text.replace(OLD_SPLITTER_BLOCK, NEW_SPLITTER_BLOCK)


# ---- Patch 2: Remove _l3_graphrag_check from hallucination.py ----

DEAD_FUNCTION_BLOCK = '''# --- L3: GraphRAG spaCy SVO exact match ---

_spacy_nlp: Any = None


def _load_spacy():
    """Lazy-load spaCy model (~50MB)."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return

    import spacy

    _spacy_nlp = spacy.load("en_core_web_sm")


def _l3_graphrag_check(sentence: str, source_doc: str) -> bool:
    """
    L3: spaCy SVO triplet exact match.
    Confirmed: handles 17% of sentences (Experiment 12).
    Returns True if sentence is confirmed grounded via exact match.
    """
    _load_spacy()

    try:
        from llm_judge.benchmarks.graphrag_science_gate import (
            _entity_overlap,
            extract_svo_triplets,
        )

        resp_triplets = extract_svo_triplets(sentence, _spacy_nlp)
        src_triplets = extract_svo_triplets(source_doc, _spacy_nlp)

        # Filter intransitive triplets
        resp_triplets = [t for t in resp_triplets if t.obj != "(intransitive)"]
        src_triplets = [t for t in src_triplets if t.obj != "(intransitive)"]

        if not resp_triplets:
            return False

        all_matched = True
        has_any = False

        for rt in resp_triplets:
            found = False
            for st in src_triplets:
                subj_sim = _entity_overlap(rt.subject, st.subject)
                obj_sim = _entity_overlap(rt.obj, st.obj)
                pred_sim = _entity_overlap(rt.predicate, st.predicate)
                score = (subj_sim + obj_sim) / 2
                if score >= 0.5 and pred_sim >= 0.5:
                    found = True
                    break
            if found:
                has_any = True
            else:
                all_matched = False

        return has_any and all_matched

    except (ImportError, Exception) as e:
        logger.debug(f"l3_graphrag.skip: {str(e)[:60]}")
        return False
'''

REPLACEMENT_BLOCK = '''# --- spaCy lazy-loader (used by L2 ensemble and the sentence splitter) ---

_spacy_nlp: Any = None


def _load_spacy():
    """Lazy-load spaCy model (~50MB).

    Raises on failure so callers can decide their own fallback policy.
    The sentence splitter catches and falls back to regex; L2 ensemble
    catches and skips the L2 layer.
    """
    global _spacy_nlp
    if _spacy_nlp is not None:
        return

    import spacy

    _spacy_nlp = spacy.load("en_core_web_sm")
'''


def patch_remove_dead_function(text: str) -> str:
    if "_l3_graphrag_check" not in text:
        warn("Dead function _l3_graphrag_check already removed; skipping.")
        return text
    if DEAD_FUNCTION_BLOCK not in text:
        fail(
            "Could not find _l3_graphrag_check block verbatim. "
            "The file may have drifted; hand-apply the removal instead."
        )
    info("Removing dead _l3_graphrag_check function (ADR-0029)")
    return text.replace(DEAD_FUNCTION_BLOCK, REPLACEMENT_BLOCK)


# ---- Patch 3: Update doc references in hallucination_graphs.py ----

GRAPHS_DOC_OLD_1 = "Replaces old L3 GraphRAG (_l3_graphrag_check) with:"
GRAPHS_DOC_NEW_1 = (
    "Supersedes the legacy _l3_graphrag_check (removed in Wave A per ADR-0029) with:"
)

GRAPHS_DOC_OLD_2 = "    This replaces the old _l3_graphrag_check() in hallucination.py."
GRAPHS_DOC_NEW_2 = (
    "    This supersedes the legacy _l3_graphrag_check() (removed per ADR-0029)."
)


def patch_graphs_docs(text: str) -> str:
    changed = False
    if GRAPHS_DOC_OLD_1 in text:
        text = text.replace(GRAPHS_DOC_OLD_1, GRAPHS_DOC_NEW_1)
        changed = True
    if GRAPHS_DOC_OLD_2 in text:
        text = text.replace(GRAPHS_DOC_OLD_2, GRAPHS_DOC_NEW_2)
        changed = True
    if changed:
        info("Updated doc references in hallucination_graphs.py")
    else:
        warn("hallucination_graphs.py doc references already updated; skipping.")
    return text


# ---- Patch 4: Symbol renames across src/, tests/, experiments/ ----

RENAME_TARGETS = [
    "src/llm_judge/calibration/hallucination.py",
    "src/llm_judge/calibration/hallucination_graphs.py",
    "tests/unit/test_5layer_pipeline.py",
    "tests/unit/test_funnel_report.py",
    "experiments/l2_independent_benchmark.py",
    "experiments/ragtruth_benchmark.py",
    "experiments/test_production_pipeline.py",
]

# Order matters: longer names first to avoid prefix collisions.
RENAMES = [
    # Function symbols
    ("_l0_deterministic_match", "_l1_substring_match"),
    ("_l1_gate1_check", "_l3_minilm_gate_check"),
    ("_l2a_minicheck_score", "_l3_minicheck_score"),
    ("_l2a_minicheck", "_l3_minicheck"),
    ("_l2_nli_check", "_l3_deberta_nli"),
    # layer_stats keys
    ('"L3a_minicheck"', '"L3_minicheck"'),
    ('"L3b_nli"', '"L3_deberta"'),
]


def apply_renames(path: Path) -> bool:
    text = read(path)
    original = text
    for old, new in RENAMES:
        text = text.replace(old, new)
    if text != original:
        write(path, text)
        return True
    return False


# ---- Patch 5: Remove tests of the deleted _l3_graphrag_check ----

# We surgically remove four test methods that directly test the dead function,
# and three patch.object lines that patch a function that no longer exists.

GRAPHRAG_TEST_PATCHES: list[str] = [
    # The four test methods that import and call _l3_graphrag_check directly.
    # These tests assert behavior of a removed function and must be deleted.
    # We match them by their unique signature (the import + call pattern).
    # Each pattern matches one test method including its decorators and body
    # up to (but not including) the next `def test_` at the same indent or
    # the class boundary.
    # Strategy: remove the entire test class containing _l3_graphrag_check tests.
]


def patch_remove_graphrag_tests(text: str) -> str:
    """Remove the test class that tests the dead _l3_graphrag_check function.

    The class is identified by being the only class that imports and calls
    _l3_graphrag_check across multiple test methods.
    """
    # Find a class that contains _l3_graphrag_check
    if "_l3_graphrag_check" not in text:
        warn("No graphrag test references found; skipping test removal.")
        return text

    # Strategy: find each test method (def test_xxx) that mentions
    # _l3_graphrag_check and remove that single method, leaving sibling tests.
    # A method runs from "    def test_xxx" to the next "    def " at the same
    # indent or the next class definition.
    pattern = re.compile(
        r"^    def (test_\w+)\(self[^)]*\) -> None:.*?"
        r"(?=^    def |^class |\Z)",
        re.DOTALL | re.MULTILINE,
    )

    def keep_or_drop(m: re.Match) -> str:
        body = m.group(0)
        if "_l3_graphrag_check" in body:
            # Remove this method entirely.
            return ""
        return body

    new_text = pattern.sub(keep_or_drop, text)

    # Also remove any leftover patch.object(hal, "_l3_graphrag_check", ...) lines.
    # These appear inside other tests' patch context lists.
    new_text = re.sub(
        r"\s*patch\.object\(hal, \"_l3_graphrag_check\"[^)]*\),?\n",
        "\n",
        new_text,
    )
    new_text = re.sub(
        r"\s*\"llm_judge\.calibration\.hallucination\._l3_graphrag_check\"[^,]*,?\n",
        "\n",
        new_text,
    )

    # Remove the now-empty TestL3GraphRAGCheck class shell and its section
    # header. The class body was emptied by the per-method removal above;
    # what remains is `class TestL3GraphRAGCheck:\n    """..."""\n` plus a
    # ##### section header above it.
    new_text = re.sub(
        r"\n# =+\n# L3: GraphRAG spaCy exact match \(mocked\)\n# =+\n+"
        r"class TestL3GraphRAGCheck:\n"
        r"    \"\"\"[^\"]+\"\"\"\n+",
        "\n\n",
        new_text,
    )

    # Remove the stale top-level docstring line that lists L3 GraphRAG
    # as a layer being tested. The function it describes is now removed.
    new_text = re.sub(
        r"^  L3: GraphRAG spaCy exact match \(mocked\)\n",
        "",
        new_text,
        flags=re.MULTILINE,
    )
    # Likewise update the integration-flow note that listed L3 alongside L2a/L2b.
    new_text = new_text.replace(
        "Integration: Full layered flow with mocked L2a/L2b/L3/L4",
        "Integration: Full layered flow with mocked L3 classifiers and L4",
    )

    if new_text != text:
        info("Removed tests for dead _l3_graphrag_check function")
    return new_text


# ---- Main ----

def main() -> None:
    assert_repo_root()

    # 1 + 2: Patch hallucination.py (splitter + dead function)
    hp = REPO_ROOT / "src/llm_judge/calibration/hallucination.py"
    text = read(hp)
    text = patch_splitter(text)
    text = patch_remove_dead_function(text)
    write(hp, text)

    # 3: Patch hallucination_graphs.py docs
    hg = REPO_ROOT / "src/llm_judge/calibration/hallucination_graphs.py"
    text = read(hg)
    text = patch_graphs_docs(text)
    write(hg, text)

    # 4: Symbol renames across all targets
    print("\nApplying symbol renames (ADR-0029)...")
    for rel in RENAME_TARGETS:
        path = REPO_ROOT / rel
        if not path.exists():
            warn(f"Target file not present, skipping: {rel}")
            continue
        if apply_renames(path):
            info(f"Renamed symbols in {rel}")
        else:
            warn(f"No renames needed in {rel} (already done?)")

    # 5: Remove tests of the dead function
    print("\nRemoving tests for deleted _l3_graphrag_check...")
    test_5lp = REPO_ROOT / "tests/unit/test_5layer_pipeline.py"
    text = read(test_5lp)
    text = patch_remove_graphrag_tests(text)
    write(test_5lp, text)

    print("\n[done] Wave A applied. Verify with:")
    print("  git status")
    print("  git diff --stat")
    print("  pytest tests/unit/test_split_sentences.py -v")
    print("  pytest tests/unit/ -x")
    print("\nIf anything looks wrong, 'git checkout -- .' to revert.")


if __name__ == "__main__":
    main()
