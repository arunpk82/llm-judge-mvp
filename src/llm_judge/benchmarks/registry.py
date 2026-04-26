"""Adapter registry: name → BenchmarkAdapter class.

Provides a lightweight name-based lookup for the seven canonical
benchmark connectors so the batch driver can dispatch on a string
(e.g. ``--benchmark ragtruth_50``) without each call site importing
every adapter module.

The registry is additive: existing direct-instantiation call sites
(``RAGTruthAdapter()`` and friends in ``benchmarks/*_science_gate.py``,
``benchmarks/run_all.py``, ``tools/run_ragtruth50.py``) keep working
unchanged. The registry only adds the lookup function — it does not
replace direct construction.

Canonical names registered at module import:

  ragtruth_50  → RAGTruthAdapter
  halueval     → HaluEvalAdapter
  fever        → FEVERAdapter
  ifeval       → IFEvalAdapter
  jigsaw       → JigsawAdapter
  toxigen      → ToxiGenAdapter
  faithdial    → FaithDialAdapter
"""

from __future__ import annotations

from llm_judge.benchmarks import BenchmarkAdapter


class BenchmarkNotFoundError(Exception):
    """Raised by :func:`get` when a benchmark name is not registered."""


_REGISTRY: dict[str, type[BenchmarkAdapter]] = {}


def register(name: str, adapter_class: type[BenchmarkAdapter]) -> None:
    """Register an adapter class under ``name``. Last write wins."""
    _REGISTRY[name] = adapter_class


def get(name: str) -> type[BenchmarkAdapter]:
    """Return the adapter class registered under ``name``.

    Raises :class:`BenchmarkNotFoundError` listing the available
    names when ``name`` is not present — callers commonly surface
    this directly to the user.
    """
    if name not in _REGISTRY:
        raise BenchmarkNotFoundError(
            f"benchmark {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def list_benchmarks() -> list[str]:
    """Return the registered benchmark names, sorted alphabetically."""
    return sorted(_REGISTRY.keys())


# --- Canonical-name registrations ---------------------------------------
# Imports kept at module-bottom so consumers who only need register/get/
# list_benchmarks for a custom registration don't pay for loading every
# adapter module up front.

from llm_judge.benchmarks.faithdial import FaithDialAdapter  # noqa: E402
from llm_judge.benchmarks.fever import FEVERAdapter  # noqa: E402
from llm_judge.benchmarks.halueval import HaluEvalAdapter  # noqa: E402
from llm_judge.benchmarks.ifeval import IFEvalAdapter  # noqa: E402
from llm_judge.benchmarks.jigsaw import JigsawAdapter  # noqa: E402
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter  # noqa: E402
from llm_judge.benchmarks.toxigen import ToxiGenAdapter  # noqa: E402

register("ragtruth_50", RAGTruthAdapter)
register("halueval", HaluEvalAdapter)
register("fever", FEVERAdapter)
register("ifeval", IFEvalAdapter)
register("jigsaw", JigsawAdapter)
register("toxigen", ToxiGenAdapter)
register("faithdial", FaithDialAdapter)
