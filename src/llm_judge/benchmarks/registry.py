"""Adapter registry: name → zero-arg factory producing a configured BenchmarkAdapter.

Provides a lightweight name-based lookup for the seven canonical
benchmark connectors so the batch driver can dispatch on a string
(e.g. ``--benchmark ragtruth_50``) without each call site importing
every adapter module — and without each call site needing to know
which adapters require post-construction configuration.

The registry holds **zero-arg factories**, not bare classes. A class
is itself a zero-arg callable, so adapters that need no configuration
can register their class directly. Adapters that require post-
construction setup (currently only ``ragtruth_50``, which must apply
``set_benchmark_filter`` to restrict the corpus to the canonical
50-case slice) register a closure that returns a fully configured
instance. ``build(name)`` invokes the factory and returns the instance.

The registry is additive: existing direct-instantiation call sites
(``RAGTruthAdapter()`` and friends in ``benchmarks/*_science_gate.py``,
``benchmarks/run_all.py``) keep working unchanged. The registry only
adds the lookup function — it does not replace direct construction.

Canonical names registered at module import:

  ragtruth_50  → RAGTruthAdapter() with set_benchmark_filter applied
  halueval     → HaluEvalAdapter
  fever        → FEVERAdapter
  ifeval       → IFEvalAdapter
  jigsaw       → JigsawAdapter
  toxigen      → ToxiGenAdapter
  faithdial    → FaithDialAdapter
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from llm_judge.benchmarks import BenchmarkAdapter

# Canonical RAGTruth-50 benchmark definition. This path is the single
# source of truth — tooling that needs the filter file should import
# this constant rather than re-declaring it.
RAGTRUTH_50_BENCHMARK_PATH = Path(
    "datasets/benchmarks/ragtruth/ragtruth_50_benchmark.json"
)


class BenchmarkNotFoundError(Exception):
    """Raised by :func:`build` when a benchmark name is not registered."""


_REGISTRY: dict[str, Callable[[], BenchmarkAdapter]] = {}


def register(name: str, factory: Callable[[], BenchmarkAdapter]) -> None:
    """Register a zero-arg factory under ``name``. Last write wins.

    A class is itself a zero-arg callable returning an instance, so
    adapters that need no configuration can pass the class directly:
    ``register('halueval', HaluEvalAdapter)``. Adapters that need
    post-construction setup pass a closure that returns the configured
    instance.
    """
    _REGISTRY[name] = factory


def build(name: str) -> BenchmarkAdapter:
    """Invoke the factory registered under ``name`` and return the instance.

    Raises :class:`BenchmarkNotFoundError` listing the available
    names when ``name`` is not present — callers commonly surface
    this directly to the user.
    """
    if name not in _REGISTRY:
        raise BenchmarkNotFoundError(
            f"benchmark {name!r} not registered. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]()


def list_benchmarks() -> list[str]:
    """Return the registered benchmark names, sorted alphabetically."""
    return sorted(_REGISTRY.keys())


# --- Canonical-name registrations ---------------------------------------
# Imports kept at module-bottom so consumers who only need register/build/
# list_benchmarks for a custom registration don't pay for loading every
# adapter module up front.

from llm_judge.benchmarks.faithdial import FaithDialAdapter  # noqa: E402
from llm_judge.benchmarks.fever import FEVERAdapter  # noqa: E402
from llm_judge.benchmarks.halueval import HaluEvalAdapter  # noqa: E402
from llm_judge.benchmarks.ifeval import IFEvalAdapter  # noqa: E402
from llm_judge.benchmarks.jigsaw import JigsawAdapter  # noqa: E402
from llm_judge.benchmarks.ragtruth import RAGTruthAdapter  # noqa: E402
from llm_judge.benchmarks.toxigen import ToxiGenAdapter  # noqa: E402


def _build_ragtruth_50() -> BenchmarkAdapter:
    """Build a RAGTruthAdapter restricted to the canonical 50-case slice."""
    adapter = RAGTruthAdapter()
    adapter.set_benchmark_filter(RAGTRUTH_50_BENCHMARK_PATH)
    return adapter


register("ragtruth_50", _build_ragtruth_50)
register("halueval", HaluEvalAdapter)
register("fever", FEVERAdapter)
register("ifeval", IFEvalAdapter)
register("jigsaw", JigsawAdapter)
register("toxigen", ToxiGenAdapter)
register("faithdial", FaithDialAdapter)
