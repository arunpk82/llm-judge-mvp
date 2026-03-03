from __future__ import annotations

import random
from typing import Any


class SamplingError(ValueError):
    """Raised when the sampling configuration is invalid."""


def deterministic_sample_rows(
    *,
    rows: list[dict[str, Any]],
    seed: int,
    sample_size: int,
    strategy: str = "case_id_random",
) -> list[dict[str, Any]]:
    """Return a deterministic subset of rows.

    Determinism contract:
      - Rows are first sorted by case_id (stable ordering).
      - Sampling uses a local RNG: random.Random(seed).

    Strategies:
      - case_id_random: seeded random sample
      - head: first N
    """

    if sample_size <= 0:
        raise SamplingError(f"sample_size must be > 0, got {sample_size}")

    if any("case_id" not in r for r in rows):
        raise SamplingError("All rows must contain 'case_id' for deterministic sampling")

    ordered = sorted(rows, key=lambda r: str(r["case_id"]))

    if sample_size > len(ordered):
        raise SamplingError(
            f"sample_size ({sample_size}) is larger than dataset size ({len(ordered)})"
        )

    if strategy == "head":
        return ordered[:sample_size]

    if strategy == "case_id_random":
        rng = random.Random(seed)
        # sample returns items in selection order; we re-sort by case_id to
        # keep downstream iteration stable and diff-friendly.
        picked = rng.sample(ordered, sample_size)
        return sorted(picked, key=lambda r: str(r["case_id"]))

    raise SamplingError(f"Unknown sample_strategy: {strategy}")