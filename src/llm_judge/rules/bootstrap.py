"""
Import modules for side-effect registration via @register decorators.

Keep this file tiny and deterministic.
"""

from __future__ import annotations

import llm_judge.rules.correctness.definition_sanity  # noqa: F401

# Register compatibility/basic correctness rule
import llm_judge.rules.correctness_basic  # noqa: F401

# Register Phase 3 deterministic rules
import llm_judge.rules.quality.nonsense_basic  # noqa: F401
import llm_judge.rules.quality.repetition_basic  # noqa: F401
