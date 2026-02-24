"""
Compatibility layer.

This module previously defined core rule types. The canonical definitions now
live in llm_judge.rules.types.

Keep importing from llm_judge.rules.base working for older code/tests.
"""

from __future__ import annotations

from llm_judge.rules.types import Flag, RuleContext, RuleResult

__all__ = ["Flag", "RuleContext", "RuleResult"]
