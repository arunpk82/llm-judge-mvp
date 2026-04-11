"""
Prompt Versioning (L4 Step 3).

Every prompt change IS a scoring change. Prompts are versioned like rule
plans — immutable files in configs/prompts/{prompt_id}/v{N}.yaml.

Provides: load, list, diff between versions, and event emission on change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from llm_judge.paths import config_root

logger = logging.getLogger(__name__)

PROMPTS_DIR = config_root() / "prompts"


@dataclass(frozen=True)
class PromptTemplate:
    """A versioned adjudication prompt template."""

    prompt_id: str
    version: str
    system_prompt: str
    user_template: str
    dimensions: list[str]
    created_at: str
    author: str


def load_prompt(
    prompt_id: str, version: str, prompts_dir: Path = PROMPTS_DIR
) -> PromptTemplate:
    """Load a specific prompt version."""
    path = prompts_dir / prompt_id / f"{version}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Prompt must be a YAML mapping: {path}")

    return PromptTemplate(
        prompt_id=str(raw.get("prompt_id", prompt_id)),
        version=str(raw.get("version", version)),
        system_prompt=str(raw.get("system_prompt", "")),
        user_template=str(raw.get("user_template", "")),
        dimensions=list(raw.get("dimensions", [])),
        created_at=str(raw.get("created_at", "")),
        author=str(raw.get("author", "unknown")),
    )


def load_latest_prompt(
    prompt_id: str, prompts_dir: Path = PROMPTS_DIR
) -> PromptTemplate:
    """Load the highest-numbered version for a prompt_id."""
    prompt_dir = prompts_dir / prompt_id
    if not prompt_dir.exists():
        raise FileNotFoundError(f"No prompts found for: {prompt_id}")

    versions = sorted(prompt_dir.glob("v*.yaml"))
    if not versions:
        raise FileNotFoundError(f"No versioned prompts in: {prompt_dir}")

    latest = versions[-1]
    version = latest.stem  # e.g., "v1"
    return load_prompt(prompt_id, version, prompts_dir)


def list_prompts(prompts_dir: Path = PROMPTS_DIR) -> list[dict[str, Any]]:
    """List all registered prompts with their versions."""
    results: list[dict[str, Any]] = []
    if not prompts_dir.exists():
        return results

    for prompt_dir in sorted(prompts_dir.iterdir()):
        if not prompt_dir.is_dir():
            continue
        versions = sorted(prompt_dir.glob("v*.yaml"))
        for v_path in versions:
            try:
                pt = load_prompt(prompt_dir.name, v_path.stem, prompts_dir)
                results.append(
                    {
                        "prompt_id": pt.prompt_id,
                        "version": pt.version,
                        "dimensions": pt.dimensions,
                        "created_at": pt.created_at,
                        "author": pt.author,
                    }
                )
            except Exception:
                continue

    return results


def diff_prompts(
    prompt_id: str,
    version_a: str,
    version_b: str,
    prompts_dir: Path = PROMPTS_DIR,
) -> dict[str, Any]:
    """
    Diff two prompt versions. Returns field-level changes.

    This is critical because a prompt change IS a scoring change —
    we need to track exactly what changed between versions.
    """
    a = load_prompt(prompt_id, version_a, prompts_dir)
    b = load_prompt(prompt_id, version_b, prompts_dir)

    changes: list[dict[str, Any]] = []

    if a.system_prompt != b.system_prompt:
        changes.append(
            {
                "field": "system_prompt",
                "type": "modified",
                "old_length": len(a.system_prompt),
                "new_length": len(b.system_prompt),
            }
        )

    if a.user_template != b.user_template:
        changes.append(
            {
                "field": "user_template",
                "type": "modified",
                "old_length": len(a.user_template),
                "new_length": len(b.user_template),
            }
        )

    if set(a.dimensions) != set(b.dimensions):
        added = set(b.dimensions) - set(a.dimensions)
        removed = set(a.dimensions) - set(b.dimensions)
        changes.append(
            {
                "field": "dimensions",
                "type": "modified",
                "added": sorted(added),
                "removed": sorted(removed),
            }
        )

    return {
        "prompt_id": prompt_id,
        "version_a": version_a,
        "version_b": version_b,
        "changes": changes,
        "has_changes": len(changes) > 0,
    }


def render_prompt(
    template: PromptTemplate,
    conversation: str,
    candidate_answer: str,
) -> tuple[str, str]:
    """
    Render a prompt template into system + user messages.

    Returns (system_message, user_message).
    """
    user_msg = template.user_template.format(
        conversation=conversation,
        candidate_answer=candidate_answer,
        dimensions=", ".join(template.dimensions),
    )
    return template.system_prompt.strip(), user_msg.strip()
