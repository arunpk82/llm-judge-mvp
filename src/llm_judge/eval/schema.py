from __future__ import annotations

EVAL_RUN_SCHEMA_VERSION = "1.0"
DIFF_REPORT_SCHEMA_VERSION = "1.0"
POLICY_RESULT_SCHEMA_VERSION = "1.0"


def assert_compatible_schema(
    *,
    baseline_version: str | None,
    candidate_version: str | None,
) -> None:
    """
    Enforce strict schema compatibility.

    Production-grade rule:
    - Missing version = fail
    - Mismatch = fail
    """
    if not baseline_version or not candidate_version:
        raise ValueError(
            f"Missing schema_version (baseline={baseline_version}, candidate={candidate_version})"
        )

    if baseline_version != candidate_version:
        raise ValueError(
            f"Schema version mismatch: baseline={baseline_version}, candidate={candidate_version}"
        )