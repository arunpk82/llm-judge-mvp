from __future__ import annotations

import json
import re as _re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError as PydanticValidationError

from llm_judge.dataset_models import DatasetCase, DatasetManifest


@dataclass
class ValidationError:
    code: str
    message: str
    file: str
    line: Optional[int] = None


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)


def validate_manifest(path: Path) -> ValidationResult:
    """Validate a dataset.yaml manifest against the DatasetManifest schema."""
    errors: list[ValidationError] = []

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        line: Optional[int] = None
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            line = exc.problem_mark.line + 1
        problem = getattr(exc, "problem", str(exc))
        errors.append(
            ValidationError(
                code="MALFORMED_YAML",
                message=f"Malformed YAML: {problem}",
                file=str(path),
                line=line,
            )
        )
        return ValidationResult(valid=False, errors=errors)

    if data is None:
        errors.append(
            ValidationError(
                code="EMPTY_MANIFEST",
                message="Manifest file is empty",
                file=str(path),
            )
        )
        return ValidationResult(valid=False, errors=errors)

    try:
        DatasetManifest.model_validate(data)
    except PydanticValidationError as exc:
        for err in exc.errors():
            field_path = ".".join(str(p) for p in err["loc"])
            errors.append(
                ValidationError(
                    code="SCHEMA_ERROR",
                    message=f"Field '{field_path}': {err['msg']}",
                    file=str(path),
                )
            )

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def _parse_and_validate_cases_jsonl(
    path: Path,
) -> tuple[list[dict[str, Any]], list[ValidationError]]:
    """Parse a JSONL cases file, validate each case, return (raw_cases, errors)."""
    errors: list[ValidationError] = []
    raw_cases: list[dict[str, Any]] = []

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [], [
            ValidationError(code="FILE_NOT_FOUND", message=str(exc), file=str(path))
        ]

    for line_num, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue

        try:
            data: dict[str, Any] = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(
                ValidationError(
                    code="MALFORMED_JSON",
                    message=f"Invalid JSON: {exc.msg}",
                    file=str(path),
                    line=line_num,
                )
            )
            continue

        raw_cases.append(data)

        try:
            DatasetCase.model_validate(data)
        except PydanticValidationError as exc:
            for err in exc.errors():
                field_path = ".".join(str(p) for p in err["loc"])
                errors.append(
                    ValidationError(
                        code="CASE_SCHEMA_ERROR",
                        message=f"Field '{field_path}': {err['msg']}",
                        file=str(path),
                        line=line_num,
                    )
                )

    return raw_cases, errors


def _parse_and_validate_cases_yaml(
    path: Path,
) -> tuple[list[dict[str, Any]], list[ValidationError]]:
    """Parse a YAML cases file, validate each case, return (raw_cases, errors)."""
    errors: list[ValidationError] = []

    try:
        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        line: Optional[int] = None
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            line = exc.problem_mark.line + 1
        problem = getattr(exc, "problem", str(exc))
        errors.append(
            ValidationError(
                code="MALFORMED_YAML",
                message=f"Malformed YAML: {problem}",
                file=str(path),
                line=line,
            )
        )
        return [], errors

    if not isinstance(data, list):
        errors.append(
            ValidationError(
                code="CASE_FORMAT_ERROR",
                message="YAML cases file must be a list of cases",
                file=str(path),
            )
        )
        return [], errors

    raw_cases: list[dict[str, Any]] = []
    for i, item in enumerate(data, start=1):
        raw_cases.append(item)
        try:
            DatasetCase.model_validate(item)
        except PydanticValidationError as exc:
            for err in exc.errors():
                field_path = ".".join(str(p) for p in err["loc"])
                errors.append(
                    ValidationError(
                        code="CASE_SCHEMA_ERROR",
                        message=f"Case {i}, field '{field_path}': {err['msg']}",
                        file=str(path),
                        line=i,
                    )
                )

    return raw_cases, errors


def _check_integrity(cases: list[dict[str, Any]], path: Path) -> list[ValidationError]:
    """Integrity checks: empty dataset, duplicate IDs, case_id completeness."""
    errors: list[ValidationError] = []

    if not cases:
        errors.append(
            ValidationError(
                code="EMPTY_DATASET",
                message="Dataset contains no cases",
                file=str(path),
            )
        )
        return errors

    # Check for duplicate IDs (legacy 'id' field)
    seen: dict[str, list[int]] = {}
    for i, case in enumerate(cases, start=1):
        case_id = case.get("id")
        if case_id is not None:
            key = str(case_id)
            seen.setdefault(key, []).append(i)

    for case_id, lines in seen.items():
        if len(lines) > 1:
            errors.append(
                ValidationError(
                    code="DUPLICATE_ID",
                    message=f"Duplicate case ID '{case_id}' found at lines: {lines}",
                    file=str(path),
                )
            )

    # EPIC-1.1: case_id completeness check.
    # Downstream consumers (sampling, judgments, diff tracking) require
    # non-empty case_id on every row. Catch this at validation time
    # instead of crashing at sampling time.
    missing_case_id: list[int] = []
    duplicate_case_ids: dict[str, list[int]] = {}
    for i, case in enumerate(cases, start=1):
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid.strip():
            missing_case_id.append(i)
        else:
            duplicate_case_ids.setdefault(cid.strip(), []).append(i)

    if missing_case_id:
        errors.append(
            ValidationError(
                code="MISSING_CASE_ID",
                message=(
                    f"Missing or empty 'case_id' at {len(missing_case_id)} row(s): "
                    f"{missing_case_id[:10]}"
                    + (" ..." if len(missing_case_id) > 10 else "")
                    + ". Deterministic sampling requires case_id on every row."
                ),
                file=str(path),
            )
        )

    for cid, lines in duplicate_case_ids.items():
        if len(lines) > 1:
            errors.append(
                ValidationError(
                    code="DUPLICATE_CASE_ID",
                    message=f"Duplicate case_id '{cid}' found at lines: {lines}",
                    file=str(path),
                )
            )

    return errors


# =====================================================================
# TASK-1.1.3: Security scanning
# =====================================================================

# Patterns that indicate potential injection attacks in dataset content.
# These catch prompt injection, system prompt overrides, and common
# jailbreak patterns that could poison evaluation results.
_INJECTION_PATTERNS: list[tuple[str, _re.Pattern[str]]] = [
    (
        "SYSTEM_OVERRIDE",
        _re.compile(
            r"(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)",
            _re.IGNORECASE,
        ),
    ),
    (
        "ROLE_INJECTION",
        _re.compile(
            r"\[\s*(?:SYSTEM|INST|SYS)\s*\]|\{\{?\s*(?:system|role)\s*[:=]",
            _re.IGNORECASE,
        ),
    ),
    (
        "JAILBREAK_PATTERN",
        _re.compile(
            r"(?:DAN|do\s+anything\s+now|you\s+are\s+now|act\s+as\s+if|pretend\s+you\s+are)\s+",
            _re.IGNORECASE,
        ),
    ),
    (
        "XML_INJECTION",
        _re.compile(
            r"<\s*/?(?:script|iframe|object|embed|form|input)\b",
            _re.IGNORECASE,
        ),
    ),
]


def _check_security(cases: list[dict[str, Any]], path: Path) -> list[ValidationError]:
    """
    TASK-1.1.3: Scan dataset content fields for injection patterns.

    Checks conversation content and candidate_answer fields for patterns
    that could poison evaluation: prompt injection, role overrides,
    jailbreak attempts, and HTML/script injection.

    Returns warnings (not hard failures) — flagged rows should be
    reviewed by a human before inclusion in evaluation datasets.
    """
    warnings: list[ValidationError] = []

    for i, case in enumerate(cases, start=1):
        texts_to_scan: list[str] = []

        # Collect text from conversation messages
        conversation = case.get("conversation", [])
        if isinstance(conversation, list):
            for msg in conversation:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        texts_to_scan.append(content)

        # Collect candidate_answer
        answer = case.get("candidate_answer", "")
        if isinstance(answer, str):
            texts_to_scan.append(answer)

        # Scan all collected text
        for text in texts_to_scan:
            for pattern_name, pattern in _INJECTION_PATTERNS:
                if pattern.search(text):
                    warnings.append(
                        ValidationError(
                            code=f"SECURITY_{pattern_name}",
                            message=(
                                f"Potential {pattern_name.lower().replace('_', ' ')} "
                                f"detected at row {i}. Review this row before "
                                f"including in evaluation dataset."
                            ),
                            file=str(path),
                            line=i,
                        )
                    )
                    break  # one warning per row is enough

    return warnings


def validate_cases_file(path: Path, fmt: str = "jsonl") -> ValidationResult:
    """Validate a raw cases file (JSONL or YAML) without a manifest."""
    if not path.exists():
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    code="FILE_NOT_FOUND",
                    message=f"Cases file not found: {path}",
                    file=str(path),
                )
            ],
        )

    if fmt == "jsonl":
        raw_cases, errors = _parse_and_validate_cases_jsonl(path)
    elif fmt == "yaml":
        raw_cases, errors = _parse_and_validate_cases_yaml(path)
    else:
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    code="UNSUPPORTED_FORMAT",
                    message=f"Unsupported format: '{fmt}'. Use 'jsonl' or 'yaml'.",
                    file=str(path),
                )
            ],
        )

    errors.extend(_check_integrity(raw_cases, path))
    errors.extend(_check_security(raw_cases, path))
    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_dataset(manifest_path: Path) -> ValidationResult:
    """Full validation: manifest schema + referenced cases file."""
    if not manifest_path.exists():
        return ValidationResult(
            valid=False,
            errors=[
                ValidationError(
                    code="FILE_NOT_FOUND",
                    message=f"Manifest not found: {manifest_path}",
                    file=str(manifest_path),
                )
            ],
        )

    manifest_result = validate_manifest(manifest_path)
    if not manifest_result.valid:
        return manifest_result

    text = manifest_path.read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    manifest = DatasetManifest.model_validate(raw)

    cases_path = manifest_path.parent / manifest.cases_file
    cases_result = validate_cases_file(cases_path, manifest.format)

    all_errors = manifest_result.errors + cases_result.errors
    return ValidationResult(valid=len(all_errors) == 0, errors=all_errors)
