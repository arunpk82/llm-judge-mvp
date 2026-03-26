from __future__ import annotations

import json
import textwrap
from pathlib import Path

import yaml

from llm_judge.dataset_validator import (
    ValidationError,
    ValidationResult,
    validate_cases_file,
    validate_dataset,
    validate_manifest,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_CASE: dict = {
    "id": "case-001",
    "case_id": "case-001",
    "conversation": [{"role": "user", "content": "Hello"}],
    "candidate_answer": "Hi there",
    "rubric_id": "chat_quality",
    "human_decision": "pass",
}

VALID_MANIFEST_YAML = textwrap.dedent("""\
    schema_version: "1"
    name: test_dataset
    version: v1
    cases_file: cases.jsonl
    format: jsonl
""")


def write_jsonl(path: Path, cases: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")


# ---------------------------------------------------------------------------
# validate_manifest
# ---------------------------------------------------------------------------


def test_validate_manifest_valid(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text(VALID_MANIFEST_YAML)
    result = validate_manifest(manifest)
    assert result.valid
    assert result.errors == []


def test_validate_manifest_malformed_yaml_has_line_number(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("name: [\nbad yaml: {unclosed")
    result = validate_manifest(manifest)
    assert not result.valid
    err = next(e for e in result.errors if e.code == "MALFORMED_YAML")
    assert err.line is not None
    assert err.line >= 1


def test_validate_manifest_empty_file(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("")
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(e.code == "EMPTY_MANIFEST" for e in result.errors)


def test_validate_manifest_missing_required_field(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    # cases_file is required but omitted
    manifest.write_text("schema_version: '1'\nname: test\nversion: v1\n")
    result = validate_manifest(manifest)
    assert not result.valid
    schema_errors = [e for e in result.errors if e.code == "SCHEMA_ERROR"]
    assert schema_errors
    assert any("cases_file" in e.message for e in schema_errors)


def test_validate_manifest_invalid_format_enum(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text(VALID_MANIFEST_YAML.replace("format: jsonl", "format: csv"))
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(e.code == "SCHEMA_ERROR" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_cases_file — JSONL
# ---------------------------------------------------------------------------


def test_validate_cases_file_valid_jsonl(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    write_jsonl(f, [VALID_CASE])
    result = validate_cases_file(f, "jsonl")
    assert result.valid
    assert result.errors == []


def test_validate_cases_file_multiple_valid_cases(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    cases = [
        {**VALID_CASE, "id": f"case-{i:03d}", "case_id": f"case-{i:03d}", "human_decision": "pass" if i % 2 == 0 else "fail"}
        for i in range(1, 6)
    ]
    write_jsonl(f, cases)
    result = validate_cases_file(f, "jsonl")
    assert result.valid


def test_validate_cases_file_empty_dataset(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text("")
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    assert any(e.code == "EMPTY_DATASET" for e in result.errors)


def test_validate_cases_file_blank_lines_skipped(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text("\n\n" + json.dumps(VALID_CASE) + "\n\n")
    result = validate_cases_file(f, "jsonl")
    assert result.valid


def test_validate_cases_file_duplicate_ids(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    write_jsonl(f, [
        {**VALID_CASE, "id": "dup-001", "case_id": "cid-001"},
        {**VALID_CASE, "id": "dup-001", "case_id": "cid-002"},
    ])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    dup = next(e for e in result.errors if e.code == "DUPLICATE_ID")
    assert "dup-001" in dup.message


def test_validate_cases_file_no_ids_no_duplicate_check(tmp_path: Path) -> None:
    """Cases without 'id' should not trigger legacy duplicate detection.
    case_id is still required and must be unique."""
    f = tmp_path / "cases.jsonl"
    case_no_id_1 = {k: v for k, v in VALID_CASE.items() if k != "id"}
    case_no_id_2 = {**case_no_id_1, "case_id": "case-002"}
    write_jsonl(f, [case_no_id_1, case_no_id_2])
    result = validate_cases_file(f, "jsonl")
    assert result.valid


def test_validate_cases_file_malformed_json_line(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text("{bad json\n")
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "MALFORMED_JSON")
    assert err.line == 1


# ---- EPIC-1.1: case_id completeness checks ----

def test_validate_missing_case_id(tmp_path: Path) -> None:
    """Rows without case_id are flagged — downstream sampling requires it."""
    f = tmp_path / "cases.jsonl"
    case_no_cid = {k: v for k, v in VALID_CASE.items() if k != "case_id"}
    write_jsonl(f, [case_no_cid])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "MISSING_CASE_ID")
    assert "case_id" in err.message
    assert "sampling" in err.message.lower()


def test_validate_duplicate_case_id(tmp_path: Path) -> None:
    """Duplicate case_ids are caught at validation time."""
    f = tmp_path / "cases.jsonl"
    write_jsonl(f, [
        {**VALID_CASE, "case_id": "dup-cid"},
        {**VALID_CASE, "id": "case-002", "case_id": "dup-cid"},
    ])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "DUPLICATE_CASE_ID")
    assert "dup-cid" in err.message


def test_validate_empty_case_id_is_missing(tmp_path: Path) -> None:
    """Empty string case_id counts as missing."""
    f = tmp_path / "cases.jsonl"
    write_jsonl(f, [{**VALID_CASE, "case_id": ""}])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    assert any(e.code == "MISSING_CASE_ID" for e in result.errors)


def test_validate_cases_file_malformed_json_reports_line_number(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    f.write_text(
        json.dumps(VALID_CASE) + "\n"
        + "{bad json on line 2\n"
    )
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "MALFORMED_JSON")
    assert err.line == 2


def test_validate_cases_file_missing_candidate_answer(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    bad = {k: v for k, v in VALID_CASE.items() if k != "candidate_answer"}
    write_jsonl(f, [bad])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "CASE_SCHEMA_ERROR")
    assert "candidate_answer" in err.message
    assert err.line == 1


def test_validate_cases_file_invalid_human_decision(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    write_jsonl(f, [{**VALID_CASE, "human_decision": "maybe"}])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    assert any(e.code == "CASE_SCHEMA_ERROR" for e in result.errors)


def test_validate_cases_file_invalid_role_in_conversation(tmp_path: Path) -> None:
    f = tmp_path / "cases.jsonl"
    bad = {
        **VALID_CASE,
        "conversation": [{"role": "system", "content": "You are helpful"}],
    }
    write_jsonl(f, [bad])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    assert any(e.code == "CASE_SCHEMA_ERROR" for e in result.errors)


def test_validate_cases_file_multiple_errors_all_reported(tmp_path: Path) -> None:
    """All validation errors across multiple cases are collected, not just the first."""
    f = tmp_path / "cases.jsonl"
    bad1 = {k: v for k, v in VALID_CASE.items() if k != "candidate_answer"}
    bad2 = {**VALID_CASE, "id": "x", "human_decision": "invalid"}
    write_jsonl(f, [bad1, bad2])
    result = validate_cases_file(f, "jsonl")
    assert not result.valid
    assert len(result.errors) >= 2


def test_validate_cases_file_not_found(tmp_path: Path) -> None:
    result = validate_cases_file(tmp_path / "missing.jsonl", "jsonl")
    assert not result.valid
    assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)


def test_validate_cases_file_unsupported_format(tmp_path: Path) -> None:
    f = tmp_path / "cases.csv"
    f.write_text("id,name\n1,test\n")
    result = validate_cases_file(f, "csv")
    assert not result.valid
    assert any(e.code == "UNSUPPORTED_FORMAT" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_cases_file — YAML
# ---------------------------------------------------------------------------


def test_validate_cases_file_valid_yaml(tmp_path: Path) -> None:
    f = tmp_path / "cases.yaml"
    f.write_text(yaml.dump([VALID_CASE]))
    result = validate_cases_file(f, "yaml")
    assert result.valid


def test_validate_cases_file_malformed_yaml_has_line_number(tmp_path: Path) -> None:
    f = tmp_path / "cases.yaml"
    f.write_text("- id: test\n  bad: [\n  unclosed")
    result = validate_cases_file(f, "yaml")
    assert not result.valid
    err = next(e for e in result.errors if e.code == "MALFORMED_YAML")
    assert err.line is not None


def test_validate_cases_file_yaml_not_a_list(tmp_path: Path) -> None:
    f = tmp_path / "cases.yaml"
    f.write_text("key: value\n")
    result = validate_cases_file(f, "yaml")
    assert not result.valid
    assert any(e.code == "CASE_FORMAT_ERROR" for e in result.errors)


def test_validate_cases_file_yaml_empty_list(tmp_path: Path) -> None:
    f = tmp_path / "cases.yaml"
    f.write_text("[]\n")
    result = validate_cases_file(f, "yaml")
    assert not result.valid
    assert any(e.code == "EMPTY_DATASET" for e in result.errors)


def test_validate_cases_file_yaml_duplicate_ids(tmp_path: Path) -> None:
    f = tmp_path / "cases.yaml"
    f.write_text(yaml.dump([{**VALID_CASE, "id": "dup"}, {**VALID_CASE, "id": "dup"}]))
    result = validate_cases_file(f, "yaml")
    assert not result.valid
    assert any(e.code == "DUPLICATE_ID" for e in result.errors)


# ---------------------------------------------------------------------------
# validate_dataset (manifest + cases)
# ---------------------------------------------------------------------------


def test_validate_dataset_valid(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "cases.jsonl", [VALID_CASE])
    (tmp_path / "dataset.yaml").write_text(VALID_MANIFEST_YAML)
    result = validate_dataset(tmp_path / "dataset.yaml")
    assert result.valid
    assert result.errors == []


def test_validate_dataset_manifest_not_found(tmp_path: Path) -> None:
    result = validate_dataset(tmp_path / "no_manifest.yaml")
    assert not result.valid
    assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)


def test_validate_dataset_cases_file_not_found(tmp_path: Path) -> None:
    # Manifest is valid but references a cases file that doesn't exist
    (tmp_path / "dataset.yaml").write_text(VALID_MANIFEST_YAML)
    result = validate_dataset(tmp_path / "dataset.yaml")
    assert not result.valid
    assert any(e.code == "FILE_NOT_FOUND" for e in result.errors)


def test_validate_dataset_malformed_manifest_stops_early(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text("{bad yaml\n")
    result = validate_dataset(tmp_path / "dataset.yaml")
    assert not result.valid
    assert any(e.code == "MALFORMED_YAML" for e in result.errors)
    # Should not have proceeded to cases validation — no FILE_NOT_FOUND error
    assert not any(e.code == "FILE_NOT_FOUND" for e in result.errors)


def test_validate_dataset_invalid_cases_propagates(tmp_path: Path) -> None:
    (tmp_path / "dataset.yaml").write_text(VALID_MANIFEST_YAML)
    bad_case = {k: v for k, v in VALID_CASE.items() if k != "candidate_answer"}
    write_jsonl(tmp_path / "cases.jsonl", [bad_case])
    result = validate_dataset(tmp_path / "dataset.yaml")
    assert not result.valid
    assert any(e.code == "CASE_SCHEMA_ERROR" for e in result.errors)


# ---------------------------------------------------------------------------
# ValidationResult and ValidationError dataclass contracts
# ---------------------------------------------------------------------------


def test_validation_result_defaults_to_empty_errors() -> None:
    r = ValidationResult(valid=True)
    assert r.errors == []


def test_validation_error_line_is_optional() -> None:
    e = ValidationError(code="TEST", message="msg", file="f.yaml")
    assert e.line is None


def test_validation_error_with_line() -> None:
    e = ValidationError(code="TEST", message="msg", file="f.yaml", line=42)
    assert e.line == 42
