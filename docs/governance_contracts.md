
# Governance Contracts – Milestone A

## Manifest v2 Requirements
- manifest_version: "v2"
- dataset_sha256
- rubric_id
- rubric_version
- rubric_sha256

## Rubric Schema Enforcement
All rubric YAML files must conform to RubricConfig schema.
Invalid schema should fail in CI.

## Versioning Policy
Any scoring behavior change must bump rubric version.
