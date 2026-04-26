"""Batch input file schema.

Lets callers run ``make demo-batch-file FILE=...`` against an ad-hoc
list of cases without writing a benchmark adapter. The file may be
JSON or YAML; ``schema_version`` lets us evolve the format.

Schema (v1):

    schema_version: 1
    cases:
      - case_id: <str>
        response: <str>
        source:   <str>           # optional, but required at run time
        metadata: <dict>          # optional, free-form
    metadata: <dict>              # optional, batch-level

Validation lives in pydantic; ``load_batch_file`` wraps validation
errors in :class:`BatchInputSchemaError` so the CLI driver can
surface them with a stable error type.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ValidationError


class BatchCase(BaseModel):
    """One ad-hoc evaluation case in a batch input file.

    ``source`` is optional in the schema (so the file is editable
    in passes), but the CLI driver will reject any case whose source
    is missing before invoking the Runner — ``SingleEvaluationRequest``
    requires it.
    """

    case_id: str
    response: str
    source: str | None = None
    metadata: dict[str, Any] | None = None


class BatchInputFile(BaseModel):
    """Top-level batch input file.

    ``schema_version`` defaults to 1; bumping the field lets future
    formats coexist while older files keep loading.
    """

    schema_version: int = 1
    cases: list[BatchCase]
    metadata: dict[str, Any] | None = None


class BatchInputSchemaError(ValueError):
    """Raised when a batch input file fails schema validation."""


def load_batch_file(path: Path | str) -> BatchInputFile:
    """Load a batch input file from disk.

    Dispatches on the file extension: ``.yaml``/``.yml`` go through
    PyYAML, ``.json`` through the stdlib ``json`` parser. Any other
    extension is rejected up front so we don't silently misparse.

    Validation errors and parse errors both surface as
    :class:`BatchInputSchemaError` with the source path included.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise BatchInputSchemaError(f"{path}: {e}") from e

    if suffix in (".yaml", ".yml"):
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise BatchInputSchemaError(f"{path}: {e}") from e
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise BatchInputSchemaError(f"{path}: {e}") from e
    else:
        raise BatchInputSchemaError(
            f"{path}: unsupported extension {suffix!r}; "
            f"expected .yaml, .yml, or .json"
        )

    if data is None:
        raise BatchInputSchemaError(f"{path}: file is empty")

    try:
        return BatchInputFile.model_validate(data)
    except ValidationError as e:
        raise BatchInputSchemaError(f"{path}: {e}") from e
