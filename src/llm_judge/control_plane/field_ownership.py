"""Per-capability field-ownership allowlist (CP-F3 closure).

Each capability is the only one allowed to stamp a given envelope
field. ``ProvenanceEnvelope.stamped(capability=..., **fields)`` looks
up ``FIELD_OWNERSHIP[capability]`` and raises
``FieldOwnershipViolationError`` if any key in ``fields`` is absent
from that frozenset. This converts the prior convention-only rule
(End-State property A3.3) into a runtime gate.

The allowlist is constructed strict-from-day-one to match every
production ``stamped()`` call site at packet drafting time:

  * CAP-1 owns dataset and benchmark provenance fields.
  * CAP-2 owns rule-engine outcome fields.
  * CAP-7 and CAP-5 stamp the capability chain only and therefore
    pass an empty kwargs dict — their entries are empty frozensets.

Future capabilities extend this map alongside the wrappers that
stamp the fields. Empty frozensets are intentional: they encode
"chain stamping only" rather than "capability not yet wired".
"""

from __future__ import annotations

FIELD_OWNERSHIP: dict[str, frozenset[str]] = {
    "CAP-1": frozenset(
        {
            # Per-case dataset lineage (existing — wrappers.py:154)
            "dataset_registry_id",
            "input_hash",
            # Benchmark provenance (CP-F1 extension — wrappers.py:154 post-packet)
            "benchmark_id",
            "benchmark_version",
            "benchmark_content_hash",
            "benchmark_registration_timestamp",
        }
    ),
    "CAP-2": frozenset({"rule_set_version", "rules_fired"}),
    "CAP-7": frozenset(),
    "CAP-5": frozenset(),
}
