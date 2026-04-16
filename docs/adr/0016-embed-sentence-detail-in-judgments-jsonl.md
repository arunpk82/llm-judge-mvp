---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0016: Embed sentence detail inside judgments.jsonl

## Context and Problem Statement

Hallucination cascade evaluation produces sentence-level detail (which
layer resolved each sentence, what verdict, what evidence) but the
user-facing unit is the response (ADR-0005). The existing `judgments.jsonl`
contract (ADR-0012) is per-response. Where should sentence detail live?

Two shapes are possible:

1. Add a new artifact, `sentences.jsonl`, one row per sentence.
2. Embed sentence detail inside each `judgments.jsonl` row as a nested
   array.

The blast radius differs considerably. `src/llm_judge/eval/baseline.py:376`
hardcodes the three-file copy in `baseline-promote`. Adding a fourth
file means extending the promote, validate, and (possibly) registry
code. Embedding means adding a field to the row, which the existing
diff engine ignores because it only flips on `judge_decision`.

## Decision Drivers

- Minimise contract changes: the artifact triplet (ADR-0012) is
  load-bearing across the platform.
- Preserve sentence-level diagnostics; they are irreplaceable for
  cascade analysis.
- Reversibility: we can split into a separate file later if query
  patterns demand it; we cannot easily unwind a contract change.

## Considered Options

1. **New `sentences.jsonl` artifact** — clean separation, one row per
   sentence, new file in the triplet → quadruplet.
2. **Embed in `judgments.jsonl`** — each row has a `sentence_detail: [...]`
   array with per-sentence records.
3. **Separate file, not in triplet** — `sentences.jsonl` written
   alongside but not promoted as part of the baseline.

## Decision Outcome

**Chosen option: embed in `judgments.jsonl` (Option 2).**

Each row in `judgments.jsonl` adds `sentence_detail: [...]` containing
per-sentence records: sentence text, ground truth label, resolving layer,
final verdict, confidence, and per-layer evidence.

Row size grows from ~500 B to ~5–50 KB depending on evidence depth.
For RAGTruth-50 (50 responses) the file stays under 2.5 MB. The existing
diff engine ignores additional fields; the contract is preserved.

## Consequences

### Positive

- Zero change to the artifact triplet contract.
- Zero change to `baseline.py`, `diff.py`, or the registry.
- Sentence detail travels with the response that contains it; no join
  needed.

### Negative

- Row size is larger; this is not a concern at RAGTruth-50 scale but
  may be at 30 000-case scale. If it becomes a query bottleneck, we
  split (writing a new superseding ADR).
- Consumers wanting sentence-level analysis must parse nested JSON;
  simple `jq` over flat rows no longer works.

## More Information

- Related: ADR-0012 (artifact triplet), ADR-0005 (dual evaluation
  units), ADR-0015 (metrics schema)
- Revisit trigger: if total `judgments.jsonl` size crosses ~100 MB or
  if sentence-level queries become a frequent operator pattern.
