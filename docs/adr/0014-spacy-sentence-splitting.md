---
status: accepted
date: 2026-04-16
deciders: [Arun]
category: B — Baseline & Eval
---

# ADR-0014: spaCy sentence splitting (not regex)

## Context and Problem Statement

The current production code at `src/llm_judge/calibration/hallucination.py:37`
splits sentences with:

```python
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
```

This regex splits on every period followed by whitespace, which is wrong
for abbreviations ("J. Paul Getty"), titles ("Dr. Smith"), and initials
("D. C."). Case `ragtruth_6` contains "J. Paul Getty" and the regex
splits it into two sentences, one of which becomes a false positive
hallucination flag.

The benchmark loader (`experiments/benchmark_loader.py`) already uses
spaCy, and the benchmark file itself mandates spaCy splitting. Production
code does not.

## Decision Drivers

- Correctness: false positives from sentence-splitting errors contaminate
  every downstream precision metric.
- Consistency: the benchmark loader and the production code must split
  the same way, or sentence indices will disagree between run and
  baseline.
- Cost: spaCy is already a pipeline dependency (L2 graph traversal).
  Using it here adds no new dependency.

## Considered Options

1. **Keep regex, add abbreviation list** — patch the regex to ignore
   known abbreviations.
2. **spaCy `doc.sents`** — use the spaCy sentencizer or dependency parser.
3. **NLTK `sent_tokenize`** — add NLTK as a dependency.

## Decision Outcome

**Chosen option: spaCy `doc.sents` (Option 2).**

Replace the regex in `_split_sentences()` with spaCy. Lazy-load the
spaCy pipeline (pattern already used in `benchmark_loader.py`) so cold-
start callers do not pay the load cost unless sentence splitting is
invoked.

## Consequences

### Positive

- Abbreviation false positives disappear.
- Benchmark loader and production code use the same splitter.
- No new dependency.

### Negative

- Cold-start latency increases on the first sentence-split call
  (~200–500 ms for spaCy load). Lazy loading keeps this out of the
  hot path for callers that never hit this code.
- spaCy version is now part of the reproducibility contract; a model
  upgrade could change split boundaries.

## More Information

- Bug surfaced by: `ragtruth_6` (J. Paul Getty)
- Location of fix: `src/llm_judge/calibration/hallucination.py:37-42`
- Related: ADR-0013 (benchmark mandates spaCy)
