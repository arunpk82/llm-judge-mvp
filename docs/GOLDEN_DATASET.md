# Golden Dataset (v1) — Contribution Guide

## Purpose
The golden dataset is the source of truth for evaluating the judge engine against a human baseline.
It is used by CI and local runs via:
- `configs/runspecs/pr_gate.yaml`
- `datasets/golden/v1.jsonl`

This program is designed to scale from tens → hundreds → thousands of cases while maintaining:
- Reproducibility
- Reviewability
- Label consistency

## Versioning
- v1 is the initial baseline and uses a single rubric: `chat_quality`.
- Backward-incompatible changes (schema shifts, rubric definition changes, scoring semantics changes) must create v2.

## File Layout
Contributors should add cases in shards to avoid merge conflicts:
- `datasets/golden/sources/v1/cases_0001_0025.jsonl`
- `datasets/golden/sources/v1/cases_0026_0050.jsonl`
- ...

The compiled dataset used by runners is:
- `datasets/golden/v1.jsonl`

In v1 we may update `v1.jsonl` directly (simple). When scaling beyond ~200 cases, introduce a compile step.

## Schema (one JSON object per line)
Required fields:
- `case_id` (string): globally unique, stable identifier
- `rubric_id` (string): MUST be `chat_quality` for v1
- `conversation` (list): list of `{ "role": "...", "content": "..." }`
- `candidate_answer` (string)
- `human_decision` (string): `"pass"` or `"fail"`

Optional fields:
- `human_scores` (object): numeric scores 1..5 for:
  - `relevance`, `clarity`, `correctness`, `tone`
- `tags` (list[string]): e.g. `["irrelevance", "rude", "multi_turn"]`
- `notes` (string): short labeling note (avoid sensitive data)

Example:
```json
{
  "case_id": "v1-0001",
  "rubric_id": "chat_quality",
  "conversation": [{"role": "user", "content": "How do I reset my router?"}],
  "candidate_answer": "Unplug it for 10 seconds, then plug it back in.",
  "human_decision": "pass",
  "human_scores": {"relevance": 5, "clarity": 4, "correctness": 4, "tone": 4},
  "tags": ["how_to", "networking"],
  "notes": "Direct and helpful."
}


Labeling Guidelines (v1 chat_quality)
	Decision: PASS
		Use pass when the answer is:
		Relevant to the user ask
		Correct (no major hallucinations)
		Reasonably clear and actionable
		Acceptable tone (no insults, harassment, or aggressive language)

	Decision: FAIL
		Use fail when ANY of the following is true:
		Clearly irrelevant to the user’s question
		Factually wrong / hallucinated in a way that would mislead
		Unsafe / disallowed content (if present, tag and fail)
		Rude / hostile / abusive tone
		Non-answer / evasive / refuses without reason

	Borderline Cases
		We want borderline cases because they stress calibration.
		Use tags like:
			borderline_relevance
			borderline_correctness
			and keep notes concise.

	Score Guidance (1..5)
		If you include human_scores, use 1..5:
			5: excellent
			4: good
			3: acceptable but weak
			2: poor
			1: unacceptable

		Minimum expectation:
			For pass, relevance/correctness should usually be >= 3
			For fail, at least one of relevance/correctness/tone is typically <= 2

	Case Design Targets (first 100)
		Aim for balance:
			40–50 PASS
			50–60 FAIL (catching failures is critical)
			Coverage mix:
				irrelevance (20)
				correctness/hallucination (20)
				rude tone (15)
				incomplete/insufficient (15)
				formatting/structure issues (10)
				multi-turn/dialogue context (10)
				borderline (10)

	Review Checklist (PR)
		Before merging:
			JSONL is valid (one object per line)
			case_id is unique
			rubric_id == chat_quality
			conversation roles are one of: system, user, assistant
			no empty content
			no secrets, personal data, or proprietary text
			balanced additions (don’t add only PASS)


	Running Locally
		Validate dataset:
			poetry run pytest -q (includes dataset validation tests)
		Run benchmark:
			poetry run python -m llm_judge.eval.run --spec configs/runspecs/pr_gate.yaml