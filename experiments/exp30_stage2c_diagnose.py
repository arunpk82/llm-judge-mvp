"""
Diagnostic: Trace the Stage 2c failure funnel.
Where are sentences dropping out of the graph verification pipeline?
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Load results
with open("experiments/exp30_stage2c_graph_precise_results.json") as f:
    data = json.load(f)

sentences = data["sentences"]

print("=" * 70)
print("DIAGNOSTIC: Stage 2c Failure Funnel")
print("=" * 70)

# 1. Parse stats
print("\n--- PARSE COVERAGE ---")
has_triples = [s for s in sentences if s["parsed_triples"] > 0]
has_entities = [s for s in sentences if s["parsed_entities"] > 0]
has_cardinals = [s for s in sentences if s["parsed_cardinals"] > 0]
no_checks = [s for s in sentences if s["checks"] == 0]

print(f"  Total sentences: {len(sentences)}")
print(
    f"  Sentences with SVO triples: {len(has_triples)} ({len(has_triples)/len(sentences)*100:.1f}%)"
)
print(
    f"  Sentences with NER entities: {len(has_entities)} ({len(has_entities)/len(sentences)*100:.1f}%)"
)
print(
    f"  Sentences with cardinals:    {len(has_cardinals)} ({len(has_cardinals)/len(sentences)*100:.1f}%)"
)
print(
    f"  Sentences with NO checks:    {len(no_checks)} ({len(no_checks)/len(sentences)*100:.1f}%)"
)

# 2. Verdict breakdown by check counts
print("\n--- VERDICT BY CHECK COUNT ---")
for bucket in ["0", "1-2", "3-5", "6+"]:
    if bucket == "0":
        subset = [s for s in sentences if s["checks"] == 0]
    elif bucket == "1-2":
        subset = [s for s in sentences if 1 <= s["checks"] <= 2]
    elif bucket == "3-5":
        subset = [s for s in sentences if 3 <= s["checks"] <= 5]
    else:
        subset = [s for s in sentences if s["checks"] >= 6]

    verdicts = Counter(s["l2b_verdict"] for s in subset)
    print(
        f"  {bucket} checks: {len(subset)} sentences → "
        f"G={verdicts.get('grounded',0)} F={verdicts.get('flagged',0)} U={verdicts.get('unknown',0)}"
    )

# 3. Evidence analysis — why are sentences "unknown"?
print("\n--- UNKNOWN SENTENCES: WHY? ---")
unknown = [s for s in sentences if s["l2b_verdict"] == "unknown"]

# Parse the evidence strings to find patterns
subject_not_found = 0
action_not_found = 0
no_actions_for_entity = 0
objects_not_confirmed = 0
no_evidence = 0
entity_only = 0

for s in unknown:
    ev = s.get("evidence", "")
    if not ev:
        no_evidence += 1
        continue

    has_subject_fail = "not found in graph" in ev and "Subject" in ev
    has_action_fail = "Action" in ev and "not found for" in ev
    has_no_actions = "No actions found for" in ev
    has_obj_fail = "not confirmed in graph" in ev
    has_entity_present = "found as" in ev

    if has_subject_fail:
        subject_not_found += 1
    elif has_no_actions:
        no_actions_for_entity += 1
    elif has_action_fail:
        action_not_found += 1
    elif has_obj_fail:
        objects_not_confirmed += 1
    elif has_entity_present and not has_action_fail:
        entity_only += 1
    else:
        no_evidence += 1

print(f"  Total unknown: {len(unknown)}")
print(
    f"  Subject not found in graph:     {subject_not_found} (pronoun/name resolution failure)"
)
print(
    f"  Entity found, no actions:        {no_actions_for_entity} (entity exists but no action edges)"
)
print(f"  Entity found, action not found:  {action_not_found} (verb mismatch)")
print(
    f"  Action found, objects not confirmed: {objects_not_confirmed} (Fix 3 blocked clearance)"
)
print(f"  Entity present only (no SVO):    {entity_only} (Fix 1+2 blocked clearance)")
print(f"  No evidence / other:             {no_evidence}")

# 4. Subject resolution failures — what subjects are failing?
print("\n--- SUBJECT RESOLUTION FAILURES (sample) ---")
subj_failures = []
for s in unknown:
    ev = s.get("evidence", "")
    if "Subject" in ev and "not found in graph" in ev:
        # Extract the subject name
        parts = ev.split("Subject '")
        for p in parts[1:]:
            subj = p.split("'")[0]
            subj_failures.append(subj)

subj_counts = Counter(subj_failures).most_common(15)
for subj, count in subj_counts:
    print(f"  '{subj}': {count} occurrences")

# 5. Action resolution failures — what verbs are failing?
print("\n--- ACTION RESOLUTION FAILURES (sample) ---")
action_failures = []
for s in unknown:
    ev = s.get("evidence", "")
    if "Action '" in ev and "not found for" in ev:
        parts = ev.split("Action '")
        for p in parts[1:]:
            verb = p.split("'")[0]
            entity = p.split("for '")[-1].split("'")[0] if "for '" in p else "?"
            action_failures.append(f"{verb} (for {entity})")

action_counts = Counter(action_failures).most_common(15)
for action, count in action_counts:
    print(f"  '{action}': {count}")

# 6. Grounded sentences — what DID work?
print("\n--- GROUNDED SENTENCES: WHAT WORKED ---")
grounded = [s for s in sentences if s["l2b_verdict"] == "grounded"]
for s in grounded[:10]:
    gt = "HALL" if s["gt_label"] == "hallucinated" else "clean"
    print(f"  [{gt}] {s['case_id']} S{s['sentence_idx']}")
    print(f"    Sentence: {s['sentence'][:80]}")
    print(f"    Evidence: {s['evidence'][:100]}")

# 7. Flagged sentences — what caught?
print("\n--- FLAGGED SENTENCES: WHAT CAUGHT ---")
flagged = [s for s in sentences if s["l2b_verdict"] == "flagged"]
for s in flagged[:10]:
    gt = "HALL" if s["gt_label"] == "hallucinated" else "clean"
    print(f"  [{gt}] {s['case_id']} S{s['sentence_idx']}")
    print(f"    Sentence: {s['sentence'][:80]}")
    print(f"    Evidence: {s['evidence'][:100]}")

# 8. The key question: how many unknown sentences HAVE triples but failed entity resolution?
print("\n--- THE GAP: Sentences with triples but unknown verdict ---")
has_triples_unknown = [s for s in unknown if s["parsed_triples"] > 0]
has_entities_unknown = [s for s in unknown if s["parsed_entities"] > 0]
print(
    f"  Unknown sentences with SVO triples: {len(has_triples_unknown)}/{len(unknown)}"
)
print(
    f"  Unknown sentences with NER entities: {len(has_entities_unknown)}/{len(unknown)}"
)
print(
    f"  Unknown sentences with BOTH: {len([s for s in unknown if s['parsed_triples'] > 0 and s['parsed_entities'] > 0])}/{len(unknown)}"
)
print(
    f"  Unknown sentences with NEITHER: {len([s for s in unknown if s['parsed_triples'] == 0 and s['parsed_entities'] == 0])}/{len(unknown)}"
)
