"""
Compare L1 A1 vs L2c: which sentences overlap?
"""

import json

# Load L1 results (Exp 29b)
with open("experiments/exp29b_l1_rules_tightened_results.json") as f:
    l1_data = json.load(f)

# Load L2c results
with open("experiments/exp30_stage2c_graph_precise_results.json") as f:
    l2_data = json.load(f)

# L1 A1: sentences where a1_string_match = True
l1_cleared = set()
for s in l1_data["sentences"]:
    if s["a1_string_match"]:
        l1_cleared.add((s["case_id"], s["sentence_idx"]))

# L2c: sentences where verdict = grounded
l2_cleared = set()
l2_flagged = set()
for s in l2_data["sentences"]:
    if s["l2b_verdict"] == "grounded":
        l2_cleared.add((s["case_id"], s["sentence_idx"]))
    elif s["l2b_verdict"] == "flagged":
        l2_flagged.add((s["case_id"], s["sentence_idx"]))

# L1 B flags: sentences flagged by B checks
l1_flagged = set()
for s in l1_data["sentences"]:
    if s["any_b_flag"]:
        l1_flagged.add((s["case_id"], s["sentence_idx"]))

print("=" * 70)
print("L1 A1 vs L2c: OVERLAP ANALYSIS")
print("=" * 70)

# Clearance overlap
both_cleared = l1_cleared & l2_cleared
l1_only = l1_cleared - l2_cleared
l2_only = l2_cleared - l1_cleared

print("\n--- CLEARANCE (GROUNDED) ---")
print(f"  L1 A1 cleared: {len(l1_cleared)}")
print(f"  L2c cleared:   {len(l2_cleared)}")
print(f"  Both cleared:  {len(both_cleared)}  (overlap)")
print(f"  L1 only:       {len(l1_only)}  (L1 clears but L2 doesn't)")
print(f"  L2 only:       {len(l2_only)}  (L2 clears but L1 doesn't)")
print(f"  Combined:      {len(l1_cleared | l2_cleared)}  (union)")

# Flag overlap
both_flagged = l1_flagged & l2_flagged
l1_flag_only = l1_flagged - l2_flagged
l2_flag_only = l2_flagged - l1_flagged

print("\n--- FLAGS ---")
print(f"  L1 B flags:    {len(l1_flagged)}")
print(f"  L2c flags:     {len(l2_flagged)}")
print(f"  Both flagged:  {len(both_flagged)}")
print(f"  L1 only:       {len(l1_flag_only)}")
print(f"  L2 only:       {len(l2_flag_only)}")
print(f"  Combined:      {len(l1_flagged | l2_flagged)}")

# What does L2 add to L1?
print("\n--- L2's UNIQUE CONTRIBUTION ---")
print(f"  New clearances (L2 only): {len(l2_only)}")
print(f"  New flags (L2 only):      {len(l2_flag_only)}")

# Check GT labels for the unique ones
l2_sentences = {(s["case_id"], s["sentence_idx"]): s for s in l2_data["sentences"]}
l1_sentences = {(s["case_id"], s["sentence_idx"]): s for s in l1_data["sentences"]}

print("\n  L2-only cleared sentences:")
for key in sorted(l2_only):
    s = l2_sentences[key]
    gt = "HALL" if s["gt_label"] == "hallucinated" else "clean"
    print(f"    [{gt}] {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:80]}")

print("\n  L1-only cleared (L2 missed these):")
for key in sorted(l1_only):
    s = l2_sentences.get(key)
    if s:
        gt = "HALL" if s["gt_label"] == "hallucinated" else "clean"
        l2_verdict = s["l2b_verdict"]
        print(
            f"    [{gt}] {s['case_id']} S{s['sentence_idx']} (L2={l2_verdict}): {s['sentence'][:70]}"
        )

# Hallucination catches overlap
l1_hall_catches = set()
for s in l1_data["sentences"]:
    if s["any_b_flag"] and s["gt_label"] == "hallucinated":
        l1_hall_catches.add((s["case_id"], s["sentence_idx"]))

l2_hall_catches = set()
for s in l2_data["sentences"]:
    if s["l2b_verdict"] == "flagged" and s["gt_label"] == "hallucinated":
        l2_hall_catches.add((s["case_id"], s["sentence_idx"]))

print("\n--- HALLUCINATION CATCH OVERLAP ---")
print(f"  L1 catches: {len(l1_hall_catches)}")
print(f"  L2 catches: {len(l2_hall_catches)}")
print(f"  Both catch: {len(l1_hall_catches & l2_hall_catches)}")
print(f"  L1 only:    {len(l1_hall_catches - l2_hall_catches)}")
print(f"  L2 only:    {len(l2_hall_catches - l1_hall_catches)}")
print(f"  Combined:   {len(l1_hall_catches | l2_hall_catches)}")

for key in sorted(l1_hall_catches - l2_hall_catches):
    s = l2_sentences.get(key)
    if s:
        print(
            f"    L1-only catch: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:80]}"
        )

for key in sorted(l2_hall_catches - l1_hall_catches):
    s = l2_sentences[key]
    print(
        f"    L2-only catch: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:80]}"
    )

# Combined pipeline
print("\n--- COMBINED L1+L2 PIPELINE ---")
combined_cleared = l1_cleared | l2_cleared
combined_flagged = (l1_flagged | l2_flagged) - combined_cleared
combined_catches = l1_hall_catches | l2_hall_catches

# Safety: any hallucination in combined_cleared?
hall_in_cleared = 0
for key in combined_cleared:
    s = l2_sentences.get(key)
    if s and s["gt_label"] == "hallucinated":
        hall_in_cleared += 1
        print(
            f"  !! Safety violation: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:70]}"
        )

print(f"\n  Combined cleared: {len(combined_cleared)} sentences")
print(f"  Combined flagged: {len(combined_flagged)} sentences")
print(f"  Combined catches: {len(combined_catches)}/16 hallucinations")
print(f"  Safety violations: {hall_in_cleared}")
print(f"  Remaining unknown: {283 - len(combined_cleared) - len(combined_flagged)}")
