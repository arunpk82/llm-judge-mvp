import json

with open("experiments/exp29b_l1_rules_tightened_results.json") as f:
    l1 = json.load(f)
with open("experiments/exp31_stage2_ensemble_results.json") as f:
    l2 = json.load(f)

l1_map = {(s["case_id"], s["sentence_idx"]): s for s in l1["sentences"]}
l2_map = {(s["case_id"], s["sentence_idx"]): s for s in l2["sentences"]}

# L1 A1 cleared
l1_cleared = {k for k, v in l1_map.items() if v["a1_string_match"]}
# L1 B flags
l1_flagged = {k for k, v in l1_map.items() if v["any_b_flag"]}
# L1 hallucination catches
l1_catches = {
    k for k, v in l1_map.items() if v["any_b_flag"] and v["gt_label"] == "hallucinated"
}

# L2 ensemble
l2_cleared = {k for k, v in l2_map.items() if v["verdict"] == "grounded"}
l2_flagged = {k for k, v in l2_map.items() if v["verdict"] == "flagged"}
l2_catches = {
    k
    for k, v in l2_map.items()
    if v["verdict"] == "flagged" and v["gt_label"] == "hallucinated"
}

print("=" * 70)
print("L1 + L2 ENSEMBLE: COMBINED PIPELINE")
print("=" * 70)

# Clearance
both_cleared = l1_cleared & l2_cleared
l1_only_cleared = l1_cleared - l2_cleared
l2_only_cleared = l2_cleared - l1_cleared
combined_cleared = l1_cleared | l2_cleared

print("\n--- CLEARANCE ---")
print(f"  L1 A1:       {len(l1_cleared)}")
print(f"  L2 ensemble: {len(l2_cleared)}")
print(f"  Overlap:     {len(both_cleared)}")
print(f"  L1 only:     {len(l1_only_cleared)}")
print(f"  L2 only:     {len(l2_only_cleared)}")
print(f"  COMBINED:    {len(combined_cleared)}")

# Safety check on combined clearance
hall_in_combined = 0
for key in combined_cleared:
    s = l2_map.get(key)
    if s and s["gt_label"] == "hallucinated":
        hall_in_combined += 1
        print(f"  !! SAFETY: {s['case_id']} S{s['sentence_idx']}")
print(f"  Safety violations: {hall_in_combined}")

# Flags
both_flagged = l1_flagged & l2_flagged
combined_flagged = (l1_flagged | l2_flagged) - combined_cleared
print("\n--- FLAGS ---")
print(f"  L1 B flags:  {len(l1_flagged)}")
print(f"  L2 flags:    {len(l2_flagged)}")
print(f"  Overlap:     {len(both_flagged)}")
print(f"  COMBINED (excl cleared): {len(combined_flagged)}")

# Catches
both_catches = l1_catches & l2_catches
l1_only_catches = l1_catches - l2_catches
l2_only_catches = l2_catches - l1_catches
combined_catches = l1_catches | l2_catches

print("\n--- HALLUCINATION CATCHES ---")
print(f"  L1 catches:  {len(l1_catches)}/16")
print(f"  L2 catches:  {len(l2_catches)}/16")
print(f"  Both catch:  {len(both_catches)}")
print(f"  L1 only:     {len(l1_only_catches)}")
print(f"  L2 only:     {len(l2_only_catches)}")
print(
    f"  COMBINED:    {len(combined_catches)}/16 ({len(combined_catches)/16*100:.0f}%)"
)

for key in sorted(l1_only_catches):
    s = l2_map[key]
    print(f"    L1-only: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:70]}")
for key in sorted(l2_only_catches):
    s = l2_map[key]
    print(f"    L2-only: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:70]}")

# Remaining unknown
remaining = 283 - len(combined_cleared) - len(combined_flagged)
# How many hallucinations remain?
all_hall = {k for k, v in l2_map.items() if v["gt_label"] == "hallucinated"}
hall_caught = combined_catches
hall_cleared_wrong = {
    k for k in combined_cleared if l2_map[k]["gt_label"] == "hallucinated"
}
hall_remaining = all_hall - hall_caught - hall_cleared_wrong

print("\n--- COMBINED L1+L2 ENSEMBLE PIPELINE ---")
print(
    f"  Cleared (grounded):     {len(combined_cleared)} ({len(combined_cleared)/283*100:.1f}%)"
)
print(
    f"  Flagged:                {len(combined_flagged)} ({len(combined_flagged)/283*100:.1f}%)"
)
print(f"  Unknown → L3/L4:       {remaining} ({remaining/283*100:.1f}%)")
print(
    f"  Hallucinations caught:  {len(combined_catches)}/16 ({len(combined_catches)/16*100:.0f}%)"
)
print(f"  Hallucinations missed:  {len(hall_remaining)}/16")
print(f"  Safety violations:      {hall_in_combined}")

print("\n  Missed hallucinations for L3/L4:")
for key in sorted(hall_remaining):
    s = l2_map[key]
    print(f"    {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:80]}")
