"""
Debug: Why does ensemble clear fewer sentences than single-graph?
Find sentences L2c cleared but ensemble didn't.
"""

import json
from collections import Counter

with open("experiments/exp30_stage2c_graph_precise_results.json") as f:
    single = json.load(f)

with open("experiments/exp31_stage2_ensemble_results.json") as f:
    ensemble = json.load(f)

single_map = {(s["case_id"], s["sentence_idx"]): s for s in single["sentences"]}
ensemble_map = {(s["case_id"], s["sentence_idx"]): s for s in ensemble["sentences"]}

# Find sentences cleared in single but not in ensemble
single_cleared = {k for k, v in single_map.items() if v["l2b_verdict"] == "grounded"}
ensemble_cleared = {k for k, v in ensemble_map.items() if v["verdict"] == "grounded"}

both = single_cleared & ensemble_cleared
single_only = single_cleared - ensemble_cleared
ensemble_only = ensemble_cleared - single_cleared

print("=" * 70)
print("CLEARANCE COMPARISON: Single vs Ensemble")
print("=" * 70)
print(f"  Single cleared: {len(single_cleared)}")
print(f"  Ensemble cleared: {len(ensemble_cleared)}")
print(f"  Both cleared: {len(both)}")
print(f"  Single only (lost): {len(single_only)}")
print(f"  Ensemble only (new): {len(ensemble_only)}")

print("\n--- SENTENCES LOST (single cleared, ensemble didn't) ---")
lost_verdicts = Counter()
lost_reasons = Counter()

for key in sorted(single_only):
    s_single = single_map[key]
    s_ens = ensemble_map[key]
    gt = s_single["gt_label"]
    ens_verdict = s_ens["verdict"]
    ens_conf = s_ens["confidence"]
    per_graph = s_ens.get("per_graph", "{}")

    lost_verdicts[ens_verdict] += 1

    # Parse per-graph verdicts
    try:
        pg = json.loads(per_graph) if isinstance(per_graph, str) else per_graph
    except:
        pg = {}

    # Count graph verdicts
    pg_vals = list(pg.values())
    n_g = pg_vals.count("grounded")
    n_f = pg_vals.count("flagged")
    n_u = pg_vals.count("unknown")
    pattern = f"G{n_g}_F{n_f}_U{n_u}"
    lost_reasons[pattern] += 1

    print(
        f"  [{gt}] {s_single['case_id']} S{s_single['sentence_idx']} → ensemble={ens_verdict} ({ens_conf})"
    )
    print(f"    Per-graph: {pg}")
    print(f"    Sentence: {s_single['sentence'][:80]}")
    print(f"    Single evidence: {s_single.get('evidence','')[:80]}")
    print(f"    Ensemble evidence: {s_ens.get('evidence','')[:80]}")

print("\n--- LOST VERDICT DISTRIBUTION ---")
for v, c in lost_verdicts.most_common():
    print(f"  {v}: {c}")

print("\n--- LOST GRAPH PATTERN DISTRIBUTION ---")
for p, c in lost_reasons.most_common():
    print(f"  {p}: {c}  (G=grounded, F=flagged, U=unknown count)")

print("\n--- SENTENCES GAINED (ensemble cleared, single didn't) ---")
for key in sorted(ensemble_only):
    s_ens = ensemble_map[key]
    s_single = single_map[key]
    gt = s_single["gt_label"]
    print(
        f"  [{gt}] {s_single['case_id']} S{s_single['sentence_idx']} single={s_single['l2b_verdict']} → ensemble=grounded ({s_ens['confidence']})"
    )
    print(f"    {s_single['sentence'][:80]}")

# Confidence distribution of ensemble grounded
print("\n--- ENSEMBLE GROUNDED: CONFIDENCE BREAKDOWN ---")
for key in sorted(ensemble_cleared):
    s = ensemble_map[key]
    pg = (
        json.loads(s["per_graph"])
        if isinstance(s["per_graph"], str)
        else s["per_graph"]
    )
    pg_vals = list(pg.values())
    n_g = pg_vals.count("grounded")
    n_f = pg_vals.count("flagged")
    n_u = pg_vals.count("unknown")

conf_dist = Counter()
for key in ensemble_cleared:
    s = ensemble_map[key]
    conf_dist[s["confidence"]] += 1
print(f"  High: {conf_dist.get('high',0)}")
print(f"  Medium: {conf_dist.get('medium',0)}")
print(f"  Low: {conf_dist.get('low',0)}")

# What if we relax: 1 grounded + 0 flagged = grounded?
print("\n--- WHAT IF: Relax to 1 grounded + 0 flagged = grounded ---")
relaxed_grounded = 0
relaxed_safety = 0
for key, s in ensemble_map.items():
    pg = (
        json.loads(s["per_graph"])
        if isinstance(s["per_graph"], str)
        else s["per_graph"]
    )
    pg_vals = list(pg.values())
    n_g = pg_vals.count("grounded")
    n_f = pg_vals.count("flagged")
    if n_g >= 1 and n_f == 0:
        relaxed_grounded += 1
        if s["gt_label"] == "hallucinated":
            relaxed_safety += 1
            print(
                f"  !! Safety: {s['case_id']} S{s['sentence_idx']}: {s['sentence'][:70]}"
            )

print(f"  Relaxed grounded: {relaxed_grounded} (currently {len(ensemble_cleared)})")
print(f"  Relaxed safety violations: {relaxed_safety}")
