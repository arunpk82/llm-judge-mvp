"""
Debug: Investigate why graph builder drops entity-action connections.

Loads fact tables, builds graphs, and compares:
- What entities exist in the JSON
- What events exist in the JSON
- What nodes/edges exist in the graph
- Which entities are missing action edges
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import the graph builder from Stage 2c
sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp30_stage2c_graph_precise import _resolve_entity, json_to_graph

with open("experiments/exp30_fact_tables.json") as f:
    fact_tables = json.load(f)

# =====================================================================
# Test on multiple cases to find patterns
# =====================================================================

# Pick cases with known action failures from diagnostic
test_cases = [
    "ragtruth_180",  # Copeland — "stab for chris copeland" failed
    "ragtruth_138",  # West — "no actions found for west"
    "ragtruth_28",  # Thomas — "charge for unnamed three women"
    "ragtruth_42",  # Blue Bell — "no actions found for blue bell ice cream"
    "ragtruth_68",  # Warren/Paul — "announce for rand paul" failed
]

for case_id in test_cases:
    ft_data = fact_tables.get(case_id, {})
    ft = ft_data.get("fact_table")
    if not ft:
        print(f"\n{case_id}: no fact table")
        continue

    print(f"\n{'='*70}")
    print(f"CASE: {case_id}")
    print(f"{'='*70}")

    # --- What the JSON has ---
    entities = ft.get("entities", [])
    events = ft.get("events", [])
    relationships = ft.get("relationships", [])

    print(
        f"\n  JSON: {len(entities)} entities, {len(events)} events, {len(relationships)} relationships"
    )

    print("\n  --- JSON ENTITIES (first 8) ---")
    for e in entities[:8]:
        aliases = e.get("aliases", []) or []
        print(
            f"    {e.get('name', '?'):30s} type={e.get('type', '?'):8s} aliases={aliases}"
        )

    print("\n  --- JSON EVENTS (first 8) ---")
    for ev in events[:8]:
        who = ev.get("who", "?")
        action = ev.get("action", "?")
        target = ev.get("target", "?")
        print(
            f"    who={str(who):25s} action={str(action):15s} target={str(target)[:40]}"
        )

    # --- Build graph ---
    G = json_to_graph(ft)
    print(f"\n  GRAPH: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Entity nodes ---
    entity_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "entity"
    ]
    action_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "action"
    ]
    value_nodes = [
        (n, d) for n, d in G.nodes(data=True) if d.get("node_type") == "value"
    ]

    print(
        f"\n  Node types: {len(entity_nodes)} entity, {len(action_nodes)} action, {len(value_nodes)} value"
    )

    # --- For each entity node, count action edges ---
    print("\n  --- ENTITY → ACTION CONNECTIONS ---")
    entities_with_actions = 0
    entities_without_actions = 0

    for node, ndata in entity_nodes[:15]:
        # Find action edges from this entity
        action_edges = [
            (t, d)
            for _, t, d in G.edges(node, data=True)
            if d.get("relation") == "action"
        ]
        all_edges = list(G.edges(node, data=True))

        if action_edges:
            entities_with_actions += 1
            verbs = [d.get("verb", "?") for _, d in action_edges]
            print(f"    ✓ {node:30s} → {len(action_edges)} actions: {verbs[:5]}")
        else:
            entities_without_actions += 1
            edge_types = [d.get("relation", "?") for _, _, d in all_edges]
            print(
                f"    ✗ {node:30s} → 0 actions (has {len(all_edges)} edges: {edge_types[:5]})"
            )

    print(
        f"\n  Summary: {entities_with_actions} with actions, {entities_without_actions} without"
    )

    # --- Check: why are events not connecting? ---
    print("\n  --- EVENT CONNECTION DEBUG ---")
    for ev in events[:5]:
        who = ev.get("who", "")
        action = ev.get("action", "").lower().strip()

        # What the graph builder does
        actors = [who] if isinstance(who, str) else (who or [])
        for actor in actors:
            actor_lower = actor.lower().strip()
            action_node_name = f"evt_{actor_lower}_{action}"

            # Check if actor exists in graph
            actor_in_graph = actor_lower in G
            action_node_in_graph = action_node_name in G

            # Check if edge exists
            edge_exists = (
                G.has_edge(actor_lower, action_node_name)
                if actor_in_graph and action_node_in_graph
                else False
            )

            status = "✓" if edge_exists else "✗"
            print(f"    {status} '{actor_lower}' → '{action_node_name}'")
            if not actor_in_graph:
                print(f"      Problem: actor '{actor_lower}' NOT in graph as node")
                # Check if it exists under different key
                resolved = _resolve_entity(G, actor_lower)
                if resolved:
                    print(f"      But resolves to: '{resolved}'")
                    # Check if THAT node has the action
                    alt_action = f"evt_{resolved}_{action}"
                    if G.has_edge(resolved, alt_action):
                        print(f"      AND '{resolved}' → '{alt_action}' EXISTS")
                    else:
                        print(f"      BUT '{resolved}' has no '{action}' action edge")
            elif not action_node_in_graph:
                print(f"      Problem: action node '{action_node_name}' NOT in graph")
            elif not edge_exists:
                print("      Problem: nodes exist but no edge between them")

    # --- Check specific failures from diagnostic ---
    print("\n  --- SPECIFIC FAILURE TESTS ---")
    test_lookups = {
        "ragtruth_180": [("chris copeland", "stab"), ("copeland", "stab")],
        "ragtruth_138": [
            ("west", "plead"),
            ("kanye west", "plead"),
            ("kanye west", "settle"),
        ],
        "ragtruth_28": [
            ("keonna thomas", "charge"),
            ("thomas", "charge"),
            ("three women", "charge"),
        ],
        "ragtruth_42": [("blue bell", "recall"), ("blue bell ice cream", "recall")],
        "ragtruth_68": [("rand paul", "announce"), ("elizabeth warren", "respond")],
    }

    for entity, verb in test_lookups.get(case_id, []):
        resolved = _resolve_entity(G, entity)
        if resolved:
            # Find all edges from resolved entity
            edges = list(G.edges(resolved, data=True))
            action_edges = [d for _, _, d in edges if d.get("relation") == "action"]
            verbs = [d.get("verb") for d in action_edges]
            print(
                f"    '{entity}' → resolves to '{resolved}' → {len(action_edges)} actions: {verbs[:5]}"
            )
        else:
            print(f"    '{entity}' → NOT FOUND in graph")

print(f"\n{'='*70}")
print("AGGREGATE ACROSS ALL 50 CASES")
print(f"{'='*70}")

total_entities = 0
total_with_actions = 0
total_without_actions = 0
total_events_in_json = 0
total_action_nodes = 0

for case_id, ft_data in fact_tables.items():
    ft = ft_data.get("fact_table")
    if not ft:
        continue
    G = json_to_graph(ft)

    entities_in_json = len(ft.get("entities", []))
    events_in_json = len(ft.get("events", []))
    total_events_in_json += events_in_json

    entity_nodes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "entity"]
    action_nodes_g = [
        n for n, d in G.nodes(data=True) if d.get("node_type") == "action"
    ]
    total_action_nodes += len(action_nodes_g)

    for node in entity_nodes:
        total_entities += 1
        action_edges = [
            1 for _, _, d in G.edges(node, data=True) if d.get("relation") == "action"
        ]
        if action_edges:
            total_with_actions += 1
        else:
            total_without_actions += 1

print(f"  Total entity nodes: {total_entities}")
print(
    f"  With action edges: {total_with_actions} ({total_with_actions/total_entities*100:.1f}%)"
)
print(
    f"  Without action edges: {total_without_actions} ({total_without_actions/total_entities*100:.1f}%)"
)
print(f"  Total events in JSON: {total_events_in_json}")
print(f"  Total action nodes in graph: {total_action_nodes}")
print(
    f"  Connection rate: {total_action_nodes/total_events_in_json*100:.1f}% of JSON events became graph edges"
)
