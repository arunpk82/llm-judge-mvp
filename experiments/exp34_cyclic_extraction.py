"""
Experiment 34 — Five-Graph Sentence-Anchored Knowledge Extraction

Architecture:
  5 separate graphs, same unified schema, different emphasis:
    G1: Entity-driven     (who/what exists)
    G2: Event-driven      (what happened)
    G3: Relationship-driven (how things connect)
    G4: Number-driven     (quantities, dates, measurements)
    G5: Negation-driven   (what's denied, absent, bounded)

  Cross-reference indexes:
    Entity Index:   entity → {which graphs, which sentences}
    Sentence Index: sentence → {which graphs have coverage}

  Gap Detection: matrix of (entity × graph) and (sentence × graph)
  Targeted Fill: sharp questions for specific empty cells
  Enrichment: proximity-based cross-graph linking

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python experiments/exp34_cyclic_extraction.py --max-sources 2
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ========================================================================
# API CALLER
# ========================================================================

def call_gemma(prompt: str, model_id: str = "gemma-4-31b-it",
               timeout: int = 600) -> tuple[dict | None, float]:
    import httpx
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, 0.0

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model_id}:generateContent?key={api_key}")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0,
                              "responseMimeType": "application/json"},
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=float(timeout)) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                raw = raw.strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            return json.loads(raw), time.time() - t0
    except json.JSONDecodeError as e:
        print(f"        JSON error: {str(e)[:60]}")
        return None, time.time() - t0
    except Exception as e:
        print(f"        API error: {str(e)[:80]}")
        return None, time.time() - t0


# ========================================================================
# PRE-PROCESS
# ========================================================================

def preprocess_source(source_text: str) -> dict:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text) if s.strip()]
        numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))
        return {"sentences": sentences, "n_sentences": len(sentences),
                "numbered_source": numbered, "spacy_entities": []}

    doc = nlp(source_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences, 1))

    spacy_entities = []
    seen_ents = set()
    ENTITY_LABELS = {"PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "EVENT", "PRODUCT"}
    for ent in doc.ents:
        if ent.label_ not in ENTITY_LABELS:
            continue
        if ent.text.lower() in seen_ents:
            continue
        seen_ents.add(ent.text.lower())
        for i, sent in enumerate(sentences, 1):
            if ent.text in sent:
                spacy_entities.append({"text": ent.text, "label": ent.label_, "sentence_id": i})
                break

    return {"sentences": sentences, "n_sentences": len(sentences),
            "numbered_source": numbered, "spacy_entities": spacy_entities}


SCHEMA = """
Return ONLY valid JSON: {"items": [{"sentence_id": N, "subject": "...", "predicate": "...", "object": "...", "qualifier": {}, "source_text": "..."}]}
"""


# ========================================================================
# EXTRACTION PROMPTS — 5 passes, unified schema, different emphasis
# ========================================================================

def extract_g1_entities(numbered_source: str) -> tuple[dict | None, float]:
    """G1: Entity-driven — who/what exists."""
    prompt = f"""You are an entity extraction engine. Extract EVERY person, organization, place, and thing.

For each entity:
- sentence_id: [N] where it FIRST appears
- subject: entity name (exact wording)
- predicate: "is_a"
- object: PERSON / ORG / PLACE / PRODUCT / OTHER
- qualifier: stated attributes (age, role, title, description)
- source_text: exact phrase

CRITICAL: Extract EVERY entity, even minor ones ("female companion", "the nightclub").

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


def extract_g2_events(numbered_source: str, entity_names: list[str]) -> tuple[dict | None, float]:
    """G2: Event-driven — what happened."""
    entity_list = ", ".join(entity_names)
    prompt = f"""You are an event extraction engine. Extract EVERY action or event.

Known entities: [{entity_list}]

For each event:
- sentence_id: [N] where it occurs
- subject: who performed the action
- predicate: the verb ("charged", "stabbed", "arrested", "traveled")
- object: who/what was acted upon
- qualifier: when, where, result, manner
- source_text: exact phrase

CRITICAL: Go sentence by sentence. Only VERBS — someone DID something.
NOT states ("is 30"). NOT relationships ("roommate of").

{SCHEMA}

Source:
{numbered_source}"""
    data, elapsed = call_gemma(prompt)
    if data is None:
        trunc = "\n".join(numbered_source.split("\n")[:30])
        data, e2 = call_gemma(prompt.replace(numbered_source, trunc))
        elapsed += e2
    return data, elapsed


def extract_g2_micro(numbered_source: str, entity_name: str) -> tuple[dict | None, float]:
    """G2 micro: what did this specific entity do?"""
    prompt = f"""What did "{entity_name}" do? List EVERY action they performed or that happened to them.

For each action:
- sentence_id: which [N]
- subject: "{entity_name}"
- predicate: verb/action
- object: target
- qualifier: when, where, result
- source_text: exact phrase

If no actions, return {{"items": []}}

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


def extract_g3_relationships(numbered_source: str, entity_names: list[str]) -> tuple[dict | None, float]:
    """G3: Relationship-driven — how entities connect."""
    entity_list = ", ".join(entity_names)
    prompt = f"""You are a relationship extraction engine. Map ALL connections between entities.

Known entities: [{entity_list}]

For each relationship:
- sentence_id: [N] where stated or implied
- subject: first entity
- predicate: relationship type (teammate, companion, employer, suspect, alias, member_of)
- object: second entity
- qualifier: conditions, context
- source_text: exact phrase

Only STRUCTURAL links between two named entities.
NOT ages, NOT actions.

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


def extract_g4_numbers(numbered_source: str) -> tuple[dict | None, float]:
    """G4: Number-driven — quantities, dates, measurements."""
    prompt = f"""You are a numerical extraction engine. Extract EVERY number, date, quantity, measurement.

For each:
- sentence_id: [N] where it appears
- subject: who/what the number describes
- predicate: "has_value" or "has_date"
- object: exact number or date
- qualifier: unit, what it measures
- source_text: exact phrase

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


def extract_g5_negations(numbered_source: str) -> tuple[dict | None, float]:
    """G5: Negation-driven — denied, absent, bounded."""
    prompt = f"""You are a negation extraction engine. Find everything denied, negated, absent, or bounded.

For each:
- sentence_id: [N] where it appears
- subject: who/what is negated
- predicate: "not" / "denied" / "absent" / "bounded_by" / "corrected_from"
- object: what is negated or the boundary
- qualifier: context
- source_text: exact phrase

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


# ========================================================================
# TARGETED FILL PROMPTS
# ========================================================================

def fill_sentence(numbered_source: str, sid: int) -> tuple[dict | None, float]:
    """Fill an uncovered sentence with ALL fact types."""
    prompt = f"""Extract ALL facts from sentence [{sid}] only.

For each fact:
- sentence_id: {sid}
- subject: main entity
- predicate: what is stated
- object: target or value
- qualifier: context
- source_text: exact phrase

If no facts, return {{"items": []}}

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


def fill_entity_in_graph(numbered_source: str, entity_name: str,
                          graph_name: str, sents: list[int]) -> tuple[dict | None, float]:
    """Fill a specific entity's gap in a specific graph."""
    sent_str = ", ".join(str(s) for s in sents[:5])

    focus = {
        "events": f'What ACTIONS did "{entity_name}" perform or undergo? Only verbs.',
        "relationships": f'How is "{entity_name}" CONNECTED to other entities? Only structural links.',
        "numbers": f'What NUMBERS, dates, or quantities are associated with "{entity_name}"?',
    }
    question = focus.get(graph_name, f'What facts about "{entity_name}" are in the document?')

    prompt = f"""{question}

"{entity_name}" appears in sentence(s) [{sent_str}]. Focus on those.

For each fact:
- sentence_id: which [N]
- subject: "{entity_name}"
- predicate: what is stated
- object: target or value
- qualifier: context
- source_text: exact phrase

If nothing found, return {{"items": []}}

{SCHEMA}

Source:
{numbered_source}"""
    return call_gemma(prompt)


# ========================================================================
# GRAPH OPERATIONS
# ========================================================================

def normalize_item(item) -> dict | None:
    if not isinstance(item, dict):
        return None
    subj = str(item.get("subject", "")).strip()
    if not subj:
        return None
    return {
        "sentence_id": item.get("sentence_id", 0),
        "subject": subj,
        "predicate": str(item.get("predicate", "")).strip(),
        "object": str(item.get("object", "")).strip(),
        "qualifier": item.get("qualifier") if isinstance(item.get("qualifier"), dict) else {},
        "source_text": str(item.get("source_text", "")).strip(),
    }


def dedup_within_graph(items: list) -> list[dict]:
    """Deduplicate within a single graph."""
    seen = set()
    result = []
    for raw in items:
        item = normalize_item(raw)
        if item is None:
            continue
        key = f"{item['subject'].lower()}|{item['predicate'].lower()}|{item['object'].lower()}"
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def add_to_graph(graph: list[dict], new_items: list, label: str = "") -> int:
    """Add items to a graph, dedup, return count of genuinely new items."""
    before = len(graph)
    existing_keys = set(
        f"{it['subject'].lower()}|{it['predicate'].lower()}|{it['object'].lower()}"
        for it in graph
    )
    added = 0
    for raw in new_items:
        item = normalize_item(raw)
        if item is None:
            continue
        key = f"{item['subject'].lower()}|{item['predicate'].lower()}|{item['object'].lower()}"
        if key not in existing_keys:
            existing_keys.add(key)
            graph.append(item)
            added += 1
    return added


# ========================================================================
# CROSS-REFERENCE INDEXES
# ========================================================================

GRAPH_NAMES = ["entities", "events", "relationships", "numbers", "negations"]


def build_entity_index(graphs: dict[str, list[dict]]) -> dict[str, dict]:
    """Map entity → {which graphs, which sentences, coverage}."""
    index = {}
    for gname in GRAPH_NAMES:
        for item in graphs[gname]:
            for field in ["subject", "object"]:
                name = item[field]
                if not name or name in ("", "None", "?", "-"):
                    continue
                nl = name.lower()
                if nl not in index:
                    index[nl] = {"name": name, "sentences": set(),
                                 "in_graphs": set()}
                index[nl]["in_graphs"].add(gname)
                if item["sentence_id"]:
                    index[nl]["sentences"].add(item["sentence_id"])

    # Serialize sets
    for entry in index.values():
        entry["sentences"] = sorted(entry["sentences"])
        entry["in_graphs"] = sorted(entry["in_graphs"])

    return index


def build_sentence_index(graphs: dict[str, list[dict]], n_sentences: int) -> dict[int, dict]:
    """Map sentence_id → {which graphs have coverage}."""
    index = {}
    for sid in range(1, n_sentences + 1):
        index[sid] = {gname: 0 for gname in GRAPH_NAMES}

    for gname in GRAPH_NAMES:
        for item in graphs[gname]:
            sid = item.get("sentence_id", 0)
            if sid in index:
                index[sid][gname] += 1

    return index


# ========================================================================
# GAP DETECTION — Matrix analysis across 5 graphs
# ========================================================================

def detect_gaps(graphs: dict, entity_idx: dict, sentence_idx: dict,
                sentences: list[str]) -> dict:
    """Find gaps by analyzing the entity×graph and sentence×graph matrices."""

    # Gap 1: Completely uncovered sentences (no items in ANY graph)
    uncovered_sentences = []
    for sid, coverage in sentence_idx.items():
        if sum(coverage.values()) == 0:
            uncovered_sentences.append(sid)

    # Gap 2: Entities in G1 but missing from G2 (entity exists but no events)
    entities_no_events = []
    for nl, entry in entity_idx.items():
        if "entities" in entry["in_graphs"] and "events" not in entry["in_graphs"]:
            entities_no_events.append({
                "name": entry["name"],
                "sentences": entry["sentences"],
                "has": entry["in_graphs"],
            })

    # Gap 3: Entities in G2 but missing from G1 (event actor not registered as entity)
    actors_not_entities = []
    for nl, entry in entity_idx.items():
        if "events" in entry["in_graphs"] and "entities" not in entry["in_graphs"]:
            actors_not_entities.append({
                "name": entry["name"],
                "sentences": entry["sentences"],
            })

    # Gap 4: Entities in G1 but missing from G3 (entity exists but no relationships)
    entities_no_rels = []
    for nl, entry in entity_idx.items():
        if "entities" in entry["in_graphs"] and "relationships" not in entry["in_graphs"]:
            entities_no_rels.append({
                "name": entry["name"],
                "sentences": entry["sentences"],
            })

    # Gap 5: Sentences with entities but no events
    sents_entity_no_event = []
    for sid, coverage in sentence_idx.items():
        if coverage["entities"] > 0 and coverage["events"] == 0:
            sents_entity_no_event.append(sid)

    # Gap 6: Adjacent sentence pairs sharing entities but no relationship
    existing_rels = set()
    for item in graphs["relationships"]:
        pair = tuple(sorted([item["subject"].lower(), item["object"].lower()]))
        existing_rels.add(pair)

    proximity_gaps = []
    for sid in range(1, len(sentences)):
        next_sid = sid + 1
        # Get entities at each sentence
        ents_here = set()
        ents_next = set()
        for gname in GRAPH_NAMES:
            for item in graphs[gname]:
                if item["sentence_id"] == sid:
                    ents_here.add(item["subject"].lower())
                    if item["object"]:
                        ents_here.add(item["object"].lower())
                elif item["sentence_id"] == next_sid:
                    ents_next.add(item["subject"].lower())
                    if item["object"]:
                        ents_next.add(item["object"].lower())
        # Shared entities across adjacent sentences
        shared = ents_here & ents_next
        if len(shared) >= 2:
            for e1 in ents_here:
                for e2 in ents_next:
                    if e1 != e2:
                        pair = tuple(sorted([e1, e2]))
                        if pair not in existing_rels:
                            proximity_gaps.append({
                                "entity1": e1, "entity2": e2,
                                "sentences": [sid, next_sid],
                                "proximity": "adjacent",
                            })
                            existing_rels.add(pair)

    return {
        "uncovered_sentences": uncovered_sentences,
        "entities_no_events": entities_no_events[:15],
        "actors_not_entities": actors_not_entities[:15],
        "entities_no_rels": entities_no_rels[:15],
        "sents_entity_no_event": sents_entity_no_event,
        "proximity_gaps": proximity_gaps[:10],
    }


# ========================================================================
# LOGGING
# ========================================================================

def log_samples(label: str, items: list, max_show: int = 3):
    if not items:
        return
    for i, item in enumerate(items[:max_show]):
        if isinstance(item, dict):
            sid = item.get("sentence_id", "?")
            subj = str(item.get("subject", "?"))[:20]
            pred = str(item.get("predicate", "?"))[:20]
            obj = str(item.get("object", "?"))[:30]
            print(f"        [{label} {i+1}] S{sid} {subj} | {pred} | {obj}")
        else:
            print(f"        [{label} {i+1}] {str(item)[:100]}")
    if len(items) > max_show:
        print(f"        ... and {len(items) - max_show} more")


def print_graph_summary(graphs: dict):
    """Print a summary table of all 5 graphs."""
    for gname in GRAPH_NAMES:
        items = graphs[gname]
        sids = set(it["sentence_id"] for it in items if it.get("sentence_id"))
        subjects = set(it["subject"].lower() for it in items if it["subject"])
        print(f"      G-{gname:<15s}: {len(items):>3d} items, "
              f"{len(sids):>2d} sentences, {len(subjects):>2d} unique subjects")


def print_entity_matrix(entity_idx: dict, max_show: int = 10):
    """Print entity × graph matrix."""
    print(f"      {'Entity':<25s} {'Sents':>5s}  Ent  Evt  Rel  Num  Neg")
    print(f"      {'-'*65}")
    for nl, entry in sorted(entity_idx.items(),
                              key=lambda x: len(x[1]["in_graphs"]), reverse=True)[:max_show]:
        name = entry["name"][:24]
        sents = ",".join(str(s) for s in entry["sentences"][:4])
        flags = ""
        for g in GRAPH_NAMES:
            flags += "  ✓  " if g in entry["in_graphs"] else "  ·  "
        print(f"      {name:<25s} {sents:>5s}{flags}")


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_source(source: str, source_idx: int, pre: dict,
               top_k_entities: int = 8, fill_max: int = 8) -> dict:

    numbered_source = pre["numbered_source"]
    sentences = pre["sentences"]
    all_times = {}

    # Initialize 5 graphs
    graphs = {gname: [] for gname in GRAPH_NAMES}

    # ==== STEP 2: EXTRACT ====
    print(f"    --- Step 2: Extract into 5 graphs ---")

    # G1: Entities
    data, t = extract_g1_entities(numbered_source)
    all_times["G1"] = round(t, 1)
    raw_items = (data.get("items") or []) if data else []
    graphs["entities"] = dedup_within_graph(raw_items)
    print(f"      G1 entities: {len(graphs['entities'])} ({t:.0f}s)")
    log_samples("G1", graphs["entities"])

    # Entity list for chaining (G1 + spaCy)
    entity_names = list(set(it["subject"] for it in graphs["entities"] if it["subject"]))
    for se in pre.get("spacy_entities", []):
        if se["text"] not in entity_names:
            entity_names.append(se["text"])
    n_g1 = len(set(it["subject"] for it in graphs["entities"] if it["subject"]))
    print(f"      Chaining list: {len(entity_names)} (G1:{n_g1} + spaCy:{len(entity_names)-n_g1})")

    # G2: Events (chained)
    data, t = extract_g2_events(numbered_source, entity_names)
    all_times["G2"] = round(t, 1)
    raw_items = (data.get("items") or []) if data else []
    graphs["events"] = dedup_within_graph(raw_items)
    print(f"      G2 events: {len(graphs['events'])} ({t:.0f}s)")
    log_samples("G2", graphs["events"])

    # G2 micro-extraction
    print(f"      [G2 micro-extraction × {top_k_entities} entities]")
    for ent_name in entity_names[:top_k_entities]:
        if not ent_name:
            continue
        data, t = extract_g2_micro(numbered_source, ent_name)
        if data:
            items = data.get("items") or []
            added = add_to_graph(graphs["events"], items)
            print(f"        {ent_name}: +{added} new (of {len(items)}) ({t:.0f}s)")
    print(f"      G2 after micro: {len(graphs['events'])}")

    # G3: Relationships (chained)
    data, t = extract_g3_relationships(numbered_source, entity_names)
    all_times["G3"] = round(t, 1)
    raw_items = (data.get("items") or []) if data else []
    graphs["relationships"] = dedup_within_graph(raw_items)
    print(f"      G3 relationships: {len(graphs['relationships'])} ({t:.0f}s)")
    log_samples("G3", graphs["relationships"])

    # G4: Numbers
    data, t = extract_g4_numbers(numbered_source)
    all_times["G4"] = round(t, 1)
    raw_items = (data.get("items") or []) if data else []
    graphs["numbers"] = dedup_within_graph(raw_items)
    print(f"      G4 numbers: {len(graphs['numbers'])} ({t:.0f}s)")
    log_samples("G4", graphs["numbers"])

    # G5: Negations
    data, t = extract_g5_negations(numbered_source)
    all_times["G5"] = round(t, 1)
    raw_items = (data.get("items") or []) if data else []
    graphs["negations"] = dedup_within_graph(raw_items)
    print(f"      G5 negations: {len(graphs['negations'])} ({t:.0f}s)")
    log_samples("G5", graphs["negations"])

    # ==== STEP 3: CROSS-REFERENCE INDEXES ====
    print(f"\n    --- Step 3: Cross-reference indexes ---")
    entity_idx = build_entity_index(graphs)
    sentence_idx = build_sentence_index(graphs, len(sentences))

    print_graph_summary(graphs)
    total_items = sum(len(g) for g in graphs.values())
    print(f"      Total across 5 graphs: {total_items}")
    print(f"      Unique entities in index: {len(entity_idx)}")
    print(f"\n      Entity × Graph matrix (top 10):")
    print_entity_matrix(entity_idx)

    # ==== STEP 4: GAP DETECTION ====
    print(f"\n    --- Step 4: Gap detection ---")
    gaps = detect_gaps(graphs, entity_idx, sentence_idx, sentences)

    print(f"      Uncovered sentences: {len(gaps['uncovered_sentences'])}/{len(sentences)}")
    for sid in gaps["uncovered_sentences"][:5]:
        print(f"        [{sid}] {sentences[sid-1][:80]}")
    if len(gaps["uncovered_sentences"]) > 5:
        print(f"        ... and {len(gaps['uncovered_sentences'])-5} more")

    print(f"      Entities in G1 but NOT in G2 (no events): {len(gaps['entities_no_events'])}")
    for gap in gaps["entities_no_events"][:5]:
        print(f"        {gap['name']} (sentences: {gap['sentences']}, has: {gap['has']})")

    print(f"      Actors in G2 but NOT in G1 (unregistered): {len(gaps['actors_not_entities'])}")
    for gap in gaps["actors_not_entities"][:3]:
        print(f"        {gap['name']}")

    print(f"      Entities in G1 but NOT in G3 (no rels): {len(gaps['entities_no_rels'])}")
    print(f"      Sentences with entities but no events: {len(gaps['sents_entity_no_event'])}")
    print(f"      Proximity gaps (adjacent, no rel): {len(gaps['proximity_gaps'])}")

    # ==== STEP 5: TARGETED FILL ====
    print(f"\n    --- Step 5: Targeted fill ---")

    # 5a: Fill uncovered sentences → items go to appropriate graphs
    n_fill = min(len(gaps["uncovered_sentences"]), fill_max)
    if n_fill > 0:
        print(f"      Filling {n_fill} uncovered sentences...")
        for sid in gaps["uncovered_sentences"][:n_fill]:
            data, t = fill_sentence(numbered_source, sid)
            if data:
                items = data.get("items") or []
                # Route each item to the correct graph based on predicate/content
                for raw_item in items:
                    item = normalize_item(raw_item)
                    if item is None:
                        continue
                    # Determine which graph this belongs to
                    pred = item["predicate"].lower()
                    if pred in ("is_a", "is a", "type"):
                        target_graph = "entities"
                    elif pred in ("has_value", "has_date", "value", "date"):
                        target_graph = "numbers"
                    elif pred in ("not", "denied", "absent", "bounded_by", "no"):
                        target_graph = "negations"
                    elif pred in ("alias", "member_of", "part_of", "teammate",
                                  "companion", "employer", "works_for", "related_to"):
                        target_graph = "relationships"
                    else:
                        target_graph = "events"  # default: actions
                    add_to_graph(graphs[target_graph], [item])
                print(f"        S{sid}: +{len(items)} items routed to graphs ({t:.0f}s)")

    # 5b: Fill entities that exist in G1 but have no events in G2
    n_ent_fill = min(len(gaps["entities_no_events"]), fill_max)
    if n_ent_fill > 0:
        print(f"      Filling {n_ent_fill} entities missing from G2 (events)...")
        for gap in gaps["entities_no_events"][:n_ent_fill]:
            data, t = fill_entity_in_graph(numbered_source, gap["name"], "events", gap["sentences"])
            if data:
                items = data.get("items") or []
                added = add_to_graph(graphs["events"], items)
                print(f"        {gap['name']}: +{added} events ({t:.0f}s)")
                log_samples(f"fill {gap['name']}", items, 2)

    # 5c: Register actors from G2 that are not in G1
    if gaps["actors_not_entities"]:
        print(f"      Auto-registering {len(gaps['actors_not_entities'])} actors into G1...")
        for gap in gaps["actors_not_entities"]:
            # Find the sentence where this actor first appears in G2
            actor_sents = []
            for item in graphs["events"]:
                if item["subject"].lower() == gap["name"].lower() and item["sentence_id"]:
                    actor_sents.append(item["sentence_id"])
            first_sent = min(actor_sents) if actor_sents else 0
            new_entity = {
                "sentence_id": first_sent,
                "subject": gap["name"],
                "predicate": "is_a",
                "object": "UNKNOWN",
                "qualifier": {"registered_from": "G2_actor"},
                "source_text": "",
            }
            add_to_graph(graphs["entities"], [new_entity])
        print(f"      G1 after registration: {len(graphs['entities'])}")

    # ==== STEP 6: REBUILD INDEXES + FINAL METRICS ====
    print(f"\n    --- Step 6: Final metrics ---")
    entity_idx = build_entity_index(graphs)
    sentence_idx = build_sentence_index(graphs, len(sentences))

    print_graph_summary(graphs)
    total_items = sum(len(g) for g in graphs.values())
    print(f"      Total across 5 graphs: {total_items}")

    # Sentence coverage
    covered = sum(1 for sid, cov in sentence_idx.items() if sum(cov.values()) > 0)
    coverage_pct = round(covered / len(sentences) * 100, 1) if sentences else 0
    print(f"      Sentence coverage: {covered}/{len(sentences)} ({coverage_pct}%)")

    # Entity coverage matrix
    n_in_g1 = sum(1 for e in entity_idx.values() if "entities" in e["in_graphs"])
    n_in_g2 = sum(1 for e in entity_idx.values() if "events" in e["in_graphs"])
    n_in_g3 = sum(1 for e in entity_idx.values() if "relationships" in e["in_graphs"])
    print(f"      Entities in G1: {n_in_g1}, in G2: {n_in_g2}, in G3: {n_in_g3}")

    metrics = {
        "entities": len(graphs["entities"]),
        "events": len(graphs["events"]),
        "relationships": len(graphs["relationships"]),
        "numbers": len(graphs["numbers"]),
        "negations": len(graphs["negations"]),
        "total": total_items,
    }

    return {
        "metrics_final": metrics,
        "sentence_coverage_pct": coverage_pct,
        "entity_index_size": len(entity_idx),
        "gaps_found": {k: len(v) if isinstance(v, list) else v
                       for k, v in gaps.items()},
        "times": all_times,
    }


# ========================================================================
# MAIN
# ========================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--top-k-entities", type=int, default=8)
    parser.add_argument("--fill-max", type=int, default=8)
    parser.add_argument("--baseline-path", default="experiments/exp31_multipass_fact_tables.json")
    args = parser.parse_args()

    print("=" * 70)
    print("EXPERIMENT 34 — Five-Graph Unified Knowledge Extraction")
    print("=" * 70)
    print("  Model: Gemma 4 31B via Google AI Studio")
    print("  5 Graphs: G1-entities, G2-events, G3-relationships, G4-numbers, G5-negations")
    print("  Pipeline: Extract → Index → Gap Detect → Fill → Re-index")
    print("=" * 70)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))

    source_map = {}
    for case in cases:
        source = "\n".join(case.request.source_context or []) if case.request.source_context else ""
        sh = hash(source)
        if sh not in source_map:
            source_map[sh] = {"source": source, "case_ids": []}
        source_map[sh]["case_ids"].append(case.case_id)

    sources = list(source_map.values())
    if args.max_sources:
        sources = sources[:args.max_sources]

    # Baseline
    baseline_metrics = []
    if os.path.exists(args.baseline_path):
        with open(args.baseline_path) as f:
            bl_data = json.load(f)

        def _bl(passes):
            def _sl(d, k): return len(d.get(k, []) if d else [])
            p1, p2, p3 = passes.get("P1_entities"), passes.get("P2_events"), passes.get("P3_relationships")
            p4, p5 = passes.get("P4_numbers"), passes.get("P5_negations")
            e, ev, r = _sl(p1,"entities"), _sl(p2,"events"), _sl(p3,"relationships")
            n = _sl(p4,"numerical_facts") + _sl(p4,"temporal_facts")
            ng = sum(_sl(p5,k) for k in ["explicit_negations","absent_information","boundaries","corrections"])
            return {"entities":e,"events":ev,"relationships":r,"numbers":n,"negations":ng,"total":e+ev+r+n+ng}

        for src in sources:
            pid = src["case_ids"][0]
            baseline_metrics.append(_bl(bl_data[pid]["passes"]) if pid in bl_data else None)
        print(f"  Baseline: {sum(1 for b in baseline_metrics if b)} sources")

    # Pre-process
    print(f"\n  Pre-processing...")
    pre_list = []
    for si, src in enumerate(sources):
        pre = preprocess_source(src["source"][:6000])
        pre_list.append(pre)
        print(f"    Source {si}: {pre['n_sentences']} sents, {len(pre['spacy_entities'])} spaCy ents")

    # Run
    t0 = time.time()
    all_results = []

    for si, src in enumerate(sources):
        case_ids = src["case_ids"]
        print(f"\n  {'='*60}")
        print(f"  [{si+1}/{len(sources)}] Source for {', '.join(case_ids[:3])}")
        print(f"  {'='*60}")

        result = run_source(src["source"][:6000], si, pre_list[si],
                           top_k_entities=args.top_k_entities, fill_max=args.fill_max)

        bl = baseline_metrics[si] if si < len(baseline_metrics) else None
        m = result["metrics_final"]
        if bl and bl["total"] > 0:
            pct = round(m["total"] / bl["total"] * 100, 1)
            print(f"\n    RESULT: {m['total']} items ({pct}% of baseline {bl['total']})")
            for k in ["entities", "events", "relationships", "numbers", "negations"]:
                bv = max(bl.get(k, 0), 1)
                print(f"      {k:15s}: {m[k]:>4d} vs {bl.get(k,0):>4d} ({round(m[k]/bv*100)}%)")
            result["vs_baseline_pct"] = pct
        else:
            result["vs_baseline_pct"] = None
        all_results.append({"source_idx": si, "case_ids": case_ids, **result})

    elapsed = time.time() - t0

    # Summary
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")

    bl_avg = {k: sum(b[k] for b in baseline_metrics if b) / max(sum(1 for b in baseline_metrics if b), 1)
              for k in ["total","entities","events","relationships"]}
    final_avg = {k: sum(r["metrics_final"][k] for r in all_results) / len(all_results)
                 for k in ["total","entities","events","relationships"]}
    pct = round(final_avg["total"] / bl_avg["total"] * 100, 1) if bl_avg["total"] > 0 else 0

    print(f"\n  {'Phase':<45s} {'Total':>7s} {'vs BL':>7s} {'Ent':>6s} {'Evt':>6s} {'Rel':>6s}")
    print(f"  {'-'*76}")
    print(f"  {'Gemini baseline':<45s} {bl_avg['total']:>7.1f} {'BASE':>7s} "
          f"{bl_avg['entities']:>6.1f} {bl_avg['events']:>6.1f} {bl_avg['relationships']:>6.1f}")
    print(f"  {'Exp 32: raw':<45s} {'137.5':>7s} {'66.7%':>7s} {'61.0':>6s} {'34.0':>6s} {'11.5':>6s}")
    print(f"  {'Exp 33c: chained+verified':<45s} {'154.0':>7s} {'85.7%':>7s} {'34.4':>6s} {'54.4':>6s} {'36.3':>6s}")
    print(f"  {'Exp 34: Five-Graph KG':<45s} {final_avg['total']:>7.1f} {pct:>6.1f}% "
          f"{final_avg['entities']:>6.1f} {final_avg['events']:>6.1f} {final_avg['relationships']:>6.1f}")
    print(f"  {'-'*76}")

    if pct >= 100: print(f"\n  ✅ EXCEEDED BASELINE — {pct}%")
    elif pct >= 90: print(f"\n  ✅ PASS — {pct}%")
    elif pct >= 80: print(f"\n  ⚠️  PARTIAL — {pct}%")
    else: print(f"\n  ❌ BELOW — {pct}%")

    output = {
        "experiment": "exp34_five_graph_kg", "model": "gemma-4-31b-it",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1), "n_sources": len(sources),
        "baseline_avg": bl_avg, "final_avg": final_avg, "vs_baseline_pct": pct,
        "results": all_results,
    }
    out_path = "experiments/exp34_cyclic_extraction_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
