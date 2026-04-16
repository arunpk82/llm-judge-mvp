"""
Experiment 35 — Five-Graph Knowledge Extraction (Simple Tasks)

Principle: ONE PROMPT, ONE TASK. Never ask the model to do 2+ things at once.

Architecture:
  Step 1: EXTRACT — proven Exp 33 prompts (one task each, 90-150s, 100% reliable)
  Step 2: SENTENCE MAP — code matches items to source sentences (zero LLM)
  Step 3: NORMALIZE — code converts to unified schema (zero LLM)
  Step 4: ROUTE — code distributes items to 5 graphs (zero LLM)
  Step 5: INDEX + GAP DETECT — code builds entity×graph matrix (zero LLM)
  Step 6: FILL — sharp single-task prompts for gaps only (LLM)

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python experiments/exp35_five_graph.py --max-sources 2
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
               timeout: int = 300) -> tuple[dict | None, float]:
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
# PRE-PROCESS: Number sentences + spaCy NER
# ========================================================================

def preprocess_source(source_text: str) -> dict:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', source_text) if s.strip()]
        return {"sentences": sentences, "n_sentences": len(sentences), "spacy_entities": []}

    doc = nlp(source_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    spacy_entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ not in ("PERSON", "ORG", "GPE", "NORP", "FAC", "LOC", "EVENT", "PRODUCT"):
            continue
        if ent.text.lower() in seen:
            continue
        seen.add(ent.text.lower())
        sent_id = next((i for i, s in enumerate(sentences, 1) if ent.text in s), 1)
        spacy_entities.append({"text": ent.text, "label": ent.label_, "sentence_id": sent_id})

    return {"sentences": sentences, "n_sentences": len(sentences), "spacy_entities": spacy_entities}


# ========================================================================
# STEP 1: EXTRACT — Proven Exp 33 prompts (one task each)
# ========================================================================

def extract_p1_entities(source: str) -> tuple[dict | None, float]:
    """One task: extract entities. PROVEN at 90s, 100% reliable."""
    prompt = f"""You are an entity extraction engine. Read the source document and extract EVERY person, organization, place, and thing mentioned.

For EACH entity, provide:
- name: exact name as used in the document
- type: PERSON / ORG / PLACE / PRODUCT / OTHER
- aliases: other names used for the same entity
- attributes: dict of stated properties (age, role, title, description)

CRITICAL: Extract EVERY entity, even minor ones.

Return ONLY valid JSON: {{"entities": [...]}}

Source:
\"\"\"{source}\"\"\""""
    return call_gemma(prompt)


def extract_p2_events(source: str, entity_names: list[str]) -> tuple[dict | None, float]:
    """One task: extract events. PROVEN at 100s, 100% reliable."""
    entity_list = ", ".join(entity_names)
    prompt = f"""You are an event extraction engine. Read the source document and extract EVERY event or action.

These entities were found: [{entity_list}]

For EACH event, provide:
- who: the actor (exact name)
- action: the verb/action (exact wording)
- target: who/what was acted upon
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)

CRITICAL: Go sentence by sentence. Include passive voice.

Return ONLY valid JSON: {{"events": [...]}}

Source:
\"\"\"{source}\"\"\""""
    data, elapsed = call_gemma(prompt)
    if data is None:
        data, e2 = call_gemma(prompt.replace(source, source[:4000]))
        elapsed += e2
    return data, elapsed


def extract_p2_micro(source: str, entity_name: str) -> tuple[dict | None, float]:
    """One task: what did this entity do? PROVEN at 30s, 100% reliable."""
    prompt = f"""Read the source document. What did "{entity_name}" do? List EVERY action they performed or that happened to them.

For EACH action, provide:
- action: the verb (exact wording)
- target: who/what was acted upon
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)

If "{entity_name}" did nothing, return {{"events": []}}

Return ONLY valid JSON: {{"events": [...]}}

Source:
\"\"\"{source}\"\"\""""
    return call_gemma(prompt)


def extract_p3_relationships(source: str, entity_names: list[str]) -> tuple[dict | None, float]:
    """One task: extract relationships. PROVEN at 150s, 100% reliable."""
    entity_list = ", ".join(entity_names)
    prompt = f"""You are a relationship extraction engine. Map ALL relationships between entities.

These entities were found: [{entity_list}]

For EVERY PAIR of entities that are connected, provide:
- entity1: first entity
- entity2: second entity
- relationship: type of connection (teammate, companion, employer, suspect, alias, etc.)

Return ONLY valid JSON: {{"relationships": [...]}}

Source:
\"\"\"{source}\"\"\""""
    return call_gemma(prompt)


def extract_p4_numbers(source: str) -> tuple[dict | None, float]:
    """One task: extract numbers. PROVEN at 90s, 100% reliable."""
    prompt = f"""You are a numerical fact extraction engine. Extract EVERY number, date, quantity, and measurement.

For EACH numerical fact, provide:
- entity: who or what the number describes
- number: the exact value
- unit: what it measures
- describes: what aspect
- context: the sentence where it appears

Return ONLY valid JSON: {{"numerical_facts": [...], "temporal_facts": [...]}}

Source:
\"\"\"{source}\"\"\""""
    return call_gemma(prompt)


def extract_p5_negations(source: str) -> tuple[dict | None, float]:
    """One task: extract negations. PROVEN at 35s, 100% reliable."""
    prompt = f"""You are a negation extraction engine. Identify everything denied, negated, absent, or bounded.

Extract:
1. explicit_negations: [{{"statement": "...", "context": "..."}}]
2. absent_information: [{{"what": "...", "why_notable": "..."}}]
3. boundaries: [{{"fact": "...", "boundary": "..."}}]
4. corrections: [{{"wrong": "...", "right": "..."}}]

Return ONLY valid JSON: {{"explicit_negations": [...], "absent_information": [...], "boundaries": [...], "corrections": [...]}}

Source:
\"\"\"{source}\"\"\""""
    return call_gemma(prompt)


def fill_sentence_facts(source: str, sentence_text: str, sent_id: int) -> tuple[dict | None, float]:
    """One task: what facts are in this sentence?"""
    prompt = f"""Read this ONE sentence and extract all facts:

"{sentence_text}"

List every fact (entity, event, relationship, number, or negation) in this sentence.
Return ONLY valid JSON: {{"facts": [{{"type": "entity/event/relationship/number/negation", "subject": "...", "predicate": "...", "object": "..."}}]}}
If no facts, return {{"facts": []}}"""
    return call_gemma(prompt)


def fill_entity_events(source: str, entity_name: str) -> tuple[dict | None, float]:
    """One task: what did this entity do? (Same as micro-extraction)"""
    return extract_p2_micro(source, entity_name)


# ========================================================================
# STEP 2: SENTENCE MAPPING — Code matches items to sentences (zero LLM)
# ========================================================================

def find_sentence_id(item_text: str, sentences: list[str]) -> int:
    """Find which sentence contains this text. Returns sentence_id (1-indexed)."""
    item_lower = item_text.lower().strip()
    if not item_lower:
        return 0

    # Exact substring match
    for i, sent in enumerate(sentences, 1):
        if item_lower in sent.lower():
            return i

    # Word overlap match (fallback)
    words = set(item_lower.split())
    if len(words) < 2:
        return 0
    best_id, best_overlap = 0, 0
    for i, sent in enumerate(sentences, 1):
        sent_words = set(sent.lower().split())
        overlap = len(words & sent_words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = i
    # Require at least 50% word overlap
    if best_overlap >= len(words) * 0.5:
        return best_id
    return 0


def map_entity_to_sentence(entity: dict, sentences: list[str]) -> int:
    """Map an entity to its first appearing sentence."""
    name = entity.get("name", "")
    return find_sentence_id(name, sentences)


def map_event_to_sentence(event: dict, sentences: list[str]) -> int:
    """Map an event to its sentence using who + action + target."""
    # Try full context first
    context = event.get("context", "")
    if context:
        sid = find_sentence_id(context, sentences)
        if sid:
            return sid

    # Try who + action combination
    who = event.get("who", "")
    action = event.get("action", "")
    target = str(event.get("target", ""))

    # Search for sentence containing both who and action
    for i, sent in enumerate(sentences, 1):
        sent_lower = sent.lower()
        has_who = who.lower() in sent_lower if who else True
        has_action = action.lower() in sent_lower if action else True
        if has_who and has_action:
            return i

    # Fallback: just who
    if who:
        return find_sentence_id(who, sentences)
    return 0


def map_relationship_to_sentence(rel: dict, sentences: list[str]) -> int:
    """Map a relationship to its sentence."""
    e1 = rel.get("entity1", "")
    e2 = rel.get("entity2", "")
    for i, sent in enumerate(sentences, 1):
        sent_lower = sent.lower()
        if e1.lower() in sent_lower and e2.lower() in sent_lower:
            return i
    # Fallback: just entity1
    return find_sentence_id(e1, sentences)


def map_number_to_sentence(num: dict, sentences: list[str]) -> int:
    """Map a number fact to its sentence."""
    context = num.get("context", "")
    if context:
        sid = find_sentence_id(context, sentences)
        if sid:
            return sid
    number = str(num.get("number", ""))
    entity = num.get("entity", "")
    for i, sent in enumerate(sentences, 1):
        if number in sent and (not entity or entity.lower() in sent.lower()):
            return i
    return find_sentence_id(entity, sentences)


def map_negation_to_sentence(neg: dict, sentences: list[str]) -> int:
    """Map a negation to its sentence."""
    context = neg.get("context", "")
    if context:
        return find_sentence_id(context, sentences)
    statement = neg.get("statement", neg.get("fact", neg.get("what", "")))
    return find_sentence_id(statement, sentences)


# ========================================================================
# STEP 3: NORMALIZE — Convert to unified schema (zero LLM)
# ========================================================================

def normalize_entity(entity: dict, sent_id: int, sent_text: str) -> dict:
    return {
        "sentence_id": sent_id,
        "subject": entity.get("name", ""),
        "predicate": "is_a",
        "object": entity.get("type", "UNKNOWN"),
        "qualifier": entity.get("attributes", {}),
        "source_text": sent_text,
    }


def normalize_event(event: dict, sent_id: int, sent_text: str) -> dict:
    return {
        "sentence_id": sent_id,
        "subject": event.get("who", ""),
        "predicate": event.get("action", ""),
        "object": str(event.get("target", "")),
        "qualifier": {k: event.get(k, "") for k in ["where", "when", "result"] if event.get(k)},
        "source_text": sent_text,
    }


def normalize_relationship(rel: dict, sent_id: int, sent_text: str) -> dict:
    return {
        "sentence_id": sent_id,
        "subject": rel.get("entity1", ""),
        "predicate": rel.get("relationship", ""),
        "object": rel.get("entity2", ""),
        "qualifier": {},
        "source_text": sent_text,
    }


def normalize_number(num: dict, sent_id: int, sent_text: str) -> dict:
    return {
        "sentence_id": sent_id,
        "subject": num.get("entity", ""),
        "predicate": "has_value",
        "object": str(num.get("number", "")),
        "qualifier": {k: num.get(k, "") for k in ["unit", "describes"] if num.get(k)},
        "source_text": sent_text,
    }


def normalize_negation(neg: dict, sent_id: int, sent_text: str) -> dict:
    statement = neg.get("statement", neg.get("fact", neg.get("what", "")))
    return {
        "sentence_id": sent_id,
        "subject": statement,
        "predicate": "negated",
        "object": neg.get("context", neg.get("boundary", neg.get("why_notable", ""))),
        "qualifier": {},
        "source_text": sent_text,
    }


# ========================================================================
# STEP 4: GRAPH OPERATIONS
# ========================================================================

GRAPH_NAMES = ["entities", "events", "relationships", "numbers", "negations"]


def dedup_key(item: dict) -> str:
    return f"{item['subject'].lower()}|{item['predicate'].lower()}|{item['object'].lower()}"


def add_to_graph(graph: list[dict], items: list[dict]) -> int:
    """Add items, dedup within graph. Returns count added."""
    existing = set(dedup_key(it) for it in graph)
    added = 0
    for item in items:
        if not item.get("subject"):
            continue
        key = dedup_key(item)
        if key not in existing:
            existing.add(key)
            graph.append(item)
            added += 1
    return added


# ========================================================================
# STEP 5: CROSS-REFERENCE INDEXES + GAP DETECTION
# ========================================================================

def build_entity_index(graphs: dict[str, list[dict]]) -> dict[str, dict]:
    index = {}
    for gname in GRAPH_NAMES:
        for item in graphs[gname]:
            for field in ["subject", "object"]:
                name = item.get(field, "")
                if not name or name in ("", "None", "?", "-", "UNKNOWN"):
                    continue
                nl = name.lower()
                if nl not in index:
                    index[nl] = {"name": name, "sentences": set(), "in_graphs": set()}
                index[nl]["in_graphs"].add(gname)
                if item.get("sentence_id"):
                    index[nl]["sentences"].add(item["sentence_id"])
    for entry in index.values():
        entry["sentences"] = sorted(entry["sentences"])
        entry["in_graphs"] = sorted(entry["in_graphs"])
    return index


def build_sentence_index(graphs: dict[str, list[dict]], n_sents: int) -> dict[int, dict]:
    index = {sid: {g: 0 for g in GRAPH_NAMES} for sid in range(1, n_sents + 1)}
    for gname in GRAPH_NAMES:
        for item in graphs[gname]:
            sid = item.get("sentence_id", 0)
            if sid in index:
                index[sid][gname] += 1
    return index


def detect_gaps(graphs: dict, entity_idx: dict, sentence_idx: dict) -> dict:
    uncovered = [sid for sid, cov in sentence_idx.items() if sum(cov.values()) == 0]

    entities_no_events = [
        {"name": e["name"], "sentences": e["sentences"], "has": e["in_graphs"]}
        for nl, e in entity_idx.items()
        if "entities" in e["in_graphs"] and "events" not in e["in_graphs"]
    ]

    actors_not_entities = [
        {"name": e["name"], "sentences": e["sentences"]}
        for nl, e in entity_idx.items()
        if "events" in e["in_graphs"] and "entities" not in e["in_graphs"]
    ]

    sents_ent_no_evt = [
        sid for sid, cov in sentence_idx.items()
        if cov["entities"] > 0 and cov["events"] == 0
    ]

    return {
        "uncovered_sentences": uncovered,
        "entities_no_events": entities_no_events[:15],
        "actors_not_entities": actors_not_entities[:15],
        "sents_entity_no_event": sents_ent_no_evt,
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
            obj = str(item.get("object", "?"))[:25]
            print(f"        [{label} {i+1}] S{sid} {subj} | {pred} | {obj}")
        else:
            print(f"        [{label} {i+1}] {str(item)[:80]}")
    if len(items) > max_show:
        print(f"        ... and {len(items) - max_show} more")


def print_graph_summary(graphs: dict):
    for gname in GRAPH_NAMES:
        items = graphs[gname]
        sids = set(it.get("sentence_id", 0) for it in items if it.get("sentence_id"))
        print(f"      G-{gname:<15s}: {len(items):>3d} items, {len(sids):>2d} sentences covered")


def print_entity_matrix(entity_idx: dict, max_show: int = 12):
    print(f"      {'Entity':<22s} {'Sents':>5s}  Ent  Evt  Rel  Num  Neg")
    print(f"      {'-'*60}")
    sorted_ents = sorted(entity_idx.items(), key=lambda x: len(x[1]["in_graphs"]), reverse=True)
    for nl, entry in sorted_ents[:max_show]:
        name = entry["name"][:21]
        sents = ",".join(str(s) for s in entry["sentences"][:3])
        flags = ""
        for g in GRAPH_NAMES:
            flags += "  ✓  " if g in entry["in_graphs"] else "  ·  "
        print(f"      {name:<22s} {sents:>5s}{flags}")
    if len(entity_idx) > max_show:
        print(f"      ... and {len(entity_idx) - max_show} more entities")


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_source(source: str, source_idx: int, pre: dict,
               top_k_entities: int = 8, fill_max: int = 8) -> dict:

    sentences = pre["sentences"]
    all_times = {}
    graphs = {g: [] for g in GRAPH_NAMES}

    # ================================================================
    # STEP 1: EXTRACT with proven prompts
    # ================================================================
    print(f"    --- Step 1: Extract (proven prompts, one task each) ---")

    # P1: Entities
    data, t = extract_p1_entities(source)
    all_times["P1"] = round(t, 1)
    p1_raw = data.get("entities", []) if data else []
    print(f"      P1 entities: {len(p1_raw)} ({t:.0f}s)")

    # P1 entity names for chaining
    entity_names = list(set(e.get("name", "") for e in p1_raw if e.get("name")))
    for se in pre.get("spacy_entities", []):
        if se["text"] not in entity_names:
            entity_names.append(se["text"])
    print(f"      Chaining list: {len(entity_names)} (P1:{len(p1_raw)} + spaCy:{len(entity_names)-len(set(e.get('name','') for e in p1_raw if e.get('name')))})")

    # P2: Events (chained)
    data, t = extract_p2_events(source, entity_names)
    all_times["P2"] = round(t, 1)
    p2_raw = data.get("events", []) if data else []
    print(f"      P2 events: {len(p2_raw)} ({t:.0f}s)")

    # P2 micro-extraction
    print(f"      [Micro-extraction × {top_k_entities} entities]")
    micro_raw = []
    for ent_name in entity_names[:top_k_entities]:
        if not ent_name:
            continue
        data, t = extract_p2_micro(source, ent_name)
        if data:
            evts = data.get("events", [])
            for evt in evts:
                evt["who"] = ent_name
            micro_raw.extend(evts)
            print(f"        {ent_name}: {len(evts)} ({t:.0f}s)")
    print(f"      Micro total: {len(micro_raw)}")

    # P3: Relationships (chained)
    data, t = extract_p3_relationships(source, entity_names)
    all_times["P3"] = round(t, 1)
    p3_raw = data.get("relationships", []) if data else []
    print(f"      P3 relationships: {len(p3_raw)} ({t:.0f}s)")

    # P4: Numbers
    data, t = extract_p4_numbers(source)
    all_times["P4"] = round(t, 1)
    p4_num = (data.get("numerical_facts", []) if data else [])
    p4_tmp = (data.get("temporal_facts", []) if data else [])
    p4_raw = p4_num + p4_tmp
    print(f"      P4 numbers: {len(p4_raw)} ({t:.0f}s)")

    # P5: Negations
    data, t = extract_p5_negations(source)
    all_times["P5"] = round(t, 1)
    p5_raw = []
    if data:
        for key in ["explicit_negations", "absent_information", "boundaries", "corrections"]:
            for item in data.get(key, []):
                item["_neg_type"] = key
                p5_raw.append(item)
    print(f"      P5 negations: {len(p5_raw)} ({t:.0f}s)")

    # ================================================================
    # STEP 2 + 3: SENTENCE MAP + NORMALIZE (zero LLM)
    # ================================================================
    print(f"\n    --- Step 2+3: Sentence map + Normalize (code only) ---")

    def get_sent_text(sid):
        return sentences[sid - 1] if 0 < sid <= len(sentences) else ""

    # G1: Entities
    for ent in p1_raw:
        sid = map_entity_to_sentence(ent, sentences)
        item = normalize_entity(ent, sid, get_sent_text(sid))
        if item["subject"]:
            graphs["entities"].append(item)

    # G2: Events (P2 + micro)
    all_events = p2_raw + micro_raw
    for evt in all_events:
        sid = map_event_to_sentence(evt, sentences)
        item = normalize_event(evt, sid, get_sent_text(sid))
        if item["subject"]:
            graphs["events"].append(item)

    # G3: Relationships
    for rel in p3_raw:
        sid = map_relationship_to_sentence(rel, sentences)
        item = normalize_relationship(rel, sid, get_sent_text(sid))
        if item["subject"]:
            graphs["relationships"].append(item)

    # G4: Numbers
    for num in p4_raw:
        sid = map_number_to_sentence(num, sentences)
        item = normalize_number(num, sid, get_sent_text(sid))
        if item["subject"]:
            graphs["numbers"].append(item)

    # G5: Negations
    for neg in p5_raw:
        sid = map_negation_to_sentence(neg, sentences)
        item = normalize_negation(neg, sid, get_sent_text(sid))
        if item["subject"]:
            graphs["negations"].append(item)

    # Dedup within each graph
    for gname in GRAPH_NAMES:
        seen = set()
        deduped = []
        for item in graphs[gname]:
            key = dedup_key(item)
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        graphs[gname] = deduped

    print_graph_summary(graphs)
    total_items = sum(len(g) for g in graphs.values())
    print(f"      Total across 5 graphs: {total_items}")
    for gname in GRAPH_NAMES:
        log_samples(gname, graphs[gname], 2)

    # ================================================================
    # STEP 5: INDEX + GAP DETECTION (zero LLM)
    # ================================================================
    print(f"\n    --- Step 5: Cross-reference + Gap detection ---")
    entity_idx = build_entity_index(graphs)
    sentence_idx = build_sentence_index(graphs, len(sentences))

    print(f"\n      Entity × Graph matrix:")
    print_entity_matrix(entity_idx)

    gaps = detect_gaps(graphs, entity_idx, sentence_idx)
    print(f"\n      Gaps found:")
    print(f"        Uncovered sentences: {len(gaps['uncovered_sentences'])}/{len(sentences)}")
    for sid in gaps["uncovered_sentences"][:5]:
        print(f"          [{sid}] {sentences[sid-1][:70]}")
    if len(gaps["uncovered_sentences"]) > 5:
        print(f"          ... and {len(gaps['uncovered_sentences'])-5} more")
    print(f"        Entities without events: {len(gaps['entities_no_events'])}")
    for g in gaps["entities_no_events"][:5]:
        print(f"          {g['name']} (has: {g['has']})")
    print(f"        Actors not registered in G1: {len(gaps['actors_not_entities'])}")
    print(f"        Sentences with entities but no events: {len(gaps['sents_entity_no_event'])}")

    # ================================================================
    # STEP 6: TARGETED FILL (LLM, one task each)
    # ================================================================
    print(f"\n    --- Step 6: Targeted fill (sharp questions) ---")

    # 6a: Fill uncovered sentences
    n_fill = min(len(gaps["uncovered_sentences"]), fill_max)
    if n_fill > 0:
        print(f"      Filling {n_fill} uncovered sentences...")
        for sid in gaps["uncovered_sentences"][:n_fill]:
            data, t = fill_sentence_facts(source, sentences[sid-1], sid)
            if data:
                facts = data.get("facts", [])
                for fact in facts:
                    ftype = fact.get("type", "event")
                    item = {
                        "sentence_id": sid,
                        "subject": fact.get("subject", ""),
                        "predicate": fact.get("predicate", ""),
                        "object": str(fact.get("object", "")),
                        "qualifier": {},
                        "source_text": sentences[sid-1],
                    }
                    if not item["subject"]:
                        continue
                    target_graph = {
                        "entity": "entities", "event": "events",
                        "relationship": "relationships", "number": "numbers",
                        "negation": "negations",
                    }.get(ftype, "events")
                    add_to_graph(graphs[target_graph], [item])
                print(f"        S{sid}: +{len(facts)} facts ({t:.0f}s)")
                log_samples(f"S{sid}", [{"sentence_id": sid, **f} for f in facts], 2)

    # 6b: Fill entities without events
    n_ent_fill = min(len(gaps["entities_no_events"]), fill_max)
    if n_ent_fill > 0:
        print(f"      Filling {n_ent_fill} entities without events...")
        for gap in gaps["entities_no_events"][:n_ent_fill]:
            data, t = fill_entity_events(source, gap["name"])
            if data:
                evts = data.get("events", [])
                normalized = []
                for evt in evts:
                    evt["who"] = gap["name"]
                    sid = map_event_to_sentence(evt, sentences)
                    normalized.append(normalize_event(evt, sid, get_sent_text(sid)))
                added = add_to_graph(graphs["events"], normalized)
                print(f"        {gap['name']}: +{added} events ({t:.0f}s)")
                log_samples(f"fill {gap['name']}", normalized, 2)

    # 6c: Auto-register actors into G1 (deterministic)
    if gaps["actors_not_entities"]:
        print(f"      Auto-registering {len(gaps['actors_not_entities'])} actors into G1...")
        for gap in gaps["actors_not_entities"]:
            sid = gap["sentences"][0] if gap["sentences"] else 0
            item = {
                "sentence_id": sid,
                "subject": gap["name"],
                "predicate": "is_a",
                "object": "UNKNOWN",
                "qualifier": {"registered_from": "G2_actor"},
                "source_text": get_sent_text(sid),
            }
            add_to_graph(graphs["entities"], [item])
        print(f"      G1 after registration: {len(graphs['entities'])}")

    # ================================================================
    # FINAL METRICS
    # ================================================================
    print(f"\n    --- Final metrics ---")
    entity_idx = build_entity_index(graphs)
    sentence_idx = build_sentence_index(graphs, len(sentences))

    print_graph_summary(graphs)
    total_items = sum(len(g) for g in graphs.values())
    covered = sum(1 for sid, cov in sentence_idx.items() if sum(cov.values()) > 0)
    coverage_pct = round(covered / len(sentences) * 100, 1) if sentences else 0

    print(f"      Total: {total_items}")
    print(f"      Sentence coverage: {covered}/{len(sentences)} ({coverage_pct}%)")
    print(f"      Entity index: {len(entity_idx)} entities")

    print(f"\n      Final Entity × Graph matrix:")
    print_entity_matrix(entity_idx)

    metrics = {g: len(graphs[g]) for g in GRAPH_NAMES}
    metrics["total"] = total_items

    return {
        "metrics_final": metrics,
        "sentence_coverage_pct": coverage_pct,
        "entity_index_size": len(entity_idx),
        "gaps": {k: len(v) if isinstance(v, list) else v for k, v in gaps.items()},
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
    print("EXPERIMENT 35 — Five-Graph KG (Simple Tasks, Code Post-Processing)")
    print("=" * 70)
    print("  Model: Gemma 4 31B via Google AI Studio")
    print("  Principle: ONE PROMPT, ONE TASK")
    print("  LLM: Extract + Fill gaps | Code: Map, Normalize, Route, Index, Detect")
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
            for k in GRAPH_NAMES:
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
    print(f"  {'Exp 35: Five-Graph (simple tasks)':<45s} {final_avg['total']:>7.1f} {pct:>6.1f}% "
          f"{final_avg['entities']:>6.1f} {final_avg['events']:>6.1f} {final_avg['relationships']:>6.1f}")
    print(f"  {'-'*76}")

    avg_cov = sum(r["sentence_coverage_pct"] for r in all_results) / len(all_results)
    print(f"\n  Avg sentence coverage: {avg_cov:.1f}%")

    if pct >= 100: print(f"\n  ✅ EXCEEDED BASELINE — {pct}%")
    elif pct >= 90: print(f"\n  ✅ PASS — {pct}%")
    elif pct >= 80: print(f"\n  ⚠️  PARTIAL — {pct}%")
    else: print(f"\n  ❌ BELOW — {pct}%")

    output = {
        "experiment": "exp35_five_graph_simple_tasks", "model": "gemma-4-31b-it",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1), "n_sources": len(sources),
        "baseline_avg": bl_avg, "final_avg": final_avg, "vs_baseline_pct": pct,
        "results": all_results,
    }
    out_path = "experiments/exp35_five_graph_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
