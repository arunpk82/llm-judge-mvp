"""
Experiment 36 — RAG-Enhanced Five-Graph Knowledge Extraction

Evolution of Exp 35: adds retrieval-augmented generation to micro-extraction
and gap-fill steps. Initial 5 passes use full source (discovery). All follow-up
queries use entity index to retrieve only relevant sentences.

Expected impact:
  Micro-extraction: 30-70s → 5-15s per query (5-10x speedup)
  Gap fills: same speedup
  Local CPU viability: 100s → 15-30s per query

Usage:
    export GEMINI_API_KEY=your_key
    poetry run python experiments/exp36_rag_five_graph.py --max-sources 2
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
# PRE-PROCESS
# ========================================================================

TYPE_LABELS = {"PERSON", "ORG", "PLACE", "PRODUCT", "OTHER", "UNKNOWN",
               "THING", "EVENT", "CONCEPT"}


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
# RAG: RETRIEVAL FUNCTIONS
# ========================================================================

def retrieve_entity_context(entity_name: str, sentences: list[str],
                             window: int = 1) -> tuple[str, list[int]]:
    """Retrieve sentences where entity appears, with ±window neighbor context."""
    entity_lower = entity_name.lower()
    matching_sids = []
    for i, sent in enumerate(sentences, 1):
        if entity_lower in sent.lower():
            matching_sids.append(i)

    if not matching_sids:
        # Fallback: word overlap
        words = set(entity_lower.split())
        for i, sent in enumerate(sentences, 1):
            if words & set(sent.lower().split()):
                matching_sids.append(i)
                if len(matching_sids) >= 3:
                    break

    # Expand with window
    expanded = set()
    for sid in matching_sids:
        for offset in range(-window, window + 1):
            s = sid + offset
            if 1 <= s <= len(sentences):
                expanded.add(s)

    sorted_sids = sorted(expanded)
    context = "\n".join(f"[{s}] {sentences[s-1]}" for s in sorted_sids)
    return context, sorted_sids


def retrieve_sentence_context(sentence_id: int, sentences: list[str],
                               window: int = 1) -> str:
    """Retrieve a sentence with ±window neighbors."""
    lines = []
    for offset in range(-window, window + 1):
        s = sentence_id + offset
        if 1 <= s <= len(sentences):
            lines.append(f"[{s}] {sentences[s-1]}")
    return "\n".join(lines)


# ========================================================================
# STEP 1: EXTRACT — Full source for discovery, RAG for micro/fills
# ========================================================================

def extract_p1_entities(source: str) -> tuple[dict | None, float]:
    """Discovery: full source. One task: extract entities."""
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
    """Discovery: full source. One task: extract events."""
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


def extract_p2_micro_rag(entity_name: str, rag_context: str,
                          rag_sids: list[int]) -> tuple[dict | None, float]:
    """RAG micro-extraction: only entity's sentences. One task: what did X do?"""
    # sid_str = ", ".join(str(s) for s in rag_sids)
    prompt = f"""Read the sentences below. What did "{entity_name}" do? List EVERY action they performed or that happened to them.

For EACH action:
- action: the verb (exact wording)
- target: who/what was acted upon
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)

If "{entity_name}" did nothing, return {{"events": []}}

Return ONLY valid JSON: {{"events": [...]}}

Relevant sentences (from document):
{rag_context}"""
    return call_gemma(prompt)


def extract_p3_relationships(source: str, entity_names: list[str]) -> tuple[dict | None, float]:
    """Discovery: full source. One task: extract relationships."""
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
    """Discovery: full source. One task: extract numbers."""
    prompt = f"""You are a numerical fact extraction engine. Extract EVERY number, date, quantity, and measurement.

For EACH:
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
    """Discovery: full source. One task: extract negations."""
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


# ========================================================================
# STEP 6: TARGETED FILL — RAG for all fill queries
# ========================================================================

def fill_sentence_rag(rag_context: str, sent_id: int) -> tuple[dict | None, float]:
    """RAG fill: sentence + neighbors only. One task: what facts here?"""
    prompt = f"""Read the sentences below. Extract ALL facts from sentence [{sent_id}].

List every fact (entity, event, relationship, number, or negation).
Return ONLY valid JSON: {{"facts": [{{"type": "entity/event/relationship/number/negation", "subject": "...", "predicate": "...", "object": "..."}}]}}
If no facts, return {{"facts": []}}

Context:
{rag_context}"""
    return call_gemma(prompt)


def fill_entity_events_rag(entity_name: str, rag_context: str) -> tuple[dict | None, float]:
    """RAG fill: entity's sentences only. One task: what did X do?"""
    prompt = f"""Read the sentences below. What did "{entity_name}" do? List EVERY action.

For EACH action:
- action: the verb
- target: who/what was acted upon
- where: location (if stated)
- when: time/date (if stated)
- result: outcome (if stated)

If no actions, return {{"events": []}}

Return ONLY valid JSON: {{"events": [...]}}

Context:
{rag_context}"""
    return call_gemma(prompt)


# ========================================================================
# SENTENCE MAPPING — Code (zero LLM)
# ========================================================================

def find_sentence_id(text: str, sentences: list[str]) -> int:
    text_lower = text.lower().strip()
    if not text_lower:
        return 0
    for i, sent in enumerate(sentences, 1):
        if text_lower in sent.lower():
            return i
    words = set(text_lower.split())
    if len(words) < 2:
        return 0
    best_id, best_overlap = 0, 0
    for i, sent in enumerate(sentences, 1):
        overlap = len(words & set(sent.lower().split()))
        if overlap > best_overlap:
            best_overlap = overlap
            best_id = i
    return best_id if best_overlap >= len(words) * 0.5 else 0


def map_entity(ent: dict, sentences: list[str]) -> int:
    return find_sentence_id(ent.get("name", ""), sentences)


def map_event(evt: dict, sentences: list[str]) -> int:
    ctx = evt.get("context", "")
    if ctx:
        sid = find_sentence_id(ctx, sentences)
        if sid:
            return sid
    who = evt.get("who", "")
    action = evt.get("action", "")
    for i, sent in enumerate(sentences, 1):
        sl = sent.lower()
        if (not who or who.lower() in sl) and (not action or action.lower() in sl):
            return i
    return find_sentence_id(who, sentences) if who else 0


def map_relationship(rel: dict, sentences: list[str]) -> int:
    e1, e2 = rel.get("entity1", ""), rel.get("entity2", "")
    for i, sent in enumerate(sentences, 1):
        sl = sent.lower()
        if e1.lower() in sl and e2.lower() in sl:
            return i
    return find_sentence_id(e1, sentences)


def map_number(num: dict, sentences: list[str]) -> int:
    ctx = num.get("context", "")
    if ctx:
        sid = find_sentence_id(ctx, sentences)
        if sid:
            return sid
    number = str(num.get("number", ""))
    entity = num.get("entity", "")
    for i, sent in enumerate(sentences, 1):
        if number in sent and (not entity or entity.lower() in sent.lower()):
            return i
    return find_sentence_id(entity, sentences)


def map_negation(neg: dict, sentences: list[str]) -> int:
    ctx = neg.get("context", "")
    if ctx:
        return find_sentence_id(ctx, sentences)
    stmt = neg.get("statement", neg.get("fact", neg.get("what", "")))
    return find_sentence_id(stmt, sentences)


# ========================================================================
# NORMALIZE — Code (zero LLM)
# ========================================================================

def norm_entity(e: dict, sid: int, st: str) -> dict | None:
    name = e.get("name", "")
    if not name or name.upper() in TYPE_LABELS:
        return None
    return {"sentence_id": sid, "subject": name, "predicate": "is_a",
            "object": e.get("type", "UNKNOWN"), "qualifier": e.get("attributes", {}),
            "source_text": st}


def norm_event(e: dict, sid: int, st: str) -> dict | None:
    who = e.get("who", "")
    if not who:
        return None
    return {"sentence_id": sid, "subject": who, "predicate": e.get("action", ""),
            "object": str(e.get("target", "")),
            "qualifier": {k: e[k] for k in ["where", "when", "result"] if e.get(k)},
            "source_text": st}


def norm_rel(r: dict, sid: int, st: str) -> dict | None:
    e1 = r.get("entity1", "")
    if not e1:
        return None
    return {"sentence_id": sid, "subject": e1, "predicate": r.get("relationship", ""),
            "object": r.get("entity2", ""), "qualifier": {}, "source_text": st}


def norm_num(n: dict, sid: int, st: str) -> dict | None:
    ent = n.get("entity", "")
    if not ent:
        return None
    return {"sentence_id": sid, "subject": ent, "predicate": "has_value",
            "object": str(n.get("number", "")),
            "qualifier": {k: n[k] for k in ["unit", "describes"] if n.get(k)},
            "source_text": st}


def norm_neg(n: dict, sid: int, st: str) -> dict | None:
    stmt = n.get("statement", n.get("fact", n.get("what", "")))
    if not stmt:
        return None
    return {"sentence_id": sid, "subject": stmt, "predicate": "negated",
            "object": n.get("context", n.get("boundary", n.get("why_notable", ""))),
            "qualifier": {}, "source_text": st}


# ========================================================================
# GRAPH OPERATIONS
# ========================================================================

GRAPH_NAMES = ["entities", "events", "relationships", "numbers", "negations"]


def dedup_key(item: dict) -> str:
    return f"{item['subject'].lower()}|{item['predicate'].lower()}|{item['object'].lower()}"


def add_to_graph(graph: list[dict], items: list[dict]) -> int:
    existing = set(dedup_key(it) for it in graph)
    added = 0
    for item in items:
        if not item or not item.get("subject"):
            continue
        key = dedup_key(item)
        if key not in existing:
            existing.add(key)
            graph.append(item)
            added += 1
    return added


# ========================================================================
# CROSS-REFERENCE + GAP DETECTION
# ========================================================================

def build_entity_index(graphs: dict) -> dict:
    index = {}
    for gname in GRAPH_NAMES:
        for item in graphs[gname]:
            for field in ["subject", "object"]:
                name = item.get(field, "")
                if not name or name.upper() in TYPE_LABELS or name in ("", "?", "-"):
                    continue
                nl = name.lower()
                if nl not in index:
                    index[nl] = {"name": name, "sentences": set(), "in_graphs": set()}
                index[nl]["in_graphs"].add(gname)
                if item.get("sentence_id"):
                    index[nl]["sentences"].add(item["sentence_id"])
    for e in index.values():
        e["sentences"] = sorted(e["sentences"])
        e["in_graphs"] = sorted(e["in_graphs"])
    return index


def build_sentence_index(graphs: dict, n: int) -> dict:
    idx = {s: {g: 0 for g in GRAPH_NAMES} for s in range(1, n + 1)}
    for g in GRAPH_NAMES:
        for item in graphs[g]:
            s = item.get("sentence_id", 0)
            if s in idx:
                idx[s][g] += 1
    return idx


def detect_gaps(graphs: dict, ent_idx: dict, sent_idx: dict) -> dict:
    uncovered = [s for s, c in sent_idx.items() if sum(c.values()) == 0]
    ent_no_evt = [{"name": e["name"], "sentences": e["sentences"], "has": e["in_graphs"]}
                  for nl, e in ent_idx.items()
                  if "entities" in e["in_graphs"] and "events" not in e["in_graphs"]]
    actors_unreg = [{"name": e["name"], "sentences": e["sentences"]}
                    for nl, e in ent_idx.items()
                    if "events" in e["in_graphs"] and "entities" not in e["in_graphs"]]
    return {
        "uncovered_sentences": uncovered,
        "entities_no_events": ent_no_evt[:15],
        "actors_not_entities": actors_unreg[:15],
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
    for g in GRAPH_NAMES:
        sids = set(it.get("sentence_id", 0) for it in graphs[g] if it.get("sentence_id"))
        print(f"      G-{g:<15s}: {len(graphs[g]):>3d} items, {len(sids):>2d} sentences")


def print_entity_matrix(ent_idx: dict, max_show: int = 12):
    print(f"      {'Entity':<22s} {'Sents':>5s}  Ent  Evt  Rel  Num  Neg")
    print(f"      {'-'*60}")
    for nl, e in sorted(ent_idx.items(), key=lambda x: len(x[1]["in_graphs"]), reverse=True)[:max_show]:
        name = e["name"][:21]
        sents = ",".join(str(s) for s in e["sentences"][:3])
        flags = "".join("  ✓  " if g in e["in_graphs"] else "  ·  " for g in GRAPH_NAMES)
        print(f"      {name:<22s} {sents:>5s}{flags}")
    if len(ent_idx) > max_show:
        print(f"      ... and {len(ent_idx) - max_show} more")


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def run_source(source: str, source_idx: int, pre: dict,
               top_k_entities: int = 8, fill_max: int = 8) -> dict:

    sentences = pre["sentences"]
    all_times = {}
    graphs: dict = {g: [] for g in GRAPH_NAMES}

    def get_st(sid):
        return sentences[sid - 1] if 0 < sid <= len(sentences) else ""

    # ================================================================
    # STEP 1a: EXTRACT with full source (discovery)
    # ================================================================
    print("    --- Step 1a: Discovery extraction (full source) ---")

    data, t = extract_p1_entities(source)
    all_times["P1"] = round(t, 1)
    p1_raw = data.get("entities", []) if data else []
    print(f"      P1 entities: {len(p1_raw)} ({t:.0f}s)")

    entity_names = list(set(e.get("name", "") for e in p1_raw if e.get("name")))
    for se in pre.get("spacy_entities", []):
        if se["text"] not in entity_names:
            entity_names.append(se["text"])
    n_p1 = len(set(e.get("name", "") for e in p1_raw if e.get("name")))
    print(f"      Chaining: {len(entity_names)} (P1:{n_p1} + spaCy:{len(entity_names)-n_p1})")

    data, t = extract_p2_events(source, entity_names)
    all_times["P2"] = round(t, 1)
    p2_raw = data.get("events", []) if data else []
    print(f"      P2 events: {len(p2_raw)} ({t:.0f}s)")

    # ================================================================
    # STEP 1b: MICRO-EXTRACTION with RAG
    # ================================================================
    print(f"\n    --- Step 1b: Micro-extraction with RAG × {top_k_entities} ---")
    micro_raw = []
    for ent_name in entity_names[:top_k_entities]:
        if not ent_name:
            continue
        rag_context, rag_sids = retrieve_entity_context(ent_name, sentences, window=1)
        n_sents = len(rag_sids)
        data, t = extract_p2_micro_rag(ent_name, rag_context, rag_sids)
        if data:
            evts = data.get("events", [])
            for evt in evts:
                evt["who"] = ent_name
            micro_raw.extend(evts)
            print(f"      {ent_name}: {len(evts)} events from {n_sents} sents ({t:.0f}s)")
        else:
            print(f"      {ent_name}: FAILED ({t:.0f}s)")
    print(f"      Micro total: {len(micro_raw)} events")

    # ================================================================
    # STEP 1c: Remaining discovery passes
    # ================================================================
    print("\n    --- Step 1c: Remaining discovery (P3-P5) ---")

    data, t = extract_p3_relationships(source, entity_names)
    all_times["P3"] = round(t, 1)
    p3_raw = data.get("relationships", []) if data else []
    print(f"      P3 relationships: {len(p3_raw)} ({t:.0f}s)")

    data, t = extract_p4_numbers(source)
    all_times["P4"] = round(t, 1)
    p4_raw = (data.get("numerical_facts", []) + data.get("temporal_facts", [])) if data else []
    print(f"      P4 numbers: {len(p4_raw)} ({t:.0f}s)")

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
    # STEP 2+3: SENTENCE MAP + NORMALIZE + ROUTE (code only)
    # ================================================================
    print("\n    --- Step 2+3: Map + Normalize + Route (code) ---")

    for ent in p1_raw:
        sid = map_entity(ent, sentences)
        item = norm_entity(ent, sid, get_st(sid))
        if item:
            graphs["entities"].append(item)

    for evt in p2_raw + micro_raw:
        sid = map_event(evt, sentences)
        item = norm_event(evt, sid, get_st(sid))
        if item:
            graphs["events"].append(item)

    for rel in p3_raw:
        sid = map_relationship(rel, sentences)
        item = norm_rel(rel, sid, get_st(sid))
        if item:
            graphs["relationships"].append(item)

    for num in p4_raw:
        sid = map_number(num, sentences)
        item = norm_num(num, sid, get_st(sid))
        if item:
            graphs["numbers"].append(item)

    for neg in p5_raw:
        sid = map_negation(neg, sentences)
        item = norm_neg(neg, sid, get_st(sid))
        if item:
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
    total = sum(len(g) for g in graphs.values())
    print(f"      Total: {total}")

    # ================================================================
    # STEP 5: INDEX + GAP DETECT (code)
    # ================================================================
    print("\n    --- Step 5: Cross-reference + Gap detection ---")
    ent_idx = build_entity_index(graphs)
    sent_idx = build_sentence_index(graphs, len(sentences))

    print("\n      Entity × Graph matrix:")
    print_entity_matrix(ent_idx)

    gaps = detect_gaps(graphs, ent_idx, sent_idx)
    print("\n      Gaps:")
    print(f"        Uncovered sentences: {len(gaps['uncovered_sentences'])}/{len(sentences)}")
    for sid in gaps["uncovered_sentences"][:5]:
        print(f"          [{sid}] {sentences[sid-1][:70]}")
    if len(gaps["uncovered_sentences"]) > 5:
        print(f"          ... and {len(gaps['uncovered_sentences'])-5} more")
    print(f"        Entities without events: {len(gaps['entities_no_events'])}")
    for g in gaps["entities_no_events"][:5]:
        print(f"          {g['name']} (has: {g['has']})")
    print(f"        Actors not in G1: {len(gaps['actors_not_entities'])}")

    # ================================================================
    # STEP 6: TARGETED FILL with RAG
    # ================================================================
    print("\n    --- Step 6: Fill gaps with RAG ---")

    # 6a: Fill uncovered sentences (RAG: sentence + neighbors)
    n_fill = min(len(gaps["uncovered_sentences"]), fill_max)
    if n_fill > 0:
        print(f"      Filling {n_fill} uncovered sentences (RAG: ±1 neighbor)...")
        for sid in gaps["uncovered_sentences"][:n_fill]:
            rag_ctx = retrieve_sentence_context(sid, sentences, window=1)
            data, t = fill_sentence_rag(rag_ctx, sid)
            if data:
                facts = data.get("facts", [])
                for fact in facts:
                    ftype = fact.get("type", "event")
                    item = {"sentence_id": sid, "subject": fact.get("subject", ""),
                            "predicate": fact.get("predicate", ""),
                            "object": str(fact.get("object", "")),
                            "qualifier": {}, "source_text": get_st(sid)}
                    if not item["subject"]:
                        continue
                    tg = {"entity": "entities", "event": "events",
                          "relationship": "relationships", "number": "numbers",
                          "negation": "negations"}.get(ftype, "events")
                    add_to_graph(graphs[tg], [item])
                print(f"        S{sid}: +{len(facts)} facts from {len(rag_ctx.split(chr(10)))} sents ({t:.0f}s)")
                log_samples(f"S{sid}", [{"sentence_id": sid, **f} for f in facts], 2)

    # 6b: Fill entities without events (RAG: entity's sentences)
    n_ent_fill = min(len(gaps["entities_no_events"]), fill_max)
    if n_ent_fill > 0:
        print(f"      Filling {n_ent_fill} entity gaps (RAG: entity sentences)...")
        for gap in gaps["entities_no_events"][:n_ent_fill]:
            rag_ctx, rag_sids = retrieve_entity_context(gap["name"], sentences, window=1)
            data, t = fill_entity_events_rag(gap["name"], rag_ctx)
            if data:
                evts = data.get("events", [])
                normalized = []
                for evt in evts:
                    evt["who"] = gap["name"]
                    sid = map_event(evt, sentences)
                    item = norm_event(evt, sid, get_st(sid))
                    if item:
                        normalized.append(item)
                added = add_to_graph(graphs["events"], normalized)
                print(f"        {gap['name']}: +{added} events from {len(rag_sids)} sents ({t:.0f}s)")
                log_samples(f"fill {gap['name']}", normalized, 2)

    # 6c: Auto-register actors into G1
    if gaps["actors_not_entities"]:
        n_reg = len(gaps["actors_not_entities"])
        for gap in gaps["actors_not_entities"]:
            sid = gap["sentences"][0] if gap["sentences"] else 0
            add_to_graph(graphs["entities"], [
                {"sentence_id": sid, "subject": gap["name"], "predicate": "is_a",
                 "object": "UNKNOWN", "qualifier": {"from": "G2_actor"}, "source_text": get_st(sid)}
            ])
        print(f"      Auto-registered {n_reg} actors into G1")

    # ================================================================
    # FINAL METRICS
    # ================================================================
    print("\n    --- Final ---")
    ent_idx = build_entity_index(graphs)
    sent_idx = build_sentence_index(graphs, len(sentences))

    print_graph_summary(graphs)
    total = sum(len(g) for g in graphs.values())
    covered = sum(1 for s, c in sent_idx.items() if sum(c.values()) > 0)
    cov_pct = round(covered / len(sentences) * 100, 1) if sentences else 0
    print(f"      Total: {total}")
    print(f"      Sentence coverage: {covered}/{len(sentences)} ({cov_pct}%)")
    print(f"      Entity index: {len(ent_idx)}")
    print("\n      Final Entity × Graph matrix:")
    print_entity_matrix(ent_idx)

    metrics = {g: len(graphs[g]) for g in GRAPH_NAMES}
    metrics["total"] = total
    return {"metrics_final": metrics, "sentence_coverage_pct": cov_pct,
            "entity_index_size": len(ent_idx), "times": all_times}


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
    print("EXPERIMENT 36 — RAG-Enhanced Five-Graph KG")
    print("=" * 70)
    print("  Model: Gemma 4 31B via Google AI Studio")
    print("  RAG: Micro-extraction + gap fills use retrieved sentences only")
    print("  Discovery: P1-P5 use full source")
    print("  Code: Map, Normalize, Route, Index, Gap Detect")
    print("=" * 70)

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)

    from llm_judge.benchmarks.ragtruth import RAGTruthAdapter
    adapter = RAGTruthAdapter()
    cases = list(adapter.load_cases(max_cases=50))

    source_map = {}
    for case in cases:
        src = "\n".join(case.request.source_context or []) if case.request.source_context else ""
        sh = hash(src)
        if sh not in source_map:
            source_map[sh] = {"source": src, "case_ids": []}
        source_map[sh]["case_ids"].append(case.case_id) # type: ignore[attr-defined] # type: ignore[attr-defined]

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
        for src in sources: # type: ignore[assignment]
            pid = src["case_ids"][0] # type: ignore[index]
            baseline_metrics.append(_bl(bl_data[pid]["passes"]) if pid in bl_data else None)
        print(f"  Baseline: {sum(1 for b in baseline_metrics if b)} sources")

    # Pre-process
    print("\n  Pre-processing...")
    pre_list = []
    for si, src in enumerate(sources): # type: ignore[assignment]
        pre = preprocess_source(src["source"][:6000]) # type: ignore[index]
        pre_list.append(pre)
        print(f"    Source {si}: {pre['n_sentences']} sents, {len(pre['spacy_entities'])} spaCy ents")

    # Run
    t0 = time.time()
    all_results = []

    for si, src in enumerate(sources): # type: ignore[assignment]
        case_ids = src["case_ids"] # type: ignore[index]
        print(f"\n  {'='*60}")
        print(f"  [{si+1}/{len(sources)}] Source for {', '.join(case_ids[:3])}")
        print(f"  {'='*60}")

        result = run_source(src["source"][:6000], si, pre_list[si], # type: ignore[index]
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
    print(f"  {'Exp 36: RAG Five-Graph':<45s} {final_avg['total']:>7.1f} {pct:>6.1f}% "
          f"{final_avg['entities']:>6.1f} {final_avg['events']:>6.1f} {final_avg['relationships']:>6.1f}")
    print(f"  {'-'*76}")

    avg_cov = sum(r["sentence_coverage_pct"] for r in all_results) / len(all_results)
    print(f"\n  Avg sentence coverage: {avg_cov:.1f}%")

    if pct >= 100:
        print(f"\n  ✅ EXCEEDED BASELINE — {pct}%")
    elif pct >= 90:
        print(f"\n  ✅ PASS — {pct}%")
    elif pct >= 80:
        print(f"\n  ⚠️  PARTIAL — {pct}%")
    else:
        print(f"\n  ❌ BELOW — {pct}%")

    output = {
        "experiment": "exp36_rag_five_graph", "model": "gemma-4-31b-it",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "elapsed_s": round(elapsed, 1), "n_sources": len(sources),
        "baseline_avg": bl_avg, "final_avg": final_avg, "vs_baseline_pct": pct,
        "results": all_results,
    }
    out_path = "experiments/exp36_rag_five_graph_results.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
