"""
Experiment 42: Gemini Extract + Gemma Verify — 100 Cases

Isolates the extraction variable from Exp 41:
  Exp 40: Gemini extract + Gemma verify = 11/11 (isolated, 11 cases)
  Exp 41: Gemma extract  + Gemma verify = 76/94 (isolated, 100 cases)
  Exp 42: Gemini extract + Gemma verify = ? (isolated, 100 cases)

If Exp 42 scores high, the problem is Gemma extraction, not Gemma verification.
If Exp 42 scores low, the problem is at scale regardless of extractor.

Also validates MiniCheck two-zone threshold (≤0.5 flag) on Gemini-extracted
sources at 100 cases.

Usage:
    poetry run python experiments/exp42_gemini_extract_gemma_verify.py          # 100 cases
    poetry run python experiments/exp42_gemini_extract_gemma_verify.py 50       # 50 cases
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMMA_MODEL = "gemma-4-31b-it"
GEMINI_TIMEOUT = 180.0
GEMMA_TIMEOUT = 300.0
MAX_RETRIES = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path("experiments/exp42_results")
NUM_CASES = int(sys.argv[1]) if len(sys.argv) > 1 else 100


# --- Gemini caller (for extraction — supports JSON mode) ---
def call_gemini(prompt: str, *, temperature: float = 0.0) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature, "topP": 1.0,
            "responseMimeType": "application/json",
        },
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=GEMINI_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 10
                print(f"    ⚠ Attempt {attempt} failed ({type(e).__name__}), retry in {wait}s...")
                time.sleep(wait)
            else:
                raise


# --- Gemma caller (for verification — no JSON mode) ---
def call_gemma(prompt: str, *, temperature: float = 0.0) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMMA_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "topP": 1.0},
    }
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=GEMMA_TIMEOUT) as client:
                resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code != 200:
                    print(f"    ⚠ HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return "\n".join(p.get("text", "") for p in parts if p.get("text"))
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 15
                print(f"    ⚠ Attempt {attempt} failed ({type(e).__name__}), retry in {wait}s...")
                time.sleep(wait)
            else:
                raise


# --- Parsers ---
def parse_json_safe(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {"_raw": raw[:500], "_parse_error": True}


def extract_verdict_from_text(raw: str) -> dict:
    upper = raw.upper()
    verdict_match = re.search(r"\b(HALLUCINATED|GROUNDED)\b", raw)
    hall_words = ["HALLUCINATED", "FABRICAT", "CONTRADICT", "NOT SUPPORTED",
                  "NOT GROUNDED", "DOES NOT MATCH", "INCORRECT", "MISATTRIBUT"]
    ground_words = ["GROUNDED", "SUPPORTED BY", "CONSISTENT WITH", "VERIFIED", "CONFIRMED"]
    h = sum(1 for w in hall_words if w in upper)
    g = sum(1 for w in ground_words if w in upper)
    if verdict_match:
        verdict = verdict_match.group(1).upper()
    elif h > g:
        verdict = "HALLUCINATED"
    elif g > h:
        verdict = "GROUNDED"
    else:
        verdict = ""
    return {"verdict": verdict, "confidence": 0.8, "_parsed_from_text": True, "_h": h, "_g": g}


def parse_verification(raw: str) -> dict:
    if not raw or not raw.strip():
        return {"verdict": "", "confidence": 0, "_empty_response": True}
    cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.DOTALL).strip()
    result = parse_json_safe(cleaned)
    if "_parse_error" in result:
        return extract_verdict_from_text(raw)
    return result


# --- MiniCheck ---
_mc_model: Any = None
_mc_tokenizer: Any = None
_MC_PROMPT = (
    "Determine whether the following claim is consistent with "
    "the corresponding document.\nDocument: {document}\nClaim: {claim}"
)

def _load_minicheck():
    global _mc_model, _mc_tokenizer
    if _mc_model is not None:
        return
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"  Loading MiniCheck (~3.1GB)...")
    _mc_tokenizer = AutoTokenizer.from_pretrained("lytang/MiniCheck-Flan-T5-Large", use_fast=False)
    _mc_model = AutoModelForSeq2SeqLM.from_pretrained("lytang/MiniCheck-Flan-T5-Large")
    _mc_model.eval()
    print(f"  MiniCheck loaded.")

def minicheck_verify(sentence: str, source_doc: str) -> dict:
    import torch
    _load_minicheck()
    prompt = _MC_PROMPT.format(document=source_doc, claim=sentence)
    inputs = _mc_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        outputs = _mc_model.generate(
            **inputs, max_new_tokens=5,
            return_dict_in_generate=True, output_scores=True,
        )
    generated = _mc_tokenizer.decode(outputs.sequences[0], skip_special_tokens=True).strip()
    verdict_supported = generated == "1"
    prob_supported = 0.5
    if outputs.scores:
        fl = outputs.scores[0][0]
        t1 = _mc_tokenizer.encode("1", add_special_tokens=False)
        t0 = _mc_tokenizer.encode("0", add_special_tokens=False)
        if t1 and t0:
            l1, l0 = fl[t1[0]].item(), fl[t0[0]].item()
            mx = max(l1, l0)
            e1 = torch.exp(torch.tensor(l1 - mx)).item()
            e0 = torch.exp(torch.tensor(l0 - mx)).item()
            prob_supported = e1 / (e1 + e0)
    verdict = "GROUNDED" if verdict_supported else "HALLUCINATED"
    zone = "FLAG" if prob_supported <= 0.5 else "ESCALATE"
    return {"verdict": verdict, "prob_supported": round(prob_supported, 4),
            "zone": zone, "input_tokens": input_len, "truncated": input_len >= 2048}


# --- Extraction prompts (for Gemini — JSON mode) ---
PROMPT_HYBRID = """You are building a hybrid event-centered knowledge graph from a source document.

SOURCE DOCUMENT:
{source}

INSTRUCTIONS:
Build a graph where each EVENT is a central hub, and typed edges connect it to entities, times, places, and other events.
For each event hub: identify the action, add typed edges (agent_of, target_of, when, where, why, how).
For each entity: track all attributes (name, age, aliases, location) and which events they participate in.

CRITICAL:
- For every edge, note the source sentence.
- Mark any information the source EXPLICITLY says is unknown as "EXPLICITLY_UNKNOWN".
- Do NOT infer or assume — only include what the source states.

Return JSON:
{{
  "event_hubs": [{{"event_id": "EVT_1", "action": "description", "source_sentence": "S0", "edges": [{{"type": "edge_type", "target": "value", "source": "S0"}}]}}],
  "entities": [{{"name": "entity name", "aliases": [], "attributes": {{}}, "participates_in": ["EVT_1"], "source_sentences": ["S0"]}}],
  "inter_event_edges": [{{"from": "EVT_1", "to": "EVT_2", "type": "relationship", "source": "S0"}}],
  "explicit_unknowns": [{{"about": "what is unknown", "unknown_slots": ["when"], "source": "S0"}}]
}}"""

PROMPT_ISOLATED = """You are compiling a source document into small, self-contained stories.

SOURCE DOCUMENT:
{source}

INSTRUCTIONS:
Create small stories (MAXIMUM 5 sentences each) that capture ALL the source's information.
Each story should be: self-contained, factually precise, explicit about unknowns, cover one thread.

CRITICAL RULES:
- Do NOT add any information not in the source
- Do NOT infer dates, ages, or details
- If the source says something is unknown/unclear, EXPLICITLY state that
- Use exact numbers, names, dates from the source

Return JSON:
{{
  "stories": [
    {{
      "thread": "thread name",
      "story": "The complete story text, max 5 sentences.",
      "key_facts": ["fact1", "fact2"],
      "explicit_unknowns": ["what is not stated"]
    }}
  ]
}}"""

PROMPT_OVERLAPPING = """You are compiling a source document into overlapping small stories.

SOURCE DOCUMENT:
{source}

Create small stories (MAXIMUM 5 sentences each) capturing ALL the source's information.
Each story MUST OVERLAP — repeat full entity identity (name, age, location, role) in every
story mentioning them. Shared context appears in multiple stories.
Do NOT add information not in the source. Use exact numbers, names, dates.

Return JSON:
{{
  "stories": [
    {{
      "thread": "thread name",
      "story": "The complete story text, max 5 sentences.",
      "key_facts": ["fact1", "fact2"],
      "explicit_unknowns": ["what is not stated"]
    }}
  ]
}}"""

VERIFY_PROMPT = """Given this SOURCE and CLAIM, output a JSON verdict.

SOURCE:
{compiled_source}

CLAIM: "{claim}"

Rules: GROUNDED = every fact supported. HALLUCINATED = any fact contradicts or fabricates.
Check: numbers, dates, names, roles, who did what.

Output ONLY this JSON:
{{"verdict": "GROUNDED" or "HALLUCINATED", "confidence": 0.0 to 1.0, "reasoning": "why"}}"""


# --- Compilation ---
def compile_hybrid(ext):
    lines = []
    for hub in ext.get("event_hubs", []):
        lines.append(f"\nEVENT: {hub.get('action', '')} ({hub.get('source_sentence', '?')})")
        for edge in hub.get("edges", []):
            lines.append(f"  {edge.get('type', '?')} -> {edge.get('target', '?')}")
    for ent in ext.get("entities", []):
        attrs = ent.get("attributes", {})
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items()) if attrs else ""
        lines.append(f"\nENTITY: {ent.get('name', '')} ({attr_str})")
    for eu in ext.get("explicit_unknowns", []):
        lines.append(f"\nUNKNOWN: {eu.get('about', '')}")
    return "\n".join(lines)

def compile_stories(ext):
    lines = []
    for story in ext.get("stories", []):
        lines.append(f"\n[{story.get('thread', '')}]")
        lines.append(story.get("story", ""))
        if story.get("explicit_unknowns"):
            lines.append(f"NOT STATED: {', '.join(story['explicit_unknowns'])}")
    return "\n".join(lines)


# --- Test data ---
def load_test_data(max_cases: int) -> list[dict]:
    resp_file = REPO_ROOT / "datasets/benchmarks/ragtruth/response.jsonl"
    source_file = REPO_ROOT / "datasets/benchmarks/ragtruth/source_info.jsonl"
    sources = {}
    with open(source_file) as f:
        for line in f:
            item = json.loads(line)
            sources[item["source_id"]] = item.get("source_info", "")
    test_cases = []
    with open(resp_file) as f:
        for line in f:
            if len(test_cases) >= max_cases:
                break
            item = json.loads(line)
            src_id = item["source_id"]
            src_text = sources.get(src_id, "")
            labels = item.get("labels", [])
            response = item.get("response", "")
            if labels:
                for lab in labels:
                    hall_text = lab.get("text", "")
                    hall_sentence = hall_text
                    for sent in re.split(r"(?<=[.!?])\s+", response):
                        if hall_text in sent:
                            hall_sentence = sent.strip()
                            break
                    test_cases.append({
                        "id": f"ragtruth_{item['id']}", "source_id": src_id,
                        "source_text": src_text, "sentence": hall_sentence,
                        "ground_truth": "hallucinated", "hall_type": lab.get("label_type", ""),
                    })
            else:
                first_sent = re.split(r"(?<=[.!?])\s+", response)[0].strip()
                test_cases.append({
                    "id": f"ragtruth_{item['id']}", "source_id": src_id,
                    "source_text": src_text, "sentence": first_sent,
                    "ground_truth": "grounded", "hall_type": "clean",
                })
    return test_cases[:max_cases]


def save_checkpoint(data: dict, name: str):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_data(NUM_CASES)

    n_hall = sum(1 for tc in test_cases if tc["ground_truth"] == "hallucinated")
    n_clean = len(test_cases) - n_hall
    unique_sources = {}
    for tc in test_cases:
        if tc["source_id"] not in unique_sources:
            unique_sources[tc["source_id"]] = tc["source_text"]

    print(f"Exp 42: Gemini extract + Gemma verify (100 cases)")
    print(f"  Test cases: {len(test_cases)} (hall: {n_hall}, clean: {n_clean})")
    print(f"  Unique sources: {len(unique_sources)}")
    print(f"  Extractor: {GEMINI_MODEL}")
    print(f"  Verifier: {GEMMA_MODEL} + MiniCheck")

    extraction_formats = ["hybrid", "isolated", "overlapping"]
    all_formats = ["raw"] + extraction_formats

    # ==== PHASE 1: GEMINI EXTRACTION ====
    print(f"\n{'='*70}")
    print(f"PHASE 1: GEMINI EXTRACTS {len(unique_sources)} SOURCES × {len(extraction_formats)} FORMATS")
    print(f"  Gemini is ~10x faster than Gemma for extraction")
    print(f"{'='*70}")

    prompts = {"hybrid": PROMPT_HYBRID, "isolated": PROMPT_ISOLATED, "overlapping": PROMPT_OVERLAPPING}
    extractions = {}
    total = len(unique_sources) * len(extraction_formats)
    done = 0

    for src_id, src_text in unique_sources.items():
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        numbered = "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))
        extractions[src_id] = {}

        for fmt in extraction_formats:
            done += 1
            cached = OUTPUT_DIR / f"extraction_{src_id}_{fmt}.json"
            if cached.exists():
                with open(cached) as f:
                    ext = json.load(f)
                if "_parse_error" not in ext and "_error" not in ext:
                    extractions[src_id][fmt] = ext
                    items = len(ext.get("event_hubs", ext.get("stories", [])))
                    print(f"  [{done}/{total}] {src_id} {fmt}: cached ✓ ({items} items)")
                    continue

            print(f"  [{done}/{total}] {src_id} {fmt}...", end="", flush=True)
            start = time.time()
            try:
                raw = call_gemini(prompts[fmt].format(source=numbered))
                ext = parse_json_safe(raw)
                elapsed = round(time.time() - start, 1)
                if "_parse_error" in ext:
                    print(f" ⚠ parse failed [{elapsed}s]")
                else:
                    items = len(ext.get("event_hubs", ext.get("stories", [])))
                    print(f" ✓ {items} items [{elapsed}s]")
            except Exception as e:
                elapsed = round(time.time() - start, 1)
                ext = {"_error": str(e)[:200]}
                print(f" ✗ {type(e).__name__} [{elapsed}s]")

            ext["_elapsed"] = elapsed
            extractions[src_id][fmt] = ext
            with open(cached, "w") as f:
                json.dump(ext, f, indent=2)

    # Compile
    compiled_sources = {}
    for src_id, src_text in unique_sources.items():
        sents = re.split(r"(?<=[.!?])\s+", src_text.strip())
        compiled_sources[src_id] = {"raw": "\n".join(f"S{i}: {s}" for i, s in enumerate(sents))}
        for fmt in extraction_formats:
            ext = extractions.get(src_id, {}).get(fmt, {})
            if "_parse_error" in ext or "_error" in ext:
                continue
            if fmt == "hybrid":
                compiled_sources[src_id][fmt] = compile_hybrid(ext)
            else:
                compiled_sources[src_id][fmt] = compile_stories(ext)

    print(f"\n  Compilation summary:")
    for fmt in all_formats:
        available = sum(1 for s in compiled_sources.values() if fmt in s)
        print(f"    {fmt:<12}: {available}/{len(unique_sources)} sources")

    save_checkpoint({"phase": "extraction_done"}, "checkpoint_p1")

    # ==== PHASE 2: MINICHECK ====
    print(f"\n{'='*70}")
    print(f"PHASE 2: MINICHECK TWO-ZONE PRE-FILTER ({len(test_cases)} cases)")
    print(f"  On Gemini-extracted sources (testing threshold validity)")
    print(f"{'='*70}")

    _load_minicheck()
    mc_results = {fmt: [] for fmt in all_formats}
    mc_start = time.time()

    for i, tc in enumerate(test_cases):
        src_id = tc["source_id"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(test_cases)}]")

        for fmt in all_formats:
            if fmt not in compiled_sources.get(src_id, {}):
                mc_results[fmt].append({"_skipped": True, "_test_id": tc["id"]})
                continue
            result = minicheck_verify(tc["sentence"], compiled_sources[src_id][fmt])
            result["_test_id"] = tc["id"]
            result["_expected"] = tc["ground_truth"]
            result["_hall_type"] = tc["hall_type"]
            correct = (
                (result["verdict"] == "HALLUCINATED" and tc["ground_truth"] == "hallucinated")
                or (result["verdict"] == "GROUNDED" and tc["ground_truth"] == "grounded")
            )
            result["_correct"] = correct
            mc_results[fmt].append(result)

    mc_elapsed = round(time.time() - mc_start, 1)
    print(f"  MiniCheck done in {mc_elapsed}s")

    # MiniCheck summary
    print(f"\n  {'Format':<14} {'Flagged':<8} {'Escalated':<10} {'Flag Prec':<10} {'Total':<8}")
    print(f"  {'-'*54}")
    for fmt in all_formats:
        res = [r for r in mc_results[fmt] if not r.get("_skipped")]
        if not res:
            continue
        flagged = [r for r in res if r["zone"] == "FLAG"]
        escalated = [r for r in res if r["zone"] == "ESCALATE"]
        flag_ok = sum(1 for r in flagged if r["_expected"] == "hallucinated")
        prec = flag_ok / len(flagged) if flagged else 1.0
        print(f"  {fmt:<14} {len(flagged):<8} {len(escalated):<10} {prec:<10.1%} {len(res):<8}")

    save_checkpoint({"phase": "minicheck_done", "mc_results": mc_results}, "checkpoint_p2")

    # ==== PHASE 3: GEMMA VERIFICATION ====
    print(f"\n{'='*70}")
    print(f"PHASE 3: GEMMA 4 VERIFICATION ({len(test_cases)} cases × {len(all_formats)} formats)")
    print(f"  On Gemini-extracted sources")
    print(f"{'='*70}")

    gemma_results = {fmt: [] for fmt in all_formats}
    gemma_start = time.time()

    for i, tc in enumerate(test_cases):
        src_id = tc["source_id"]
        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - gemma_start
            rate = (i + 1) / max(elapsed, 1) * 3600
            remaining = (len(test_cases) - i - 1) / max(rate, 1) * 3600
            print(f"  [{i+1}/{len(test_cases)}] ~{remaining/60:.0f} min remaining")

        for fmt in all_formats:
            if fmt not in compiled_sources.get(src_id, {}):
                gemma_results[fmt].append({"_skipped": True, "_test_id": tc["id"]})
                continue

            start = time.time()
            try:
                raw_resp = call_gemma(
                    VERIFY_PROMPT.format(
                        compiled_source=compiled_sources[src_id][fmt],
                        claim=tc["sentence"],
                    )
                )
                elapsed_s = round(time.time() - start, 1)
                result = parse_verification(raw_resp)
            except Exception as e:
                elapsed_s = round(time.time() - start, 1)
                result = {"verdict": "ERROR", "confidence": 0, "_error": True}

            result["_elapsed"] = elapsed_s
            result["_test_id"] = tc["id"]
            result["_expected"] = tc["ground_truth"]
            result["_hall_type"] = tc["hall_type"]

            verdict = result.get("verdict", "").upper()
            correct = (
                (verdict == "HALLUCINATED" and tc["ground_truth"] == "hallucinated")
                or (verdict == "GROUNDED" and tc["ground_truth"] == "grounded")
            )
            result["_correct"] = correct
            gemma_results[fmt].append(result)

        if (i + 1) % 20 == 0:
            save_checkpoint({"phase": f"gemma_{i+1}", "gemma_results": gemma_results}, f"checkpoint_p3_{i+1}")

    gemma_elapsed = round(time.time() - gemma_start, 1)
    print(f"  Gemma verification done in {gemma_elapsed/60:.1f} min")

    # ==== PHASE 4: RESULTS ====
    print(f"\n{'='*70}")
    print(f"PHASE 4: RESULTS — {len(test_cases)} CASES")
    print(f"{'='*70}")

    for label, results_dict in [("MINICHECK (Gemini-extracted)", mc_results),
                                 ("GEMMA 4 31B (Gemini-extracted)", gemma_results)]:
        print(f"\n  {label}:")
        print(f"  {'Format':<14} {'Correct':<10} {'Prec':<8} {'Recall':<8} {'F1':<8} {'FP':<4} {'FN':<4}")
        print(f"  {'-'*56}")
        for fmt in all_formats:
            res = [r for r in results_dict[fmt]
                   if not r.get("_skipped") and not r.get("_error") and not r.get("_empty_response")
                   and r.get("verdict", "").upper() in ("GROUNDED", "HALLUCINATED")]
            if not res:
                continue
            correct = sum(1 for r in res if r["_correct"])
            tp = sum(1 for r in res if r["_expected"] == "hallucinated" and r["verdict"].upper() == "HALLUCINATED")
            fp = sum(1 for r in res if r["_expected"] == "grounded" and r["verdict"].upper() == "HALLUCINATED")
            fn = sum(1 for r in res if r["_expected"] == "hallucinated" and r["verdict"].upper() == "GROUNDED")
            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"  {fmt:<14} {correct}/{len(res):<8} {prec:<8.1%} {rec:<8.1%} {f1:<8.3f} {fp:<4} {fn:<4}")

    # Comparison with Exp 41
    print(f"\n  CROSS-EXPERIMENT COMPARISON (Gemma verify, isolated format):")
    print(f"  {'Experiment':<40} {'Score':<12} {'FP':<6} {'FN':<6}")
    print(f"  {'-'*64}")
    print(f"  {'Exp 40: Gemini ext + Gemma ver (11 cases)':<40} {'11/11':<12} {'0':<6} {'0':<6}")
    print(f"  {'Exp 41: Gemma ext + Gemma ver (100 cases)':<40} {'76/94':<12} {'13':<6} {'5':<6}")
    iso_res = [r for r in gemma_results["isolated"]
               if not r.get("_skipped") and not r.get("_error")
               and r.get("verdict", "").upper() in ("GROUNDED", "HALLUCINATED")]
    if iso_res:
        c = sum(1 for r in iso_res if r["_correct"])
        fp42 = sum(1 for r in iso_res if r["_expected"] == "grounded" and r["verdict"].upper() == "HALLUCINATED")
        fn42 = sum(1 for r in iso_res if r["_expected"] == "hallucinated" and r["verdict"].upper() == "GROUNDED")
        print(f"  {'Exp 42: Gemini ext + Gemma ver (100 cases)':<40} {c}/{len(iso_res):<10} {fp42:<6} {fn42:<6}")

    # Two-zone analysis
    print(f"\n  TWO-ZONE THRESHOLD (on Gemini-extracted sources):")
    for fmt in all_formats:
        mc = [r for r in mc_results[fmt] if not r.get("_skipped")]
        if not mc:
            continue
        flagged = [r for r in mc if r["zone"] == "FLAG"]
        flag_correct = sum(1 for r in flagged if r["_expected"] == "hallucinated")
        flag_wrong = len(flagged) - flag_correct
        print(f"    {fmt}: {len(flagged)} flagged, {flag_correct} correct, {flag_wrong} wrong → "
              f"precision {flag_correct/len(flagged)*100:.1f}%" if flagged else f"    {fmt}: 0 flagged")

    # Save final
    save_checkpoint({
        "experiment": "Exp 42: Gemini extract + Gemma verify",
        "num_cases": len(test_cases),
        "extractor": GEMINI_MODEL,
        "verifier": GEMMA_MODEL,
        "mc_results": mc_results,
        "gemma_results": gemma_results,
    }, "exp42_final_results")

    print(f"\n✓ All results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
