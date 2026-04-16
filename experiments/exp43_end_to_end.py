"""
Experiment 43: End-to-End Pipeline Evaluation — 50 Responses, ~306 Sentences

The definitive L3 experiment. Runs the full L1→L2→L3 pipeline on the
RAGTruth-50 benchmark with all fixes and improvements from Exp 38-42.

Improvements applied:
  I1: spaCy sentence splitting (fixes ragtruth_6 J. Paul Getty bug)
  I2: Fact-counting prompt (derived confidence replaces self-reported)
  I3: Multi-pass micro-verification (4 focused passes, flag-wins ensemble)
  I4: Cross-model validation (MiniCheck + Gemma agree before flagging)
  I5: MiniCheck on raw source only (not extracted formats)
  I6: Gemini extraction cached from Exp 42

Phases (sequential, ~7 hours total):
  Phase 0: Load benchmark (spaCy split, ~2 min)
  Phase 1: MiniCheck on raw source (local CPU, ~75 min)
  Phase 2: Gemma fact-counting (API, ~75 min)
  Phase 3: Gemma 4 micro-passes ×4 (API, ~5 hours)
  Phase 4: Ensemble + F1 comparison (local, ~2 min)

Usage:
    poetry run python experiments/exp43_end_to_end.py
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

GEMMA_MODEL = "gemma-4-31b-it"
GEMMA_TIMEOUT = 300.0
MAX_RETRIES = 3
REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = Path("experiments/exp43_results")
EXP42_DIR = Path("experiments/exp42_results")

# Ensure repo root is on sys.path so 'from experiments.X import Y' works
sys.path.insert(0, str(REPO_ROOT))


# ---- API caller ----
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
                    print(f"    [!] HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
            parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            if not parts:
                return ""
            return "\n".join(p.get("text", "") for p in parts if p.get("text"))
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.HTTPStatusError) as e:
            if attempt < MAX_RETRIES:
                wait = attempt * 15
                print(f"    [!] Attempt {attempt} failed ({type(e).__name__}), retry in {wait}s...")
                time.sleep(wait)
            else:
                raise


def parse_json_safe(raw: str) -> dict:
    cleaned = re.sub(r"<\|channel>thought\n.*?<channel\|>", "", raw, flags=re.DOTALL).strip()
    for attempt_text in [cleaned, cleaned]:
        try:
            return json.loads(attempt_text)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {"_raw": raw[:500], "_parse_error": True}


# ---- MiniCheck ----
_mc_model: Any = None
_mc_tokenizer: Any = None

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
    prompt = (
        "Determine whether the following claim is consistent with "
        f"the corresponding document.\nDocument: {source_doc}\nClaim: {sentence}"
    )
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
    return {"prob_supported": round(prob_supported, 4),
            "verdict": "GROUNDED" if verdict_supported else "HALLUCINATED",
            "input_tokens": input_len, "truncated": input_len >= 2048}


# ---- Gemma prompts ----

PROMPT_FACT_COUNT = """Decompose the CLAIM into individual facts, then check each against the SOURCE.

SOURCE:
{source}

CLAIM: "{claim}"

For EACH fact in the claim:
- SUPPORTED: fact is explicitly stated in the source (cite the sentence)
- NOT_FOUND: fact is absent from the source
- CONTRADICTED: fact conflicts with what the source says
- SHIFTED: words used subtly change the meaning vs the source
- INFERRED: fact is a reasonable inference from context (e.g., "last November" → "2015")

Return ONLY this JSON:
{{"facts": [{{"fact": "...", "status": "SUPPORTED|NOT_FOUND|CONTRADICTED|SHIFTED|INFERRED", "source_ref": "which sentence or n/a"}}], "supported": 0, "not_found": 0, "contradicted": 0, "shifted": 0, "inferred": 0, "total": 0, "verdict": "GROUNDED or HALLUCINATED"}}"""

PROMPT_P1_ENTITY = """Check ENTITY-FACT BINDING only. For each fact in the claim, verify it is attributed to the correct entity in the source.

SOURCE:
{source}

CLAIM: "{claim}"

Question: Does the claim attribute any fact to the wrong entity?
Example error: source says Person A's net worth is $5B, claim says Person B's net worth is $5B.

Return ONLY this JSON:
{{"entity_binding_correct": true or false, "issues": [{{"claim_says": "...", "source_says": "...", "problem": "misattribution"}}]}}"""

PROMPT_P2_NUMBERS = """Check NUMBERS, DATES, and QUANTITIES only. For each number/date in the claim, find the exact match in the source.

SOURCE:
{source}

CLAIM: "{claim}"

Rules:
- "last November" in the source does NOT support "November 2015" in the claim — the year must be explicit
- "several" does NOT support a specific number
- Approximate matches are NOT exact matches

Return ONLY this JSON:
{{"numbers_dates_correct": true or false, "checks": [{{"claim_value": "...", "source_value": "...", "match": "exact|approximate|missing|contradicted"}}]}}"""

PROMPT_P3_SEMANTIC = """Check for SEMANTIC SHIFTS only. Compare descriptive phrases in the claim against the source. Do they mean the SAME thing, or has the meaning shifted?

SOURCE:
{source}

CLAIM: "{claim}"

Pay special attention to:
- Emotion words: "found it amusing" vs "couldn't stop laughing" (different: one is a mental state, other is an observable behavior)
- Intensity: "embarrassment" vs "mortified" (different intensity)
- Connotation: "fallen leader" vs "former leader" ("fallen" implies disgrace)
- Causation: "caused by X" vs "attributed to X" (different certainty)

Return ONLY this JSON:
{{"semantic_match": true or false, "shifts": [{{"claim_phrase": "...", "source_phrase": "...", "shift_type": "emotion|intensity|connotation|causation|other"}}]}}"""

PROMPT_P4_ADDITIONS = """Check for ADDED INFORMATION only. Does the claim contain ANY fact, opinion, characterization, or detail that is NOT present in the source?

SOURCE:
{source}

CLAIM: "{claim}"

Flag ANY of these:
- Specific details not in source (dates, names, numbers, locations)
- Editorial characterizations ("a powerful symbol of hope")
- Attributed motives ("in an effort to improve...")
- Causal claims not in source
- Conclusions not explicitly stated

Return ONLY this JSON:
{{"has_additions": true or false, "additions": [{{"added_text": "...", "why_added": "not in source"}}]}}"""


# ---- Checkpointing ----
def save_checkpoint(data: dict, name: str):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Checkpoint saved: {path}")


def load_checkpoint(name: str) -> dict | None:
    path = OUTPUT_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ---- Main ----
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ==== PHASE 0: LOAD BENCHMARK ====
    print(f"\n{'='*70}")
    print("PHASE 0: LOAD RAGTruth-50 BENCHMARK")
    print(f"{'='*70}")

    try:
        from experiments.benchmark_loader import load_ragtruth_50
    except ModuleNotFoundError:
        # Fallback: import directly from file
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "benchmark_loader",
            REPO_ROOT / "experiments" / "benchmark_loader.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        load_ragtruth_50 = mod.load_ragtruth_50

    responses, sentences = load_ragtruth_50()

    n_hall = sum(1 for s in sentences if s["ground_truth"] == "hallucinated")
    n_clean = len(sentences) - n_hall

    print(f"  Responses: {len(responses)}")
    print(f"  Sentences: {len(sentences)} (spaCy split)")
    print(f"  Hallucinated: {n_hall} ({n_hall*100/len(sentences):.1f}%)")
    print(f"  Clean: {n_clean} ({n_clean*100/len(sentences):.1f}%)")

    save_checkpoint({
        "phase": "loaded",
        "num_responses": len(responses),
        "num_sentences": len(sentences),
        "num_hall": n_hall,
        "num_clean": n_clean,
    }, "checkpoint_p0")

    # ==== PHASE 1: MINICHECK ON RAW SOURCE ====
    print(f"\n{'='*70}")
    print(f"PHASE 1: MINICHECK ON RAW SOURCE ({len(sentences)} sentences)")
    print(f"{'='*70}")

    cached_p1 = load_checkpoint("checkpoint_p1")
    if cached_p1 and len(cached_p1.get("mc_results", [])) == len(sentences):
        print("  Loaded from checkpoint.")
        mc_results = cached_p1["mc_results"]
    else:
        _load_minicheck()
        mc_results = []
        start_time = time.time()

        for i, sent in enumerate(sentences):
            if (i + 1) % 20 == 0 or i == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 1) * 60
                print(f"  [{i+1}/{len(sentences)}] ~{rate:.0f} sent/min")

            result = minicheck_verify(sent["sentence"], sent["source_text"])
            result["_idx"] = i
            result["_response_id"] = sent["response_id"]
            result["_expected"] = sent["ground_truth"]
            mc_results.append(result)

        elapsed = round(time.time() - start_time, 1)
        print(f"  MiniCheck done in {elapsed}s")
        save_checkpoint({"phase": "minicheck_done", "mc_results": mc_results}, "checkpoint_p1")

    # MiniCheck summary
    flagged = sum(1 for r in mc_results if r["prob_supported"] <= 0.5)
    escalated = len(mc_results) - flagged
    flag_hall = sum(1 for i, r in enumerate(mc_results)
                    if r["prob_supported"] <= 0.5 and sentences[i]["ground_truth"] == "hallucinated")
    flag_prec = flag_hall / flagged if flagged else 1.0
    print(f"  Flagged (<=0.5): {flagged}, Escalated (>0.5): {escalated}, Flag precision: {flag_prec:.1%}")

    # ==== PHASE 2: GEMMA FACT-COUNTING ====
    print(f"\n{'='*70}")
    print(f"PHASE 2: GEMMA FACT-COUNTING ({len(sentences)} sentences)")
    print(f"{'='*70}")

    cached_p2 = load_checkpoint("checkpoint_p2")
    if cached_p2 and len(cached_p2.get("fc_results", [])) == len(sentences):
        print("  Loaded from checkpoint.")
        fc_results = cached_p2["fc_results"]
    else:
        fc_results = []
        start_time = time.time()

        for i, sent in enumerate(sentences):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                remaining = (len(sentences) - i - 1) * elapsed / max(i + 1, 1)
                print(f"  [{i+1}/{len(sentences)}] ~{remaining/60:.0f} min remaining")

            try:
                raw = call_gemma(PROMPT_FACT_COUNT.format(
                    source=sent["source_text"], claim=sent["sentence"],
                ))
                result = parse_json_safe(raw)
            except Exception as e:
                result = {"_error": str(e)[:200]}

            result["_idx"] = i
            result["_response_id"] = sent["response_id"]
            result["_expected"] = sent["ground_truth"]
            fc_results.append(result)

            if (i + 1) % 50 == 0:
                save_checkpoint({"phase": f"fact_count_{i+1}", "fc_results": fc_results},
                                f"checkpoint_p2_{i+1}")

        elapsed = round(time.time() - start_time, 1)
        print(f"  Fact-counting done in {elapsed/60:.1f} min")
        save_checkpoint({"phase": "fact_count_done", "fc_results": fc_results}, "checkpoint_p2")

    # ==== PHASE 3: GEMMA 4 MICRO-PASSES ====
    print(f"\n{'='*70}")
    print(f"PHASE 3: GEMMA 4 MICRO-PASSES ({len(sentences)} × 4 passes)")
    print(f"{'='*70}")

    passes = {
        "p1_entity": PROMPT_P1_ENTITY,
        "p2_numbers": PROMPT_P2_NUMBERS,
        "p3_semantic": PROMPT_P3_SEMANTIC,
        "p4_additions": PROMPT_P4_ADDITIONS,
    }

    pass_results = {}
    for pass_name, prompt_tmpl in passes.items():
        cached = load_checkpoint(f"checkpoint_p3_{pass_name}")
        if cached and len(cached.get("results", [])) == len(sentences):
            print(f"  {pass_name}: loaded from checkpoint.")
            pass_results[pass_name] = cached["results"]
            continue

        print(f"\n  Starting {pass_name}...")
        results = []
        start_time = time.time()

        for i, sent in enumerate(sentences):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                remaining = (len(sentences) - i - 1) * elapsed / max(i + 1, 1)
                print(f"    [{i+1}/{len(sentences)}] ~{remaining/60:.0f} min remaining")

            try:
                raw = call_gemma(prompt_tmpl.format(
                    source=sent["source_text"], claim=sent["sentence"],
                ))
                result = parse_json_safe(raw)
            except Exception as e:
                result = {"_error": str(e)[:200]}

            result["_idx"] = i
            result["_expected"] = sent["ground_truth"]
            results.append(result)

            if (i + 1) % 50 == 0:
                save_checkpoint({"phase": f"{pass_name}_{i+1}", "results": results},
                                f"checkpoint_p3_{pass_name}_{i+1}")

        elapsed = round(time.time() - start_time, 1)
        print(f"  {pass_name} done in {elapsed/60:.1f} min")
        save_checkpoint({"phase": f"{pass_name}_done", "results": results},
                        f"checkpoint_p3_{pass_name}")
        pass_results[pass_name] = results

    # ==== PHASE 4: ENSEMBLE + F1 ====
    print(f"\n{'='*70}")
    print(f"PHASE 4: ENSEMBLE + F1 COMPARISON")
    print(f"{'='*70}")

    def compute_f1(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec, rec, f1

    # Build per-sentence ensemble view
    ensemble = []
    for i, sent in enumerate(sentences):
        e = {
            "idx": i,
            "sentence": sent["sentence"][:80],
            "response_id": sent["response_id"],
            "expected": sent["ground_truth"],
            "hall_type": sent["hall_type"],
            "mc_prob": mc_results[i]["prob_supported"],
        }

        # Fact-counting
        fc = fc_results[i]
        if not fc.get("_error"):
            e["fc_verdict"] = fc.get("verdict", "").upper()
            total_facts = fc.get("total", 0)
            supported = fc.get("supported", 0)
            e["fc_ratio"] = round(supported / total_facts, 3) if total_facts > 0 else 0.5
            e["fc_contradicted"] = fc.get("contradicted", 0)
            e["fc_shifted"] = fc.get("shifted", 0)
            e["fc_not_found"] = fc.get("not_found", 0)
        else:
            e["fc_verdict"] = "ERROR"
            e["fc_ratio"] = 0.5

        # Micro-passes
        p1 = pass_results.get("p1_entity", [{}]*len(sentences))[i]
        p2 = pass_results.get("p2_numbers", [{}]*len(sentences))[i]
        p3 = pass_results.get("p3_semantic", [{}]*len(sentences))[i]
        p4 = pass_results.get("p4_additions", [{}]*len(sentences))[i]

        e["p1_ok"] = p1.get("entity_binding_correct", True) if not p1.get("_error") else True
        e["p2_ok"] = p2.get("numbers_dates_correct", True) if not p2.get("_error") else True
        e["p3_ok"] = p3.get("semantic_match", True) if not p3.get("_error") else True
        e["p4_ok"] = not p4.get("has_additions", False) if not p4.get("_error") else True

        # Flag-wins: ANY micro-pass fails → HALLUCINATED
        e["any_pass_fails"] = not (e["p1_ok"] and e["p2_ok"] and e["p3_ok"] and e["p4_ok"])
        e["all_pass_ok"] = e["p1_ok"] and e["p2_ok"] and e["p3_ok"] and e["p4_ok"]

        ensemble.append(e)

    # ---- Strategy comparisons ----
    strategies = {
        "MiniCheck alone (<=0.5)": lambda e: e["mc_prob"] <= 0.5,
        "Fact-count verdict": lambda e: e["fc_verdict"] == "HALLUCINATED",
        "Fact-count (any CONTRADICTED/SHIFTED)": lambda e: e.get("fc_contradicted", 0) > 0 or e.get("fc_shifted", 0) > 0,
        "Micro-pass flag-wins (any fail)": lambda e: e["any_pass_fails"],
        "Micro-pass + mc<=0.5": lambda e: e["any_pass_fails"] and e["mc_prob"] <= 0.5,
        "Micro-pass + mc<=0.9": lambda e: e["any_pass_fails"] and e["mc_prob"] <= 0.9,
        "Cross-val (fc=HALL + mc<=0.5)": lambda e: e["fc_verdict"] == "HALLUCINATED" and e["mc_prob"] <= 0.5,
        "Full pipeline (micro OR fc=HALL) + mc<=0.5": lambda e: (e["any_pass_fails"] or e["fc_verdict"] == "HALLUCINATED") and e["mc_prob"] <= 0.5,
    }

    print(f"\n  {'Strategy':<45} {'TP':<5} {'FP':<5} {'FN':<5} {'Prec':<8} {'Rec':<8} {'F1':<8}")
    print(f"  {chr(9472)*82}")

    # Published baselines
    print(f"  {'GPT-4-turbo (published)':<45} {'':5} {'':5} {'':5} {'':8} {'':8} {'0.634':<8}")
    print(f"  {'Llama-2-13B fine-tuned (published)':<45} {'':5} {'':5} {'':5} {'':8} {'':8} {'0.787':<8}")
    print(f"  {chr(9472)*82}")

    for name, fn in strategies.items():
        tp = sum(1 for e in ensemble if fn(e) and e["expected"] == "hallucinated")
        fp = sum(1 for e in ensemble if fn(e) and e["expected"] == "grounded")
        fn_count = sum(1 for e in ensemble if not fn(e) and e["expected"] == "hallucinated")
        prec, rec, f1 = compute_f1(tp, fp, fn_count)
        marker = ""
        if f1 > 0.787:
            marker = " > Llama"
        if fp == 0:
            marker += " [0 FP]"
        print(f"  {name:<45} {tp:<5} {fp:<5} {fn_count:<5} {prec:<8.1%} {rec:<8.1%} {f1:<8.3f}{marker}")

    # ---- Tiered pipeline ----
    print(f"\n  TIERED PIPELINE (FLAG / CLEAR / ESCALATE):")
    tp = fp = fn_count = tn = 0
    escalated = 0
    for e in ensemble:
        mc = e["mc_prob"]
        fc_hall = e["fc_verdict"] == "HALLUCINATED"
        any_fail = e["any_pass_fails"]
        all_ok = e["all_pass_ok"]

        # FLAG tier
        if mc <= 0.5 and (fc_hall or any_fail):
            decision = "HALL"
        # CLEAR tier
        elif mc >= 0.9 and all_ok and not fc_hall:
            decision = "GROUNDED"
        # ESCALATE
        else:
            escalated += 1
            continue

        if decision == "HALL" and e["expected"] == "hallucinated": tp += 1
        elif decision == "HALL" and e["expected"] == "grounded": fp += 1
        elif decision == "GROUNDED" and e["expected"] == "hallucinated": fn_count += 1
        else: tn += 1

    resolved = tp + fp + fn_count + tn
    prec, rec, f1 = compute_f1(tp, fp, fn_count)
    print(f"    Resolved: {resolved}/{len(ensemble)}")
    print(f"    Escalated: {escalated}/{len(ensemble)} ({escalated*100/len(ensemble):.0f}%)")
    print(f"    TP={tp} FP={fp} FN={fn_count} TN={tn}")
    print(f"    Precision={prec:.1%} Recall={rec:.1%} F1={f1:.3f}")

    # ---- Per-type breakdown ----
    print(f"\n  PER HALLUCINATION TYPE (fact-count verdict):")
    from collections import Counter
    type_counts = Counter(e["hall_type"] for e in ensemble if e["expected"] == "hallucinated")
    print(f"  {'Type':<25} {'Total':<7} {'Caught':<8} {'Missed':<8} {'Recall':<8}")
    print(f"  {chr(9472)*55}")
    for htype, count in type_counts.most_common():
        caught = sum(1 for e in ensemble
                     if e["expected"] == "hallucinated" and e["hall_type"] == htype
                     and e["fc_verdict"] == "HALLUCINATED")
        missed = count - caught
        rec = caught / count if count else 0
        print(f"  {htype:<25} {count:<7} {caught:<8} {missed:<8} {rec:<8.1%}")

    # ---- Error analysis on micro-passes ----
    print(f"\n  MICRO-PASS BREAKDOWN:")
    for pname in ["p1_entity", "p2_numbers", "p3_semantic", "p4_additions"]:
        short = pname.replace("p1_", "P1:").replace("p2_", "P2:").replace("p3_", "P3:").replace("p4_", "P4:")
        key = pname.split("_")[0] + "_ok"
        flagged_by = sum(1 for e in ensemble if not e.get(key.replace(pname.split("_")[0], pname[:2]), True))
        # Recompute using actual key names
        if pname == "p1_entity":
            f_by = sum(1 for e in ensemble if not e["p1_ok"])
            f_correct = sum(1 for e in ensemble if not e["p1_ok"] and e["expected"] == "hallucinated")
        elif pname == "p2_numbers":
            f_by = sum(1 for e in ensemble if not e["p2_ok"])
            f_correct = sum(1 for e in ensemble if not e["p2_ok"] and e["expected"] == "hallucinated")
        elif pname == "p3_semantic":
            f_by = sum(1 for e in ensemble if not e["p3_ok"])
            f_correct = sum(1 for e in ensemble if not e["p3_ok"] and e["expected"] == "hallucinated")
        else:
            f_by = sum(1 for e in ensemble if not e["p4_ok"])
            f_correct = sum(1 for e in ensemble if not e["p4_ok"] and e["expected"] == "hallucinated")
        f_wrong = f_by - f_correct
        prec = f_correct / f_by if f_by else 1.0
        print(f"    {short:<15} flagged {f_by}, correct {f_correct}, wrong {f_wrong}, prec {prec:.1%}")

    # Save final results
    save_checkpoint({
        "experiment": "Exp 43: End-to-end pipeline",
        "benchmark": "RAGTruth-50",
        "num_sentences": len(sentences),
        "num_hall": n_hall,
        "num_clean": n_clean,
        "mc_results": mc_results,
        "fc_results": fc_results,
        "pass_results": pass_results,
        "ensemble": ensemble,
    }, "exp43_final_results")

    print(f"\n{'='*70}")
    print(f"All results saved to {OUTPUT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
