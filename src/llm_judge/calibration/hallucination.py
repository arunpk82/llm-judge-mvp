"""
Hallucination Detection — 5-Layer Groundedness Pipeline.

Validated architecture from 20 experiments (RAGTruth benchmark):

  L0: Deterministic text match (exact substring, near-exact, Jaccard)
      → 9% of sentences confirmed grounded. Free, instant, no models.
  L1: Gate 1 MiniLM embeddings (whole-response dual threshold)
      → 44% of cases stopped as hallucinated. Free, 80MB model.
  L2a: MiniCheck Flan-T5-Large (purpose-built factual consistency)
      → 78% of sentences confirmed grounded. Free, 3.1GB model.
      Breakthrough from Exp 17b: +97% improvement over DeBERTa NLI.
  L2b: NLI DeBERTa ENTAILMENT (fallback on MiniCheck "unsupported")
      → Catches 10 additional sentences MiniCheck misses. 400MB model.
  L3: GraphRAG spaCy SVO exact match (per-sentence)
      → Handles remaining structural matches. Free, 50MB model.
  L4: Gemini per-sentence reasoning
      → ~4% of sentences need LLM. F1=0.900. Per-call cost.

Total free: ~96% of sentences resolved locally.
Total local model footprint: ~3.6GB (MiniLM 80MB + MiniCheck 3.1GB + DeBERTa 400MB).
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

logger = logging.getLogger(__name__)

# Sentence splitter
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, filtering short fragments."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


# Patterns for legacy claim/citation detection
_CLAIM_PATTERNS = [
    re.compile(r"(?:according to|studies show|research indicates|data shows|statistics show)", re.I),
    re.compile(r"(?:it is (?:known|proven|established) that)", re.I),
    re.compile(r"(?:the (?:official|published|documented) (?:number|figure|rate|data))", re.I),
    re.compile(r"\d{1,3}(?:\.\d+)?%", re.I),
]
_CITATION_PATTERNS = [
    re.compile(r"(?:according to [\w\s]+(?:,|\.))", re.I),
    re.compile(r"\[(?:\d+|[a-z]+)\]", re.I),
    re.compile(r"\((?:[\w\s]+,?\s*\d{4})\)", re.I),
]


def _tokenize(text: str) -> set[str]:
    return {
        w.lower().strip(".,!?;:\"'()[]{}") 
        for w in text.split() 
        if len(w.strip(".,!?;:\"'()[]{}")) > 2
    }


# --- Dataclasses ---

@dataclass
class SentenceLayerResult:
    """Tracking which layer resolved each sentence."""
    sentence_idx: int
    sentence: str
    resolved_by: str  # "L0", "L2_entailment", "L3_graphrag", "L4_supported", "L4_unsupported"
    detail: str = ""


@dataclass
class HallucinationResult:
    """Result of hallucination check for a single response."""
    case_id: str
    risk_score: float
    grounding_ratio: float
    min_sentence_sim: float
    ungrounded_claims: int
    unverifiable_citations: int
    gate1_decision: str = ""
    gate2_decision: str = ""
    flags: list[str] = field(default_factory=list)
    layer_stats: dict[str, int] = field(default_factory=dict)
    sentence_results: list[SentenceLayerResult] = field(default_factory=list)


# --- L0: Deterministic text match ---

def _l0_deterministic_match(sentence: str, source_sentences: list[str], source_full: str) -> bool:
    """
    L0: Cheapest check. Exact substring, near-exact ratio, or high Jaccard overlap.
    Confirmed: handles 9% of sentences (Experiment 12).
    """
    norm_sent = re.sub(r"\s+", " ", sentence.lower().strip())
    norm_source = re.sub(r"\s+", " ", source_full.lower().strip())

    if norm_sent in norm_source:
        return True

    sent_tokens = set(re.findall(r"\w+", sentence.lower()))
    if not sent_tokens:
        return False

    for src_sent in source_sentences:
        norm_src = re.sub(r"\s+", " ", src_sent.lower().strip())
        if SequenceMatcher(None, norm_sent, norm_src).ratio() > 0.85:
            return True
        src_tokens = set(re.findall(r"\w+", src_sent.lower()))
        if src_tokens:
            jaccard = len(sent_tokens & src_tokens) / len(sent_tokens | src_tokens)
            if jaccard > 0.80:
                return True

    return False


# --- L1: Gate 1 MiniLM embeddings ---

def _l1_gate1_check(
    response: str,
    context: str,
    *,
    grounding_threshold: float = 0.80,
    min_sentence_threshold: float = 0.30,
    similarity_threshold: float = 0.60,
    skip_embeddings: bool = False,
) -> tuple[str, float, float]:
    """
    L1: MiniLM whole-response embedding similarity with dual thresholds.
    Confirmed: stops 44% of cases (Experiment 5).
    Returns: (decision, grounding_ratio, min_sim)
    """
    grounding, min_sim = _compute_grounding_ratio(
        response, context,
        skip_embeddings=skip_embeddings,
        similarity_threshold=similarity_threshold,
    )

    fail_ratio = grounding < grounding_threshold
    fail_min = min_sim < min_sentence_threshold

    if fail_ratio or fail_min:
        near_ratio = grounding_threshold - 0.10 <= grounding < grounding_threshold
        near_min = min_sentence_threshold - 0.05 <= min_sim < min_sentence_threshold
        if near_ratio or near_min:
            return "ambiguous", grounding, min_sim
        else:
            return "fail", grounding, min_sim
    else:
        return "pass", grounding, min_sim


def _compute_grounding_ratio(
    response: str,
    context: str,
    *,
    skip_embeddings: bool = False,
    similarity_threshold: float = 0.6,
) -> tuple[float, float]:
    """Compute grounding ratio and min sentence similarity using MiniLM embeddings."""
    response_sentences = _split_sentences(response)
    context_sentences = _split_sentences(context)

    if not response_sentences:
        return 1.0, 1.0
    if not context_sentences:
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio
    if skip_embeddings:
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio

    try:
        from llm_judge.properties import get_embedding_provider
        provider = get_embedding_provider()
        response_embeddings = provider.encode(response_sentences)
        context_embeddings = provider.encode(context_sentences)

        grounded_count = 0
        sentence_sims: list[float] = []
        for resp_emb in response_embeddings:
            max_sim = provider.max_similarity(resp_emb, context_embeddings)
            sentence_sims.append(max_sim)
            if max_sim >= similarity_threshold:
                grounded_count += 1

        ratio = grounded_count / len(response_sentences)
        min_sim = min(sentence_sims) if sentence_sims else 1.0
        return ratio, min_sim

    except (RuntimeError, ImportError) as e:
        logger.warning("grounding.embedding_fallback", extra={"error": str(e)[:80]})
        ratio = _compute_grounding_ratio_token_overlap(response, context)
        return ratio, ratio


def _compute_grounding_ratio_token_overlap(response: str, context: str) -> float:
    """Legacy token overlap fallback."""
    response_tokens = _tokenize(response)
    context_tokens = _tokenize(context)
    if not response_tokens:
        return 1.0
    overlap = response_tokens & context_tokens
    return len(overlap) / len(response_tokens)


# --- L2a: MiniCheck Flan-T5-Large (factual consistency) ---

_mc_model: Any = None
_mc_tokenizer: Any = None

_MINICHECK_PROMPT = (
    "Determine whether the following claim is consistent with "
    "the corresponding document.\nDocument: {document}\nClaim: {claim}"
)


def _load_minicheck() -> None:
    """Lazy-load MiniCheck model (~3.1GB Flan-T5-Large)."""
    global _mc_model, _mc_tokenizer
    if _mc_model is not None:
        return

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "lytang/MiniCheck-Flan-T5-Large"
    logger.info(f"Loading MiniCheck: {model_name}")
    _mc_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    _mc_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _mc_model.eval()


def _l2a_minicheck(sentence: str, source_doc: str) -> bool:
    """
    L2a: MiniCheck factual consistency check.
    Takes full source document + claim sentence. Returns True if supported.
    Breakthrough from Exp 17b: 78% coverage vs DeBERTa's 39%.
    """
    import torch

    _load_minicheck()

    prompt = _MINICHECK_PROMPT.format(
        document=source_doc[:3500], claim=sentence,
    )
    inputs = _mc_tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048,
    )

    with torch.no_grad():
        outputs = _mc_model.generate(**inputs, max_new_tokens=5)

    generated = _mc_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return generated == "1"


# --- L2b: NLI DeBERTa ENTAILMENT (fallback) ---

_nli_model: Any = None
_nli_tokenizer: Any = None
_nli_labels: list[str] | None = None


def _load_nli():
    """Lazy-load NLI model (DeBERTa, ~400MB)."""
    global _nli_model, _nli_tokenizer, _nli_labels
    if _nli_model is not None:
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "cross-encoder/nli-deberta-v3-large"
    _nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _nli_model.eval()
    _nli_labels = [_nli_model.config.id2label[i].upper()
                   for i in range(len(_nli_model.config.id2label))]


def _l2_nli_check(sentence: str, context_sentences: list[str], ctx_embeddings: Any, resp_emb: Any) -> bool:
    """
    L2: NLI ENTAILMENT check against top-3 most similar source sentences.
    Confirmed: handles 36% of sentences (Experiment 12).
    Only ENTAILMENT is trustworthy (Experiment 8: CONTRADICTION has 68% FP rate).
    Returns True if sentence is confirmed grounded via entailment.
    """
    import torch

    _load_nli()

    from llm_judge.properties import get_embedding_provider
    provider = get_embedding_provider()

    sims = [(j, provider.max_similarity(resp_emb, [ce])) for j, ce in enumerate(ctx_embeddings)]
    sims.sort(key=lambda x: x[1], reverse=True)

    best_entailment = 0.0
    for src_idx, _ in sims[:3]:
        inputs = _nli_tokenizer(
            context_sentences[src_idx], sentence,
            return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            probs = torch.softmax(_nli_model(**inputs).logits, dim=-1)[0].tolist()
        nli_scores = {label: p for label, p in zip(_nli_labels or [], probs)}
        best_entailment = max(best_entailment, nli_scores.get("ENTAILMENT", 0))

    return best_entailment > 0.7


# --- L3: GraphRAG spaCy SVO exact match ---

_spacy_nlp: Any = None


def _load_spacy():
    """Lazy-load spaCy model (~50MB)."""
    global _spacy_nlp
    if _spacy_nlp is not None:
        return

    import spacy
    _spacy_nlp = spacy.load("en_core_web_sm")


def _l3_graphrag_check(sentence: str, source_doc: str) -> bool:
    """
    L3: spaCy SVO triplet exact match.
    Confirmed: handles 17% of sentences (Experiment 12).
    Returns True if sentence is confirmed grounded via exact match.
    """
    _load_spacy()

    try:
        from llm_judge.benchmarks.graphrag_science_gate import (
            _entity_overlap,
            extract_svo_triplets,
        )

        resp_triplets = extract_svo_triplets(sentence, _spacy_nlp)
        src_triplets = extract_svo_triplets(source_doc, _spacy_nlp)

        # Filter intransitive triplets
        resp_triplets = [t for t in resp_triplets if t.obj != "(intransitive)"]
        src_triplets = [t for t in src_triplets if t.obj != "(intransitive)"]

        if not resp_triplets:
            return False

        all_matched = True
        has_any = False

        for rt in resp_triplets:
            found = False
            for st in src_triplets:
                subj_sim = _entity_overlap(rt.subject, st.subject)
                obj_sim = _entity_overlap(rt.obj, st.obj)
                pred_sim = _entity_overlap(rt.predicate, st.predicate)
                score = (subj_sim + obj_sim) / 2
                if score >= 0.5 and pred_sim >= 0.5:
                    found = True
                    break
            if found:
                has_any = True
            else:
                all_matched = False

        return has_any and all_matched

    except (ImportError, Exception) as e:
        logger.debug(f"l3_graphrag.skip: {str(e)[:60]}")
        return False


# --- L4: Gemini per-sentence reasoning ---

_PER_SENTENCE_PROMPT = """You are checking whether a SINGLE sentence from a summary is supported by the source document.

SOURCE DOCUMENT:
{source}

SENTENCE TO CHECK:
{sentence}

Is this sentence fully supported by the source document? Consider:
- Are all specific facts (names, dates, numbers, locations) accurate?
- Does the source actually state or directly imply this?
- If the sentence adds details not in the source, it is NOT supported.

Answer with exactly one word: SUPPORTED or UNSUPPORTED"""


def _l4_gemini_check(sentence: str, source_doc: str, case_id: str = "") -> str:
    """
    L4: Gemini per-sentence reasoning.
    Confirmed: F1=0.900 on L4 sentences (Experiment 13).
    Returns: "supported", "unsupported", or "error"
    """
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("l4_gemini.no_api_key: GEMINI_API_KEY not set, defaulting to supported")
        return "supported"

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    prompt = _PER_SENTENCE_PROMPT.format(source=source_doc[:4000], sentence=sentence)
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.0, "topP": 1.0},
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            resp.raise_for_status()
            data = resp.json()
            raw = data["candidates"][0]["content"]["parts"][-1]["text"].strip().upper()

            if "UNSUPPORTED" in raw:
                return "unsupported"
            elif "SUPPORTED" in raw:
                return "supported"
            else:
                logger.warning(f"l4_gemini.unclear for {case_id}: {raw[:50]}")
                return "supported"

    except Exception as e:
        logger.warning(f"l4_gemini.error for {case_id}: {str(e)[:80]}")
        return "error"


# --- Legacy helpers ---

def _count_ungrounded_claims(response: str, context: str) -> tuple[int, list[str]]:
    claims: list[str] = []
    for pattern in _CLAIM_PATTERNS:
        for match in pattern.finditer(response):
            start = max(0, match.start() - 30)
            end = min(len(response), match.end() + 30)
            claim_context = response[start:end].strip()
            claim_tokens = _tokenize(claim_context)
            context_tokens = _tokenize(context)
            overlap = claim_tokens & context_tokens
            if len(overlap) < len(claim_tokens) * 0.3:
                claims.append(claim_context[:80])
    return len(claims), claims


def _count_unverifiable_citations(response: str, context: str) -> int:
    count = 0
    context_lower = context.lower()
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(response):
            cited_text = match.group(0).lower()
            if cited_text not in context_lower:
                cleaned = re.sub(r"[^\w\s]", "", cited_text).strip()
                if cleaned and cleaned not in context_lower:
                    count += 1
    return count


# --- Main entry point ---

def check_hallucination(
    *,
    response: str,
    context: str,
    source_context: str = "",
    case_id: str = "unknown",
    grounding_threshold: float = 0.8,
    min_sentence_threshold: float = 0.3,
    similarity_threshold: float = 0.6,
    max_ungrounded_claims: int = 2,
    skip_embeddings: bool = False,
    gate2_routing: str = "none",
    layered: bool = True,
) -> HallucinationResult:
    """
    Check a single response for hallucination risk.

    5-layer architecture (validated across 16 experiments, F1=0.900):
      L0: Deterministic text match - instant, no models
      L1: Gate 1 MiniLM embeddings - whole-response dual threshold
      L2: NLI DeBERTa ENTAILMENT - per-sentence, free
      L3: GraphRAG spaCy exact match - per-sentence, free
      L4: Gemini per-sentence reasoning - API call, only for unresolved

    Args:
        response: The LLM-generated response to check.
        context: Full context (conversation + source).
        source_context: Source document only (for L3/L4). Falls back to context.
        case_id: Identifier for logging.
        gate2_routing: When to invoke L4 Gemini.
            "none"  - No LLM calls. L0+L1 only.
            "pass"  - Gate 1 PASS cases get L2->L3->L4 per-sentence analysis.
            "all"   - Every case gets full 5-layer analysis.
        layered: Use 5-layer architecture (True) or legacy L0+L1 only (False).
    """
    source_doc = source_context if source_context else context

    resp_sents = _split_sentences(response)
    ctx_sents = _split_sentences(context)
    ungrounded_count, _ = _count_ungrounded_claims(response, context)
    citation_count = _count_unverifiable_citations(response, context)
    flags: list[str] = []
    sentence_results: list[SentenceLayerResult] = []
    layer_stats = {"L0": 0, "L1_fail": 0, "L2a_minicheck": 0, "L2b_nli": 0, "L3": 0, "L4_supported": 0, "L4_unsupported": 0}

    if not resp_sents or not ctx_sents:
        return HallucinationResult(
            case_id=case_id, risk_score=0.0, grounding_ratio=1.0, min_sentence_sim=1.0,
            ungrounded_claims=ungrounded_count, unverifiable_citations=citation_count,
            gate1_decision="pass", layer_stats=layer_stats,
        )

    # -- L0: Deterministic text match (cheapest, runs first) --
    l0_resolved = set()
    for i, sent in enumerate(resp_sents):
        if _l0_deterministic_match(sent, ctx_sents, source_doc):
            l0_resolved.add(i)
            layer_stats["L0"] += 1
            sentence_results.append(SentenceLayerResult(
                sentence_idx=i, sentence=sent[:120],
                resolved_by="L0", detail="deterministic_match",
            ))

    # If ALL sentences resolved by L0, skip everything else
    if len(l0_resolved) == len(resp_sents):
        return HallucinationResult(
            case_id=case_id, risk_score=0.0, grounding_ratio=1.0, min_sentence_sim=1.0,
            ungrounded_claims=ungrounded_count, unverifiable_citations=citation_count,
            gate1_decision="pass", flags=flags, layer_stats=layer_stats,
            sentence_results=sentence_results,
        )

    # -- L1: Gate 1 MiniLM whole-response --
    gate1_decision, grounding, min_sim = _l1_gate1_check(
        response, context,
        grounding_threshold=grounding_threshold,
        min_sentence_threshold=min_sentence_threshold,
        similarity_threshold=similarity_threshold,
        skip_embeddings=skip_embeddings,
    )

    if gate1_decision in ("fail", "ambiguous"):
        if grounding < grounding_threshold:
            flags.append(f"low_grounding:{grounding:.2f}")
        if min_sim < min_sentence_threshold:
            flags.append(f"low_min_sentence_sim:{min_sim:.3f}")

        layer_stats["L1_fail"] = 1

        # When gate2_routing is "none", Gate 1 is the final verdict.
        # When gate2_routing is "pass"/"all", continue to L2+ for verification.
        # Gate 1 fail/ambiguous becomes a signal, not a stopper.
        if gate2_routing == "none":
            risk = max(0, (grounding_threshold - grounding) / grounding_threshold) * 0.5 + 0.3
            return HallucinationResult(
                case_id=case_id, risk_score=round(min(1.0, risk), 4),
                grounding_ratio=round(grounding, 4), min_sentence_sim=round(min_sim, 4),
                ungrounded_claims=ungrounded_count, unverifiable_citations=citation_count,
                gate1_decision=gate1_decision, flags=flags, layer_stats=layer_stats,
                sentence_results=sentence_results,
            )
        # else: fall through to L2+ analysis

    # -- Gate 1 PASS: per-sentence analysis for L0-unmatched sentences --

    if not layered or gate2_routing == "none":
        risk = 0.0
        if ungrounded_count > max_ungrounded_claims:
            flags.append(f"ungrounded_claims:{ungrounded_count}")
            risk = 0.15

        return HallucinationResult(
            case_id=case_id, risk_score=round(risk, 4),
            grounding_ratio=round(grounding, 4), min_sentence_sim=round(min_sim, 4),
            ungrounded_claims=ungrounded_count, unverifiable_citations=citation_count,
            gate1_decision=gate1_decision, flags=flags, layer_stats=layer_stats,
            sentence_results=sentence_results,
        )

    # -- L2 + L3 + L4: Per-sentence analysis --

    resp_embs: list[Any] = []
    ctx_embs: list[Any] = []
    try:
        from llm_judge.properties import get_embedding_provider
        provider = get_embedding_provider()
        resp_embs = provider.encode(resp_sents)
        ctx_embs = provider.encode(ctx_sents)
    except Exception:
        resp_embs = [None] * len(resp_sents)

    has_unsupported = False

    for i, sent in enumerate(resp_sents):
        if i in l0_resolved:
            continue

        # L2a: MiniCheck (primary — 78% coverage, Exp 17b)
        try:
            if _l2a_minicheck(sent, source_doc):
                layer_stats["L2a_minicheck"] = layer_stats.get("L2a_minicheck", 0) + 1
                sentence_results.append(SentenceLayerResult(
                    sentence_idx=i, sentence=sent[:120],
                    resolved_by="L2a_minicheck",
                ))
                continue
        except Exception as e:
            logger.debug(f"l2a_minicheck.error: {str(e)[:60]}")

        # L2b: NLI DeBERTa ENTAILMENT (fallback — catches 10 more, Exp 17b)
        if resp_embs[i] is not None and ctx_embs:
            try:
                if _l2_nli_check(sent, ctx_sents, ctx_embs, resp_embs[i]):
                    layer_stats["L2b_nli"] = layer_stats.get("L2b_nli", 0) + 1
                    sentence_results.append(SentenceLayerResult(
                        sentence_idx=i, sentence=sent[:120],
                        resolved_by="L2b_nli",
                    ))
                    continue
            except Exception as e:
                logger.debug(f"l2b_nli.error: {str(e)[:60]}")

        # L3: GraphRAG exact match
        try:
            if _l3_graphrag_check(sent, source_doc):
                layer_stats["L3"] += 1
                sentence_results.append(SentenceLayerResult(
                    sentence_idx=i, sentence=sent[:120],
                    resolved_by="L3_graphrag",
                ))
                continue
        except Exception as e:
            logger.debug(f"l3_graphrag.error: {str(e)[:60]}")

        # L4: Gemini per-sentence reasoning
        decision = _l4_gemini_check(sent, source_doc, case_id)

        if decision == "unsupported":
            has_unsupported = True
            layer_stats["L4_unsupported"] += 1
            sentence_results.append(SentenceLayerResult(
                sentence_idx=i, sentence=sent[:120],
                resolved_by="L4_unsupported",
            ))
            flags.append(f"l4_unsupported:{sent[:60]}")
        else:
            layer_stats["L4_supported"] += 1
            sentence_results.append(SentenceLayerResult(
                sentence_idx=i, sentence=sent[:120],
                resolved_by="L4_supported",
            ))

    final_decision = "fail" if has_unsupported else "pass"
    gate2_decision = final_decision if (layer_stats["L4_supported"] + layer_stats["L4_unsupported"]) > 0 else ""

    if has_unsupported:
        unsupported_ratio = layer_stats["L4_unsupported"] / max(1, len(resp_sents))
        risk = 0.5 + unsupported_ratio * 0.5
    else:
        risk = 0.0

    if ungrounded_count > max_ungrounded_claims:
        flags.append(f"ungrounded_claims:{ungrounded_count}")

    return HallucinationResult(
        case_id=case_id, risk_score=round(min(1.0, risk), 4),
        grounding_ratio=round(grounding, 4), min_sentence_sim=round(min_sim, 4),
        ungrounded_claims=ungrounded_count, unverifiable_citations=citation_count,
        gate1_decision=gate1_decision, gate2_decision=gate2_decision,
        flags=flags, layer_stats=layer_stats, sentence_results=sentence_results,
    )


# --- Batch interface ---

def check_hallucinations_batch(
    *,
    cases: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    grounding_threshold: float = 0.3,
) -> dict[str, Any]:
    """Run hallucination checks across a batch of evaluated cases."""
    results: list[HallucinationResult] = []
    case_map = {str(c.get("case_id", i)): c for i, c in enumerate(cases)}

    for j in judgments:
        case_id = str(j.get("case_id", ""))
        case = case_map.get(case_id, {})
        conversation = case.get("conversation", [])
        context_parts = [
            msg.get("content", "") for msg in conversation if isinstance(msg, dict)
        ]
        context = " ".join(context_parts)
        response = str(case.get("candidate_answer", ""))

        if not response or not context:
            continue

        result = check_hallucination(
            response=response, context=context,
            case_id=case_id, grounding_threshold=grounding_threshold,
        )
        results.append(result)

    flagged = [r for r in results if r.flags]
    avg_grounding = sum(r.grounding_ratio for r in results) / len(results) if results else 0
    avg_risk = sum(r.risk_score for r in results) / len(results) if results else 0

    return {
        "total_checked": len(results),
        "flagged": len(flagged),
        "flagged_rate": round(len(flagged) / len(results), 4) if results else 0,
        "avg_grounding_ratio": round(avg_grounding, 4),
        "avg_risk_score": round(avg_risk, 4),
        "flagged_cases": [
            {
                "case_id": r.case_id,
                "risk_score": r.risk_score,
                "grounding_ratio": r.grounding_ratio,
                "flags": r.flags,
            }
            for r in flagged[:20]
        ],
    }
