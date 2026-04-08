# NLI Layer (L2) Improvement Roadmap

## Current state (after Exp 17b)
- **Models**: MiniCheck Flan-T5-Large (~3.1GB) + DeBERTa NLI (~400MB) + MiniLM (~80MB)
- **MiniCheck alone**: 78% L2 coverage (124/160 sentences), replaces MiniLM retrieval step
- **DeBERTa alone**: 39% L2 coverage (63/160 sentences), needs MiniLM top-3 retrieval
- **Ensemble (both)**: 96% L2 coverage (153/160 sentences), catches nearly everything
- **Neither model catches**: 26 sentences (16%) still need L4

## Goal: ACHIEVED and exceeded
- Original target: 36% → 50%+
- Delivered: 36% → 78% (MiniCheck) or 96% (ensemble)
- Remaining goal: close the gap on the final 26 "neither" sentences

## Completed experiments

### Exp 17: AlignScore ❌ ABANDONED
- Pins torch 1.13.1 (2022), breaks sentence-transformers stack
- Learning: unmaintained packages are not viable for production
- AlignScore concept (multi-task alignment) is sound but the implementation is dead

### Exp 17b: MiniCheck ✅ BREAKTHROUGH
- **Result**: L2 coverage 39% → 78% (+97% improvement)
- MiniCheck catches 71 sentences DeBERTa misses (all NLI=0.000, paraphrase blindness)
- DeBERTa catches 10 sentences MiniCheck misses (long, complex entity-rich sentences)
- 53 sentences caught by both (strong consensus)
- **Learning L24**: Purpose-built factual consistency models dramatically outperform generic NLI
- **Learning L25**: MiniCheck takes full document context — no retrieval step needed
- **Architecture**: MiniCheck replaces both MiniLM retrieval AND DeBERTa in a single call

### Exp 20: Threshold sweep ✅ NO IMPACT
- DeBERTa scores are completely bimodal (>0.9 or <0.3)
- Every threshold from 0.3 to 0.85 gives identical results (63 catches)
- **Learning L23**: Threshold tuning is useless for cross-encoder NLI models

## Remaining experiments

### Exp 23: Ensemble strategy (MiniCheck + DeBERTa) ⬜ PENDING
- **What**: Design the optimal combination of both models
- **Why**: MiniCheck catches 124, DeBERTa catches 63, union is 153 (96%)
- **Options**:
  - A) MiniCheck first → DeBERTa fallback on MiniCheck "0" sentences only
  - B) Run both in parallel → OR logic (either says grounded = grounded)
  - C) Weighted confidence: MiniCheck 0.7× + DeBERTa 0.3× (learn weights)
- **Key question**: Do the 10 DeBERTa-only catches include any hallucinations?
  If they're all true positives, the ensemble is safe.
  If some are false positives, DeBERTa is adding noise.
- **Pass**: Ensemble catches ≥150/160 with FP ≤ current rate
- **RAM cost**: +400MB (DeBERTa) on top of 3.1GB (MiniCheck) = 3.5GB total
- **Locally run**: Yes, both models run on CPU

### Exp 24: MiniCheck threshold tuning ⬜ PENDING
- **What**: Sweep MiniCheck confidence threshold (currently using 0.5)
- **Why**: MiniCheck may not be bimodal like DeBERTa — could benefit from tuning
- **Hypothesis**: Threshold 0.3-0.4 might catch more sentences without FP increase
- **Metric**: L2 coverage vs FP rate at each threshold
- **Pass**: Find threshold with better precision/recall tradeoff
- **Zero cost**: Re-read existing exp17b results JSON, recompute at different thresholds

### Exp 25: "Neither" sentence analysis ⬜ PENDING
- **What**: Deep-dive the 26 sentences neither model catches
- **Why**: These are the hardest cases — understanding why helps all layers
- **Questions**:
  - Are they genuinely unsupported (correct to escalate to L4)?
  - Are they complex multi-claim sentences that need decomposition?
  - Are they entity-dense (names, dates, numbers) where both models fail?
- **Pass**: Categorize all 26 and identify which are fixable at L2

### Exp 19: SummaC matrix for DeBERTa ⬜ LOW PRIORITY
- **What**: Check ALL source sentences instead of top-3
- **Why**: Originally high priority, but MiniCheck already handles full document
- **Value**: Only relevant if DeBERTa stays in ensemble — might help it catch more
  of the 71 sentences it currently misses
- **Reassessment**: LOW — MiniCheck already covers this gap

### Exp 21: Claim decomposition ⬜ MEDIUM PRIORITY
- **What**: Split complex sentences into subclaims before checking
- **Why**: May help with the 26 "neither" sentences
- **Prerequisite**: Run Exp 25 first to see if decomposition would actually help
- **Risk**: Adds a decomposition model (T5/LLM) — more RAM, more latency
- **Locally run**: Yes if using T5-small for decomposition (~240MB)

### Exp 22: HHEM (Vectara) ⬜ LOW PRIORITY
- **What**: Test Vectara's hallucination model (DeBERTa fine-tuned)
- **Why**: Could replace generic DeBERTa in ensemble — same size, better trained
- **Reassessment**: LOW — MiniCheck already outperforms. Only worth testing if
  we want a lighter alternative to MiniCheck (400MB vs 3.1GB)
- **Model**: `vectara/hallucination_evaluation_model` (~400MB)

## Recommended execution order
1. **Exp 24** (MiniCheck threshold) — zero cost, re-read existing data
2. **Exp 25** ("neither" analysis) — understand the 26 hard sentences
3. **Exp 23** (ensemble design) — combine MiniCheck + DeBERTa optimally
4. **Exp 21** (claim decomposition) — only if Exp 25 shows it would help
5. **Exp 22** (HHEM) — only if we need a lighter DeBERTa replacement
6. **Exp 19** (SummaC matrix) — deprioritised, MiniCheck covers the gap

## Architecture constraint (updated)
All L2 models must:
- Run locally on CPU (no API dependency)
- Load in < 30 seconds
- Total L2 RAM budget: ~4GB (MiniLM 80MB + MiniCheck 3.1GB + DeBERTa 400MB)
- Return a confidence score for threshold tuning
- Degrade gracefully if any model is unavailable

## Pipeline with MiniCheck (proposed)
```
L0 Deterministic:     9% free  (instant, no models)
L1 Gate 1 (MiniLM):  44% cases stopped  (80MB, always loaded)
L2a MiniCheck:       78% of remaining sentences  (3.1GB, lazy-loaded)
L2b DeBERTa NLI:     fallback on MiniCheck "0"   (400MB, lazy-loaded)
L3 GraphRAG:         handles remaining  (spaCy, lazy-loaded)
L4 Gemini:           ~4% of sentences  (API, last resort)
Total free:          ~96% of sentences resolved locally
```
