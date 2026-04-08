# NLI Layer (L2) Improvement Roadmap

## Current state (Experiment 12 baseline)
- **Model**: cross-encoder/nli-deberta-v3-large (~400MB)
- **Approach**: Per-sentence, check top-3 source sentences by MiniLM cosine similarity
- **Decision**: ENTAILMENT > 0.7 → confirmed grounded (CONTRADICTION ignored, 68% FP rate)
- **Coverage**: 36% of sentences resolved at L2 (63/175 in funnel analysis)
- **Gap**: 39% of sentences still reach L4 (Gemini API calls)

## Goal
Increase L2 coverage from 36% → 50%+ without increasing false positives.
Every 1% improvement at L2 = fewer Gemini API calls at L4.

## Experiments

### Exp 17: AlignScore Science Gate ⬜ PENDING
- **What**: Replace DeBERTa NLI with AlignScore (RoBERTa trained on 4.7M examples across 7 tasks)
- **Why**: AlignScore understands paraphrasing, QA, fact verification — not just textual entailment
- **Hypothesis**: AlignScore catches synonym/paraphrase cases DeBERTa misses ("close" vs "shut")
- **Metric**: L2 coverage % on same 175 sentences from funnel analysis
- **Pass**: L2 > 40% with FP ≤ current rate
- **Model**: `AlignScore-large` (~1.4GB, RoBERTa-large backbone)
- **Install**: `pip install alignscore`

### Exp 18: MiniCheck Science Gate ⬜ PENDING
- **What**: Replace DeBERTa NLI with MiniCheck (Flan-T5 fine-tuned on synthetic hallucination data)
- **Why**: Purpose-built for factual consistency, achieves near GPT-4 accuracy
- **Hypothesis**: MiniCheck better at numerical claims, entity relationships
- **Metric**: Same funnel analysis comparison
- **Pass**: L2 > 40% with FP ≤ current rate
- **Model**: `lytang/MiniCheck-Flan-T5-Large` (~1GB)
- **Install**: `pip install minicheck`

### Exp 19: SummaC-style full matrix ⬜ PENDING
- **What**: Check ALL source sentences instead of top-3 by cosine similarity
- **Why**: MiniLM cosine similarity ≠ NLI entailment. Best entailment source might not be top-3 cosine.
- **Hypothesis**: Full matrix catches cases where the supporting source sentence is semantically distant
- **Metric**: Count of sentences where max-entailment source ≠ top-3 cosine source
- **Pass**: L2 improvement > 3% (justifies the extra NLI calls)
- **Cost**: N source sentences × NLI calls instead of 3. For typical 10-sentence source: 3.3x more NLI calls.
- **Model**: Same DeBERTa — only aggregation changes

### Exp 20: Threshold sweep on DeBERTa ⬜ PENDING
- **What**: Sweep entailment threshold from 0.5 to 0.9 on RAGTruth benchmark
- **Why**: We use 0.7 but haven't validated this is optimal for CS domain
- **Hypothesis**: Lower threshold (0.5-0.6) catches more paraphrases without increasing FP
- **Metric**: L2 coverage vs FP rate at each threshold
- **Pass**: Find threshold that improves L2 by ≥ 3% with FP increase ≤ 1%
- **Model**: Same DeBERTa — only threshold changes

### Exp 21: Claim decomposition + NLI ⬜ PENDING
- **What**: Split complex sentences into subclaims before NLI check
- **Why**: "She was arrested on March 26 and could face 15 years" has 2 claims, one might be supported
- **Hypothesis**: Finer-grained claims get higher entailment scores
- **Metric**: Per-subclaim entailment rate vs per-sentence rate
- **Pass**: Qualitative — does decomposition reveal hidden entailments?
- **Model**: T5 or LLM for decomposition + DeBERTa for NLI
- **Risk**: Decomposition quality affects everything downstream

### Exp 22: HHEM (Vectara) ⬜ PENDING
- **What**: Test Vectara's Hallucination Evaluation Model (DeBERTa fine-tuned for hallucination)
- **Why**: Purpose-built, production-deployed at Vectara for RAG evaluation
- **Metric**: Same funnel comparison
- **Model**: `vectara/hallucination_evaluation_model` (~400MB)

## Execution order
1. **Exp 20** (threshold sweep) — zero cost, may find quick wins
2. **Exp 17** (AlignScore) — most promising, multi-task training
3. **Exp 18** (MiniCheck) — if AlignScore doesn't deliver
4. **Exp 19** (SummaC matrix) — if model swap doesn't help, try better aggregation
5. **Exp 21** (claim decomposition) — most complex, do last
6. **Exp 22** (HHEM) — optional, similar to AlignScore

## Decision criteria
After each experiment, update the funnel analysis:
```
Current:  L0=9% L2=36% L3=17% L4=39% (free=61%)
Target:   L0=9% L2=50%+ L3=17% L4=25%- (free=75%+)
```

If L2 reaches 50%+, L4 drops to ~25% — a 36% reduction in Gemini API calls.

## Architecture constraint
Any L2 replacement must:
- Run locally (no API dependency)
- Load in < 30 seconds
- Process sentences in < 100ms each
- Fit in ~2GB RAM alongside MiniLM (80MB)
- Return a confidence score for threshold tuning
