# Industry Benchmark Validation (EPIC 7.16)

## Quick Start

```bash
# Run all available benchmarks
python -m llm_judge.benchmarks.run_all

# Run single benchmark
python -m llm_judge.benchmarks.run_all --benchmark ragtruth

# Quick test (100 cases)
python -m llm_judge.benchmarks.run_all --max-cases 100

# Custom paths
python -m llm_judge.benchmarks.run_all --data-dir datasets/benchmarks --output-dir reports/benchmarks
```

## Dataset Download Instructions

All datasets go into `datasets/benchmarks/<name>/` — files stay unmodified.

### RAGTruth (Cat 1: Properties 1.1–1.3, 1.5) ✅ INTEGRATED
```bash
cd datasets/benchmarks
git clone https://github.com/ParticleMedia/RAGTruth.git ragtruth_repo
cp ragtruth_repo/dataset/response.jsonl ragtruth/
cp ragtruth_repo/dataset/source_info.jsonl ragtruth/
rm -rf ragtruth_repo
```
- **Size**: 17,790 responses (2,700 test)
- **Published baseline**: GPT-4 F1=0.635
- **Our baseline**: F1=0.307 (deterministic methods only)
- **Citation**: Niu et al., ACL 2024

### HaluEval (Cat 1: Properties 1.1, 1.5) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/halueval
# Download from: https://github.com/RUCAIBox/HaluEval
# Files needed: qa_data.json, dialogue_data.json, summarization_data.json, general_data.json
```
- **Size**: 35,000 examples
- **License**: MIT
- **Citation**: Li et al., EMNLP 2023

### IFEval (Cat 4: Properties 4.1, 4.2) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/ifeval
# Download from HuggingFace: https://huggingface.co/datasets/google/IFEval
# Or use: pip install datasets && python -c "from datasets import load_dataset; ds = load_dataset('google/IFEval'); ds['train'].to_json('ifeval.jsonl')"
```
- **Size**: ~500 prompts with verifiable instructions
- **License**: Apache 2.0
- **Citation**: Zhou et al., 2023

### ToxiGen (Cat 3: Property 3.1) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/toxigen
# Download from: https://github.com/microsoft/TOXIGEN
# Or HuggingFace: https://huggingface.co/datasets/toxigen/toxigen-data
```
- **Size**: 274,000 statements
- **License**: MIT
- **Citation**: Hartvigsen et al., 2022

### Jigsaw / Civil Comments (Cat 3: Property 3.1) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/jigsaw
# Download from: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification
# File needed: train.csv (or all_data.csv)
```
- **Size**: ~160,000 comments
- **License**: CC0

### FaithDial (Cat 1: Properties 1.1–1.3) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/faithdial
# Download from HuggingFace: https://huggingface.co/datasets/McGill-NLP/FaithDial
```
- **Size**: ~18,000 dialogue turns
- **License**: MIT
- **Citation**: Dziri et al., 2022

### FEVER (Cat 1: Properties 1.2, 1.3) — ADAPTER NEEDED
```bash
cd datasets/benchmarks/fever
# Download from: https://fever.ai/resources.html
# File needed: paper_test.jsonl
```
- **Size**: 185,000 claims
- **License**: Open

## Property Coverage Matrix

| Property | RAGTruth | HaluEval | FaithDial | FEVER | IFEval | ToxiGen | Jigsaw |
|----------|----------|----------|-----------|-------|--------|---------|--------|
| 1.1 Groundedness | ✅ | ✅ | ✅ | | | | |
| 1.2 Ungrounded Claims | ✅ | | ✅ | ✅ | | | |
| 1.3 Citation Verification | ✅ | | ✅ | ✅ | | | |
| 1.5 Fabrication Detection | ✅ | ✅ | | | | | |
| 3.1 Toxicity & Bias | | | | | | ✅ | ✅ |
| 4.1 Instruction Following | | | | | ✅ | | |
| 4.2 Format & Structure | | | | | ✅ | | |

## Adapter Status

| Benchmark | Adapter | Status |
|-----------|---------|--------|
| RAGTruth | `ragtruth.py` | ✅ Complete + baseline run |
| HaluEval | `halueval.py` | 🔲 Needs adapter |
| IFEval | `ifeval.py` | 🔲 Needs adapter |
| ToxiGen | `toxigen.py` | 🔲 Needs adapter |
| FaithDial | `faithdial.py` | 🔲 Needs adapter |
| Jigsaw | `jigsaw.py` | 🔲 Needs adapter |
| FEVER | `fever.py` | 🔲 Needs adapter |

## Current Baseline Results (RAGTruth)

```
Response-level: P=0.606 R=0.206 F1=0.307 (vs GPT-4 baseline: 0.635)

Per-property:
  1.1 Groundedness:         P=1.000 R=0.001 F1=0.002  ← token overlap useless
  1.2 Ungrounded Claims:    P=0.000 R=0.000 F1=0.000  ← regex patterns never fire
  1.3 Citation Verification: P=0.345 R=0.235 F1=0.279 ← some signal
  1.5 Fabrication Detection: P=0.000 R=0.000 F1=0.000  ← threshold miscalibrated
```

## Architecture

```
datasets/benchmarks/ragtruth/     ← unmodified RAGTruth files (read-only)
datasets/benchmarks/halueval/     ← unmodified HaluEval files
datasets/benchmarks/ifeval/       ← unmodified IFEval files
datasets/benchmarks/toxigen/      ← unmodified ToxiGen files

src/llm_judge/benchmarks/
    __init__.py                   ← BenchmarkAdapter interface, GroundTruth model
    ragtruth.py                   ← RAGTruthAdapter (complete)
    halueval.py                   ← HaluEvalAdapter (to build)
    ifeval.py                     ← IFEvalAdapter (to build)
    toxigen.py                    ← ToxiGenAdapter (to build)
    runner.py                     ← BenchmarkRunner
    metrics.py                    ← BenchmarkMetrics
    report.py                     ← BenchmarkReport
    run_all.py                    ← CLI runner
```
