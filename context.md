# Task-Aware Compression — Project Context

**Course:** NLP with Deep Learning (Sem 6)  
**Team:** Sanyam Verma, Archit Jaju, Pushkar Kulkarni, Harjot Singh

---

## Research Question

Can **task-aware** compression of sentence embeddings preserve task-relevant linguistic information better than **task-agnostic** compression?

**Base Encoder:** `sentence-transformers/all-mpnet-base-v2` → 768-dimensional embeddings  
**Compression Targets:** 32, 64, 128, 256 dimensions (compression ratios: 24×, 12×, 6×, 3×)

---

## Repository Structure

```
Task-Aware-Compression/
├── config.py                  # Central config: paths, hyperparameters, task registry
├── requirements.txt
├── README.md
│
├── compression/               # Core compression methods
│   ├── base.py                # Abstract BaseCompressor class
│   ├── linear.py              # Linear projection (768→d), optional BatchNorm
│   ├── autoencoder.py         # AE (768→512→d→512→768)
│   └── distillation.py        # StudentCompressor MLP (768→512→256→d)
│
├── embeddings/
│   ├── encoder.py             # SentenceEncoder wrapper with caching support
│   └── cache/                 # Pre-computed 768-dim embeddings (.npy), 48 files
│                              # 3 tasks × 3 splits × {s1, s2, labels}
│
├── data/
│   └── dataset_loader.py      # Unified loaders for STS-B, SNLI, SST-2
│                              # SentencePairDataset / SingleSentenceDataset
│
├── models/
│   ├── task_heads.py          # STSHead, NLIHead, ClassificationHead
│   └── checkpoints/           # Saved .pt files (see Checkpoints section)
│
├── evaluation/
│   └── metrics.py             # evaluate_sts(), evaluate_classification(),
│                              # compute_embedding_quality()
│
├── scripts/
│   ├── generate_embeddings.py # Step 1: encode datasets → cache .npy
│   ├── train_compression.py   # Step 2: train all method/mode/dim combos
│   ├── evaluate.py            # Step 3: load checkpoints → compute metrics → JSON
│   └── plot_results.py        # Step 4: generate 4 publication-quality figures
│
├── analysis/                  # Placeholder — empty, intended for linguistic analysis
├── notebooks/                 # Jupyter notebooks for exploration and analysis
└── results/
    ├── metrics/               # 48 JSON files + full_results_table.csv
    └── plots/                 # Generated figures
```

---

## What Has Been Implemented

### 1. Three Compression Methods

| Method | Architecture | Key Detail |
|--------|-------------|------------|
| **Linear** | 768 → d (single layer, optional BN) | Simplest; decoder for reconstruction loss |
| **Autoencoder** | 768 → 512 → d → 512 → 768 | Bottleneck = compressed rep; reconstruction + optional task loss |
| **Representation Matching Distillation** | 768 → 512 → 256 → d (MLP student) | Projection head maps back to teacher space for distillation loss |

### 2. Two Training Modes

- **Task-Agnostic:** Trains on a balanced pool of STS + NLI + SST-2 embeddings using reconstruction loss only. No task labels used.
- **Task-Aware:** Trains compressor + task head jointly using task-specific labels (task loss drives the bottleneck).

### 3. Three Downstream Tasks

| Task | Dataset | Metric | Head |
|------|---------|--------|------|
| Semantic Textual Similarity | STS-B (`stsb_multi_mt/en`) | Spearman ρ | STSHead (regression) |
| Natural Language Inference | SNLI | Accuracy / F1 | NLIHead (3-class) |
| Sentiment Classification | SST-2 (GLUE) | Accuracy / F1 | ClassificationHead (binary) |

### 4. Task Head Design

All heads use the same feature engineering pattern:
- **STS/NLI:** `[z₁; z₂; |z₁−z₂|; z₁⊙z₂]` → hidden layer (ReLU, LayerNorm, Dropout) → output
- **Classification:** `[z]` → hidden layer → output
- **Loss:** MSE for STS, CrossEntropy otherwise

### 5. Complete Pipeline (All Steps Executed)

```
generate_embeddings.py  →  train_compression.py  →  evaluate.py  →  plot_results.py
```

All four steps have been run. Results and checkpoints are fully populated.

---

## Checkpoints

**Location:** `models/checkpoints/`

```
checkpoints/
├── autoencoder/
│   ├── task_agnostic/     dim32.pt, dim64.pt, dim128.pt, dim256.pt
│   ├── task_aware/        {sts,nli,classification}_dim{32,64,128,256}.pt
│   └── mixed_*/           (sts_nli, nli_classification) dim128.pt, dim256.pt
├── autoencoder_head/      task heads for aware models (same structure)
├── distillation/          (same structure)
├── distillation_head/
├── linear/
└── linear_head/
```

**~96 compressor checkpoints** + corresponding head checkpoints for all method/mode/dim combinations.

---

## Results

**Location:** `results/metrics/`

**Naming convention:**  
- Agnostic: `{method}_agnostic_dim{d}.json`  
- Aware: `{method}_aware_{task}_dim{d}.json`  
- Mixed: `{method}_mixed_{task_a}_{task_b}_dim{d}.json`
- PCA Baseline: `pca_dim{d}.json`
- Baseline: `baseline.json` (full 768-dim ceiling)

**Matrix covered:**

| | dim32 | dim64 | dim128 | dim256 |
|--|--|--|--|--|
| linear_agnostic | ✅ | ✅ | ✅ | ✅ |
| autoencoder_agnostic | ✅ | ✅ | ✅ | ✅ |
| distillation_agnostic | ✅ | ✅ | ✅ | ✅ |
| pca | ✅ | ✅ | ✅ | ✅ |
| autoencoder_mixed | ❌ | ❌ | ✅×2 | ✅×2 |
| *_aware_sts | ✅×3 | ✅×3 | ✅×3 | ✅×3 |
| *_aware_nli | ✅×3 | ✅×3 | ✅×3 | ✅×3 |
| *_aware_classification | ✅×3 | ✅×3 | ✅×3 | ✅×3 |

**Total:** 56 JSON files + `full_results_table.csv` + `task_selector_table.csv`

**Metrics tracked:**
- STS: Spearman ρ, p-value
- NLI / Classification: Accuracy, F1 (macro), classification report
- Embedding quality: mean/std cosine similarity, mean/std L2 distance (compression error proxy)

---

## Key Configuration (`config.py`)

```python
ENCODER_MODEL    = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM    = 768
COMPRESSION_DIMS = [32, 64, 128, 256]
BATCH_SIZE       = 64
NUM_EPOCHS       = 10
LEARNING_RATE    = 3e-4
WEIGHT_DECAY     = 1e-5
AE_HIDDEN_DIM    = 512
AE_DROPOUT       = 0.1
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA    = 0.7    # weight: task loss vs distillation loss
```

---

## Current Status

| Component | Status |
|-----------|--------|
| Compression methods (linear, AE, representation matching distillation) | ✅ Complete |
| Dual training modes (agnostic / aware) | ✅ Complete |
| Dataset loading & embedding caching | ✅ Complete |
| Task heads (STS, NLI, classification) | ✅ Complete |
| Evaluation metrics & cross-task testing | ✅ Complete |
| PCA baselines & Mixed Joint Learning | ✅ Complete |
| All checkpoints saved | ✅ Complete |
| All result JSONs generated | ✅ Complete |
| Aggregated results CSV & Task Selector | ✅ Complete |
| Plotting framework | ✅ Complete (9 Figures) |
| `analysis/` module | ✅ Complete |
| Notebooks | ✅ Complete |

---

## What Could Still Be Done

- **Cross-task generalization deep dive:** Qualitative analysis of which tasks share compressible structure
- **Ablation studies:** Dropout, batch norm, activation function, hidden dim choices
- **Inference benchmarking:** Latency and memory footprint comparison
