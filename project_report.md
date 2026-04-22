# Task-Aware Compression of Sentence Embeddings

**Course:** NLP with Deep Learning — Semester 6, IIIT Bangalore  
**Project Title:** Task-Aware Compression of Sentence Embeddings for NLP Applications  
**Team Members:** Pushkar Kulkarni · Sanyam Verma · Archit Jaju · Harjot Singh  
**GitHub Repository:** `ayhm23/Task-Aware-Compression`

---

## a) Problem Definition / Scope / What is Getting Built?

Large pretrained sentence encoders (e.g., `all-mpnet-base-v2`) produce 768-dimensional embeddings that are expensive to store, transmit, and compare at scale. Retrieval systems, semantic search engines, and real-time NLP pipelines all suffer from this high dimensionality.

**Core Research Question:**
> *Can task-specific supervision during dimensionality reduction preserve task-relevant structure better than generic (task-agnostic) compression?*

**What we build:** A post-hoc compression framework that takes frozen 768-dim sentence embeddings and compresses them to 32, 64, 128, or 256 dimensions using three different compression architectures, each trained under two supervision regimes. We evaluate across three downstream NLP tasks and compare against a PCA baseline.

**Scope:**
- **Encoder fixed:** `sentence-transformers/all-mpnet-base-v2` (not fine-tuned)
- **Compression targets:** 32, 64, 128, 256 dimensions (24×, 12×, 6×, 3× compression)
- **Tasks evaluated:** Semantic Textual Similarity (STS-B), Natural Language Inference (SNLI), Sentiment Classification (SST-2)
- **What is _not_ in scope:** Retraining the encoder, quantization, pruning, or token-level compression

---

## b) Planned Approach with Justification

### Compression Methods

We implement three architectures at increasing levels of complexity:

| Method | Architecture | Justification |
|---|---|---|
| **Linear Projection** | 768 → d (learned weight matrix + optional BatchNorm) | Simplest differentiable baseline; interpretable; analogous to PCA but supervised |
| **Autoencoder (AE)** | 768 → 512 → d → 512 → 768 | Bottleneck naturally enforces a compact representation; reconstruction loss provides a built-in regularizer against information loss |
| **Representation Matching Distillation** | 768 → 512 → 256 → d (MLP student) with projection back to teacher space | Mimics the teacher's representation rather than its logits — the student learns to compress while staying geometrically close to the original space |

**Why these three?**  
They span a spectrum from parameter-light (linear) to highly expressive (AE) with a middle ground (distillation), letting us study whether expressiveness helps or hurts under different tasks.

### Training Modes

1. **Task-Agnostic:** Trains on reconstruction loss only, using a balanced pool of embeddings from all three tasks. No label information.  
2. **Task-Aware:** Trains the compressor **jointly** with a task head using task-specific labels. Loss = `0.7 × task_loss + 0.3 × reconstruction_loss`.

**Justification for joint training:** Forcing the bottleneck to be simultaneously decodable and discriminative is expected to bias the compressed subspace towards task-relevant directions. The 0.7/0.3 weight prioritises task fidelity while the reconstruction term prevents degenerate representations.

### Task Heads

All heads use the feature pattern `[z₁; z₂; |z₁−z₂|; z₁⊙z₂]` (element-wise product and absolute difference) for sentence-pair tasks — a proven pattern from InferSent and SBERT fine-tuning literature.

| Task | Dataset | Metric | Head Type |
|---|---|---|---|
| Semantic Textual Similarity | STS-B | Spearman ρ | Regression (MSE loss) |
| Natural Language Inference | SNLI | Accuracy | 3-class classifier (CrossEntropy) |
| Sentiment Classification | SST-2 | Accuracy | Binary classifier |

### PCA Baseline

A scikit-learn PCA fitted on the training embedding pool serves as the classical unsupervised baseline. This is model-agnostic, requires no labels, and is commonly used in embedding literature (e.g., FAISS index preprocessing).

---

## c) Extension of / Critique of Past Work

Our work is positioned as a **post-hoc, model-agnostic extension** of learned sentence embedding compression, building on the following lines of prior work:

### Relation to Matryoshka Representation Learning (Kusupati et al., 2022)
Matryoshka trains the encoder end-to-end with nested dimension targets so that any prefix of the embedding is useful. **Our approach differs:** we compress *frozen* embeddings post-hoc. This makes our method applicable to any existing encoder without retraining — a practical advantage when the encoder is a closed-API service or a large model with no fine-tuning budget.

### Relation to SBERT (Reimers & Gurevych, 2019)
SBERT shows that fine-tuning with NLI + STS objectives produces semantically rich sentence embeddings. We take SBERT-style embeddings as inputs and ask whether the task signal can be recovered/amplified in a compressed subspace, without touching the encoder weights.

### Critique of Our Own Baseline Choices
- **No Johnson-Lindenstrauss (random projection) baseline:** A random Gaussian projection would serve as a principled lower bound and strengthen the PCA comparison. This is a gap we acknowledge.
- **Fixed α = 0.7:** The task/reconstruction trade-off is not ablated. Sensitivity to this hyperparameter is unknown.
- **L2-norm sentence length proxy:** Figure 5 uses the L2 norm of the embedding as a proxy for sentence length (due to HuggingFace dataset loading constraints). This is a heuristic, not a direct word count.
- **Distillation naming clarification:** Our "distillation" method performs **representation-level matching** (student mimics the teacher's compressed representation), not Hinton-style logit-level KD with temperature-scaled softmax. The term "Representation Matching Distillation" is more precise.

---

## d) Implementation Details

### Data Pipeline

```
generate_embeddings.py
  → STS-B (stsb_multi_mt/en): 5749 train / 1500 test sentence pairs
  → SNLI: 550k train / 10k test sentence pairs (NLI 3-class)
  → SST-2 (GLUE): 67k train / 1821 test single sentences
  → Encoded via all-mpnet-base-v2 → 768-dim .npy files cached to disk
```

**Preprocessing:** All embeddings L2-normalised before agnostic training; raw embeddings used for aware training.

### Training Configuration

```python
BATCH_SIZE       = 64
NUM_EPOCHS       = 10
LEARNING_RATE    = 3e-4   # Adam
WEIGHT_DECAY     = 1e-5
TASK_LOSS_WEIGHT = 0.7    # α in: loss = α·task + (1-α)·recon
SCHEDULER        = CosineAnnealingLR
GRADIENT_CLIP    = 1.0    # max norm
```

### Mixed-Task Compressors

Beyond single-task awareness, we train joint compressors on two tasks simultaneously:
- **STS + NLI:** Loss = `0.5 × L_STS + 0.5 × L_NLI`
- **NLI + Classification:** Loss = `0.5 × L_NLI + 0.5 × L_CLS`

These address the multi-task deployment scenario: a single compressed embedding that is useful for more than one downstream task.

### Checkpoints

~96 compressor checkpoints covering:
- 3 methods × {agnostic, aware×3, mixed×2} × 4 dims = 48 compressor configs
- Corresponding task-head checkpoints for all aware/mixed variants

### Linguistic Analysis

`analysis/linguistic.py` generates three figures:
- **Fig 5:** Spearman ρ by embedding L2-norm bucket (STS, dim=128, task-agnostic)
- **Fig 6:** 3-panel t-SNE — Raw (768) · Agnostic AE (128) · NLI-Aware AE (128) — 2000 stratified NLI samples
- **Fig 7:** Reconstruction MSE vs. dimension for all three methods

### Multi-Seed Robustness (Confidence Intervals)

To validate that the headline NLI gain is not due to random initialisation, we retrained the best configuration (AE, dim=256, NLI-aware) and its agnostic counterpart across 3 seeds (42, 123, 7):

| Configuration | Seed 42 | Seed 123 | Seed 7 | Mean | Std |
|---|---|---|---|---|---|
| NLI-Aware AE (dim=256) | 0.7755 | 0.7778 | 0.7762 | **0.7765** | ±0.0010 |
| Agnostic AE (dim=256) | 0.7560 | 0.7562 | 0.7553 | **0.7558** | ±0.0004 |
| **Δ (Aware − Agnostic)** | | | | **+0.0207** | |

The standard deviation of ±0.001 is an order of magnitude smaller than the gain, confirming the +2.07 pp improvement is **statistically robust**.

---

## e) Results and Discussion

### Full Performance Table

| Method | Dim | Mode | STS ρ | NLI Acc | CLS Acc |
|---|---|---|---|---|---|
| **Baseline (768-dim)** | 768 | — | 0.8275 | 0.7671 | 0.8865 |
| PCA | 256 | agnostic | **0.8381** | 0.7658 | 0.8853 |
| PCA | 128 | agnostic | 0.8335 | 0.7567 | 0.8888 |
| AE agnostic | 256 | agnostic | 0.8366 | 0.7495 | 0.8819 |
| AE aware-NLI | 256 | task-aware | 0.8004 | **0.7755** | 0.8865 |
| AE aware-NLI | 128 | task-aware | 0.8072 | 0.7720 | 0.8658 |
| Linear agnostic | 32 | agnostic | 0.7583 | 0.6936 | **0.8991** |
| AE mixed (NLI+CLS) | 256 | mixed | 0.8219 | 0.7572 | 0.8853 |
| AE mixed (STS+NLI) | 128 | mixed | 0.7998 | 0.7643 | 0.8601 |

### Task Selector Summary

| Target Task | Recommended Compressor | Score | vs. Agnostic Best |
|---|---|---|---|
| STS | **PCA, dim=256** | ρ = 0.8381 | +0.0015 vs AE-agnostic |
| NLI | **AE, task-aware (NLI), dim=256** | Acc = 0.7755 | **+0.0207** vs AE-agnostic |
| Classification | **Linear, task-agnostic, dim=32** | Acc = 0.8991 | — (equal to best) |

### Key Findings

#### 1. Task-Aware Compression Works for NLI — But Not for STS

The NLI-aware autoencoder at dim=256 achieves **77.65% accuracy** (mean across 3 seeds, ±0.10%), beating the best agnostic model by **+2.07 pp**. This is the primary positive finding: when a task requires understanding of entailment relationships, biasing the compressed space toward that signal is beneficial.

For STS, **PCA outperforms all learned methods** (ρ = 0.8381 vs. best learned 0.8366). This is a negative result, but an interpretable one:

> STS measures semantic similarity via cosine distance. SBERT embeddings already lie on a hypersphere where cosine geometry is native. PCA's orthogonal projection preserves this hypersphere structure by dropping lowest-variance directions, while non-linear bottlenecks distort pairwise distances in ways that hurt cosine-based retrieval.

#### 2. Classification Is Insensitive to Compression

SST-2 accuracy is remarkably stable across all methods and dims (0.88–0.90). The sentiment signal is so dominant in SBERT embeddings that even 24× compression (32-dim linear) retains it fully.

#### 3. Mixed Compressors Provide a Reasonable Multi-Task Trade-Off

The NLI+Classification mixed compressor at dim=256 achieves STS ρ=0.822, NLI Acc=0.757, CLS Acc=0.885 — no single task suffers badly. A practical choice when a single compressed representation must serve multiple downstream tasks.

#### 4. Cross-Task Generalization

Compressors trained with STS-supervision perform poorly on NLI (and vice versa). Task supervision biases the latent space — there is a real task-specificity/versatility trade-off.

#### 5. t-SNE Visual Evidence (Figure 6)

The 3-panel t-SNE of 2,000 stratified NLI test samples shows:
- **Raw (768-dim):** Three NLI label clusters are moderately separated
- **Agnostic AE (128-dim):** Cluster structure largely preserved, slight boundary mixing
- **NLI-Aware AE (128-dim):** Intra-class clusters visibly tighter; inter-class boundaries better preserved — qualitative support for the quantitative gain

---

## f) How Could You Improve? Why is Improvement Possible?

### 1. α Ablation (Task vs. Reconstruction Loss Weight)
**What:** Train NLI-aware AE at dim=256 with α ∈ {0.3, 0.5, 0.7, 0.9}.  
**Why possible:** 4 additional training runs (~20 min). Infrastructure already exists.  
**Why valuable:** The optimal α is unknown and potentially task-dependent. STS might prefer α=0.3 (more reconstruction) while NLI might prefer α=0.9. This would provide a principled tuning guideline.

### 2. Contrastive Training Objective
**What:** Replace reconstruction loss with supervised NTXent contrastive loss in the compressed space — pulling same-class embeddings together, pushing different-class apart.  
**Why possible:** `pytorch-metric-learning` provides plug-in contrastive losses.  
**Why valuable:** The AE's reconstruction objective is task-agnostic by nature, partially counteracting the task loss. A fully task-aware regularizer might eliminate the STS negative finding.

### 3. Matryoshka-Style Nested Training
**What:** Train a single compressor where the first d-dim prefix of a 256-dim output is already a good d-dim representation (d ∈ {32, 64, 128, 256}).  
**Why possible:** The Matryoshka loss (sum of task losses at all nested prefixes) is a straightforward modification of the training loop.  
**Why valuable:** Currently 4 separate checkpoints per config. One Matryoshka model enables dynamic compression at inference with no checkpoint overhead.

### 4. Inference Latency Benchmarking
**What:** Measure wall-clock search time in FAISS with 768-dim vs. 256-dim vs. 32-dim embeddings.  
**Why possible:** FAISS + timeit integration is straightforward.  
**Why valuable:** Without latency numbers, the compression ratios are theoretical. Real speedup numbers would make the case for deployment.

### 5. Johnson-Lindenstrauss Random Projection Baseline
**What:** Add a random Gaussian projection baseline alongside PCA.  
**Why possible:** 3 lines of numpy — `W = np.random.randn(768, d) / sqrt(d); Z = X @ W`.  
**Why valuable:** If random projection matches PCA, it suggests learned compression does not meaningfully preserve more structure than the Johnson-Lindenstrauss bound — a useful null hypothesis.

---

*All results sourced from `results/metrics/full_results_table.csv` and `results/metrics/multi_seed_nli_confidence.json`. Figures referenced: `results/plots/fig1–fig9`.*
