# Task-Aware Compression of Sentence Embeddings

**Course Project — NLP with Deep Learning**

**Team:** Sanyam Verma | Archit Jaju | Pushkar Kulkarni | Harjot Singh

---

## 🎯 Project Goal

Explore whether task-aware compression of sentence embeddings can preserve task-relevant linguistic
information better than generic (task-agnostic) compression, and study the resulting trade-offs across
compression ratio, task performance, and cross-task generalization.

---

## 📁 Repository Structure

```
task_aware_compression/
│
├── data/                  # Dataset loading & preprocessing
│   ├── __init__.py
│   └── dataset_loader.py
│
├── embeddings/            # Embedding generation using pretrained encoders
│   ├── __init__.py
│   └── encoder.py
│
├── compression/           # Compression strategies
│   ├── __init__.py
│   ├── linear.py          # Linear projection
│   ├── autoencoder.py     # Autoencoder-based compression
│   └── distillation.py    # Knowledge distillation
│
├── models/                # Task-specific heads / fine-tuning
│   ├── __init__.py
│   └── task_heads.py
│
├── evaluation/            # Evaluation & metrics
│   ├── __init__.py
│   └── metrics.py
│
├── analysis/              # Linguistic analysis
│   ├── __init__.py
│   └── linguistic.py
│
├── scripts/               # Runnable scripts
│   ├── generate_embeddings.py
│   ├── train_compression.py
│   └── evaluate.py
│
├── results/
│   ├── plots/             # Saved figures
│   └── metrics/           # Saved JSON/CSV metrics
│
├── notebooks/             # Jupyter notebooks for exploration
│
├── config.py              # Central config file
├── requirements.txt
└── README.md
```

---

## 🔧 Setup

```bash
pip install -r requirements.txt
```

---

## 🚀 Quickstart

```bash
# Step 1: Generate embeddings
python scripts/generate_embeddings.py

# Step 2: Train compression models
# Example: Task-aware mode for STS using linear projection
python scripts/train_compression.py --method linear --mode task_aware --task sts

# Step 3: Evaluate
python scripts/evaluate.py --method linear --mode task_aware --task sts

# Step 4: Plot results
python scripts/plot_results.py

# Step 5b: PCA baseline + task selector
python scripts/evaluate.py --pca
python scripts/evaluate.py --task_selector

# Step 5c: Mixed compressor (multi-task scenario)
python scripts/train_compression.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128
python scripts/evaluate.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128

# Step 6: Generate all figures including new ones
python scripts/plot_results.py
python analysis/linguistic.py

# Step 5: Linguistic Analysis & Exploration
python analysis/linguistic.py
# jupyter notebook notebooks/results_analysis.ipynb
```

---

## 📊 Tasks Covered

| Task | Dataset | Metric |
|------|---------|--------|
| Semantic Textual Similarity | STS-B | Spearman ρ |
| Natural Language Inference | SNLI / MultiNLI | Accuracy |
| Text Classification | SST-2 | Accuracy |

---

## 📐 Compression Methods

| Method | Type |
|--------|------|
| Linear Projection | Task-agnostic / Task-aware |
| Autoencoder | Task-agnostic / Task-aware |
| Knowledge Distillation | Task-aware |

---

## 💡 Key Results & Limitations

- **Dataset Gap & Negative Finding (STS):** We observed that task-aware compression models never beat task-agnostic baselines for the Semantic Textual Similarity (STS) task. The best overall structure for STS was an *agnostic autoencoder* ($\rho$ = 0.8366), while the best *STS-aware* models (linear: 0.8350, autoencoder: 0.8351) produced strictly worse or identical performance compared to pre-computation. Thus, task-awareness makes zero improvement on STS, which acts as our core negative finding.
- **Proxy Sentence Lengths:** Note that Figure 5 estimates sentence length via the $L_2$ norm of pre-computed HuggingFace embeddings offline, as a fallback heuristic to avoid silent failures while loading datasets directly. 

---

## 📝 Citation

If you use this codebase, please cite the course project:
> Kulkarni, Verma, Jaju, Singh (2025). Task-Aware Compression of Sentence Embeddings for NLP Applications.
