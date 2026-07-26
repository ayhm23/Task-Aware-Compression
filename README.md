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

# Step 3: Evaluate compression models
python scripts/evaluate.py --method linear --mode task_aware --task sts

# Step 4: Run PCA baselines and Task Selector
python scripts/evaluate.py --pca
python scripts/evaluate.py --task_selector

# Step 5: Multi-task Mixed compressor training and evaluation
python scripts/train_compression.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128
python scripts/evaluate.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128

# Step 6: Linguistic Analysis & Exploration
python analysis/linguistic.py
# jupyter notebook notebooks/results_analysis.ipynb

# Step 7: Plot all aggregated results
python scripts/plot_results.py
```

---

## 📊 Tasks Covered

| Task | Dataset | Metric |
|------|---------|--------|
| Semantic Textual Similarity | STS-B | Spearman ρ |
| Natural Language Inference | SNLI | Accuracy |
| Text Classification | SST-2 | Accuracy |

---

## 📐 Compression Methods

| Method | Type |
|--------|------|
| Linear Projection | Task-agnostic / Task-aware |
| Autoencoder | Task-agnostic / Task-aware |
| Representation Matching Distillation | Task-aware |

---

## 💡 Key Results & Limitations

- **Task-Aware Gain (NLI):** The NLI-aware autoencoder at dim=256 reaches **77.65% mean accuracy** across 3 seeds (42, 123, 7; std ±0.10pp), beating the best task-agnostic autoencoder by **+2.07 pp**. The gap holds well outside the seed-to-seed noise, confirming that biasing the compressed space toward entailment structure genuinely helps when the downstream task needs it. This is the project's headline positive finding — see `project_report.md` for the full multi-seed table.
- **Dataset Gap & Negative Finding (STS):** We observed that learned task-aware networks fail to outperform classical unsupervised methods for the Semantic Textual Similarity (STS) task. The task selector reveals that the best overall compressor for STS is **PCA at dim=256** ($\rho$ = 0.8381), which defeats both the best *agnostic autoencoder* ($\rho$ = 0.8366) and the best *STS-aware* models ($\rho$ = 0.8351). **Why?** Since STS measures semantic similarity via cosine distances, any non-linear transformations (like Autoencoders) or aggressive neural bottlenecks distort the relative distance topology of the embeddings more than a simple orthogonal PCA projection. PCA simply drops the least-varying dimensions, preserving the native hypersphere structure that the sentence-transformers rely on.
- **Proxy Sentence Lengths:** Note that Figure 5 estimates sentence length via the $L_2$ norm of pre-computed HuggingFace embeddings offline, as a fallback heuristic to avoid silent failures while loading datasets directly. 

---

## 📝 Citation

If you use this codebase, please cite the course project:
> Kulkarni, Verma, Jaju, Singh (2025). Task-Aware Compression of Sentence Embeddings for NLP Applications.
