"""
analysis/linguistic.py
─────────────────────────────────────────────────────────────────────────────
Linguistic Factor Analysis for Task-Aware Compression of Sentence Embeddings.

Generates 3 figures:

  Fig 5 — Sentence Length Analysis
          Grouped bar chart: Spearman ρ on STS test split, broken down by
          sentence length bucket (short/medium/long) for each method at dim=128.

  Fig 6 — t-SNE Comparison
          Side-by-side t-SNE of NLI test embeddings: raw 768-dim vs
          autoencoder-agnostic 128-dim, coloured by NLI label.

  Fig 7 — Compression Error Analysis
          Line chart: mean cosine similarity between original and
          zero-padded compressed embeddings, for each method across dims.

Usage:
  python analysis/linguistic.py
─────────────────────────────────────────────────────────────────────────────
"""

import os
import sys

# Reach project root from inside analysis/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from config import (
    EMBEDDING_DIM, COMPRESSION_DIMS, EMBEDDINGS_DIR, PLOTS_DIR, BASE_DIR,
)
from compression.linear       import LinearCompressor
from compression.autoencoder  import Autoencoder, DeepAutoencoder
from compression.distillation import StudentCompressor

# ─── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9.5,
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
METHODS = ["linear", "autoencoder", "distillation"]
DIMS    = [32, 64, 128, 256]

METHOD_COLOR  = {"linear": "#2196F3", "autoencoder": "#4CAF50", "distillation": "#FF5722"}
METHOD_LABEL  = {"linear": "Linear",  "autoencoder": "Autoencoder", "distillation": "Distillation"}

NLI_COLOR = {0: "#E91E63", 1: "#3F51B5", 2: "#009688"}
NLI_LABEL = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _load_npy(name: str) -> np.ndarray:
    path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
    return np.load(path)


def _get_compressor(method: str, dim: int) -> torch.nn.Module:
    """Instantiate a compressor (same logic as evaluate.py)."""
    if method == "linear":
        return LinearCompressor(input_dim=EMBEDDING_DIM, output_dim=dim)
    elif method == "autoencoder":
        if dim >= 64:
            return Autoencoder(input_dim=EMBEDDING_DIM, output_dim=dim)
        else:
            return DeepAutoencoder(input_dim=EMBEDDING_DIM, output_dim=dim)
    elif method == "distillation":
        return StudentCompressor(input_dim=EMBEDDING_DIM, output_dim=dim)
    raise ValueError(f"Unknown method: {method}")


def _load_compressor(method: str, mode: str, task_or_none, dim: int) -> torch.nn.Module:
    """Load a trained compressor checkpoint."""
    folder = os.path.join(BASE_DIR, "models", "checkpoints", method, mode)
    tag    = f"{task_or_none}_dim{dim}" if task_or_none else f"dim{dim}"
    path   = os.path.join(folder, f"{tag}.pt")

    model = _get_compressor(method, dim)
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def _compress_batch(model: torch.nn.Module, embeddings: np.ndarray,
                    batch_size: int = 512) -> np.ndarray:
    """Run model inference in batches; return compressed numpy array."""
    out = []
    t   = torch.tensor(embeddings, dtype=torch.float32)
    for i in range(0, len(t), batch_size):
        chunk = t[i : i + batch_size]
        out.append(model(chunk).numpy())
    return np.concatenate(out, axis=0)


def _save(fig, filename: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


# ─── Fig 5: Sentence Length Analysis ──────────────────────────────────────────

def _bucket_label(n_words: int) -> str:
    if n_words <= 8:
        return "short"
    elif n_words <= 15:
        return "medium"
    else:
        return "long"


def plot_sentence_length_analysis():
    """
    Compute per-bucket Spearman ρ for the STS task at dim=128
    using task-agnostic compressors for each method.
    """
    print("[Fig 5] Sentence length analysis...")

    # ── Load embeddings & labels ──────────────────────────────────
    s1_emb = _load_npy("sts_test_s1")       # (N, 768)
    labels  = _load_npy("sts_test_labels")  # (N,)

    # ── Load original STS test sentences from HuggingFace ─────────
    try:
        from datasets import load_dataset
        ds        = load_dataset("stsb_multi_mt", "en", split="test")
        sentences = ds["sentence1"]
    except Exception as e:
        print(f"  Warning: could not load HuggingFace dataset ({e}). "
              "Using proxy word counts from embedding norms.")
        sentences = None

    N = len(labels)

    if sentences is not None and len(sentences) == N:
        word_counts = np.array([len(s.split()) for s in sentences])
    else:
        # Fallback: use L2 norm of embedding as a rough proxy for length
        # (longer sentences tend to have higher norm in SBERT)
        norms       = np.linalg.norm(s1_emb, axis=1)
        pcts        = np.percentile(norms, [33, 66])
        word_counts = np.where(norms < pcts[0], 5,        # "short"  ≈ ≤8 words
                      np.where(norms < pcts[1], 11, 20))  # "medium", "long"

    buckets = {
        "short":  np.where(word_counts <= 8)[0],
        "medium": np.where((word_counts > 8) & (word_counts <= 15))[0],
        "long":   np.where(word_counts > 15)[0],
    }
    bucket_names = ["short", "medium", "long"]
    bucket_labels = [f"Low Norm\n(Bottom 33%)\nn={len(buckets['short'])}",
                     f"Medium Norm\n(Middle 33%)\nn={len(buckets['medium'])}",
                     f"High Norm\n(Top 33%)\nn={len(buckets['long'])}"]

    dim = 128
    results = {m: [] for m in METHODS}

    for method in METHODS:
        model = _load_compressor(method, "task_agnostic", None, dim)
        compressed = _compress_batch(model, s1_emb)  # (N, 128)

        for bname in bucket_names:
            idx = buckets[bname]
            if len(idx) < 10:
                results[method].append(np.nan)
                continue

            # For STS we need s2 as well to compute similarity — but the
            # linguistic question is: how faithfully are s1 embeddings
            # preserved per sentence length bucket?
            # We compare cosine similarity between original and compressed
            # (zero-padded to 768) as the "compression fidelity" metric,
            # but for task performance we compute Spearman on the subset.

            # Load s2 for the subset
            s2_emb = _load_npy("sts_test_s2")
            s2_compressed = _compress_batch(model, s2_emb)

            # Compute cosine similarity between compressed s1 and s2
            def _cos_sim(a, b):
                a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
                b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
                return (a * b).sum(axis=1)

            pred_sim = _cos_sim(compressed[idx], s2_compressed[idx])
            spearman, _ = spearmanr(pred_sim, labels[idx])
            results[method].append(float(spearman))

    # ── Plot ──────────────────────────────────────────────────────
    x         = np.arange(len(bucket_names))
    bar_width  = 0.25
    fig, ax   = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        "STS Performance by Embedding L2 Norm (dim=128, Task-Agnostic)\n"
        "Note: L2 Norm is used as a rough offline proxy for sentence complexity/length.",
        fontsize=11, fontweight="bold"
    )

    for i, method in enumerate(METHODS):
        offset = (i - 1) * bar_width
        bars = ax.bar(x + offset, results[method], bar_width,
                      label=METHOD_LABEL[method],
                      color=METHOD_COLOR[method],
                      edgecolor="white", alpha=0.9)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, fontsize=10)
    ax.set_ylabel("Spearman ρ")
    ax.set_xlabel("Embedding L2 Norm Bucket")
    ax.legend(loc="lower right", framealpha=0.85)

    all_vals = [v for m in METHODS for v in results[m] if not np.isnan(v)]
    if all_vals:
        ax.set_ylim(max(0, min(all_vals) - 0.05), min(1.0, max(all_vals) + 0.06))

    fig.tight_layout()
    _save(fig, "fig5_sentence_length_analysis.png")


# ─── Fig 6: t-SNE Visualisation ───────────────────────────────────────────────

def plot_tsne_comparison():
    """
    3-panel t-SNE of NLI test embeddings:
      Left   — raw 768-dim
      Centre — autoencoder agnostic, compressed to 128-dim
      Right  — NLI-aware autoencoder, compressed to 128-dim
    2 000 stratified samples, coloured by NLI label.
    """
    print("[Fig 6] t-SNE comparison...")

    s1_emb = _load_npy("nli_test_s1")       # (N, 768)
    labels  = _load_npy("nli_test_labels").astype(int)   # (N,)

    # ── Stratified sample ─────────────────────────────────────────
    n_sample  = 2000
    rng       = np.random.default_rng(42)
    classes   = np.unique(labels)
    per_class = n_sample // len(classes)
    idx_list  = []
    for c in classes:
        pool = np.where(labels == c)[0]
        chosen = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        idx_list.append(chosen)
    # Fill remainder from first class if rounding left gaps
    idx = np.concatenate(idx_list)
    if len(idx) < n_sample:
        remaining = rng.choice(len(labels), size=n_sample - len(idx), replace=False)
        idx = np.concatenate([idx, remaining])
    idx    = idx[:n_sample]
    rng.shuffle(idx)

    raw_sample    = s1_emb[idx]          # (2000, 768)
    labels_sample = labels[idx]          # (2000,)

    # ── Compress with autoencoder agnostic dim=128 ─────────────────
    model_agnostic = _load_compressor("autoencoder", "task_agnostic", None, 128)
    comp_agnostic  = _compress_batch(model_agnostic, raw_sample)   # (2000, 128)

    # ── Compress with NLI-aware autoencoder dim=128 ────────────────
    model_aware = _load_compressor("autoencoder", "task_aware", "nli", 128)
    comp_aware  = _compress_batch(model_aware, raw_sample)   # (2000, 128)

    # ── t-SNE ─────────────────────────────────────────────────────
    print("  Running t-SNE on raw embeddings...")
    tsne_raw  = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    xy_raw    = tsne_raw.fit_transform(raw_sample.astype(np.float32))

    print("  Running t-SNE on agnostic compressed embeddings...")
    tsne_agn  = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    xy_agn    = tsne_agn.fit_transform(comp_agnostic.astype(np.float32))

    print("  Running t-SNE on NLI-aware compressed embeddings...")
    tsne_aw   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    xy_aw     = tsne_aw.fit_transform(comp_aware.astype(np.float32))

    # ── Plot ──────────────────────────────────────────────────────
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle("t-SNE of NLI Test Embeddings (2 000 stratified samples)",
                 fontsize=13, fontweight="bold")

    scatter_kw = dict(s=8, alpha=0.65, linewidths=0)

    for ax, xy, title in [(ax1, xy_raw,  "Raw Embeddings (768-dim)"),
                           (ax2, xy_agn,  "Autoencoder Agnostic (128-dim)"),
                           (ax3, xy_aw,   "NLI-Aware Autoencoder (128-dim)")]:
        for label_id in [0, 1, 2]:
            mask = labels_sample == label_id
            ax.scatter(xy[mask, 0], xy[mask, 1],
                       c=NLI_COLOR[label_id],
                       label=NLI_LABEL[label_id],
                       **scatter_kw)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("t-SNE dim 1")
        ax.set_ylabel("t-SNE dim 2")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.legend(loc="best", framealpha=0.85, markerscale=2.5)

    fig.tight_layout()
    _save(fig, "fig6_tsne_comparison.png")


# ─── Fig 7: Compression Error Analysis ────────────────────────────────────────

def plot_compression_error():
    print("[Fig 7] Compression error analysis (reconstruction MSE)...")
    s1_emb = _load_npy("sts_test_s1")

    results = {m: [] for m in METHODS}

    for method in METHODS:
        for dim in DIMS:
            model = _load_compressor(method, "task_agnostic", None, dim)
            t = torch.tensor(s1_emb, dtype=torch.float32)

            with torch.no_grad():
                z = model(t)                       # compress
                if hasattr(model, "reconstruct"):
                    x_hat = model.reconstruct(z)   # AE / Linear decoder
                elif hasattr(model, "project_to_teacher_space"):
                    x_hat = model.project_to_teacher_space(z)  # distillation
                else:
                    x_hat = t  # fallback

                mse = float(((t - x_hat) ** 2).mean())
            results[method].append(mse)

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        "Reconstruction MSE: Original vs Decoded  (Task-Agnostic, STS Test Set)",
        fontsize=12, fontweight="bold"
    )
    markers = {"linear": "o", "autoencoder": "s", "distillation": "^"}
    for method in METHODS:
        ax.plot(DIMS, results[method],
                color=METHOD_COLOR[method], marker=markers[method],
                linewidth=2, markersize=8, label=METHOD_LABEL[method])
        for dim, val in zip(DIMS, results[method]):
            ax.annotate(f"{val:.4f}", (dim, val),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=8, color=METHOD_COLOR[method])

    ax.set_xlabel("Compressed Dimension")
    ax.set_ylabel("Mean Squared Error (lower = better)")
    ax.set_xticks(DIMS)
    ax.set_xticklabels([f"{d}\n({768//d}× ratio)" for d in DIMS])
    ax.legend(loc="upper right", framealpha=0.85)
    fig.tight_layout()
    _save(fig, "fig7_compression_error.png")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("[Linguistic Analysis] Starting...")
    print(f"  Device : {DEVICE}")
    print(f"  Output : {PLOTS_DIR}\n")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_sentence_length_analysis()
    plot_tsne_comparison()
    plot_compression_error()

    print("\n[Done] Figures 5, 6, 7 saved.")


if __name__ == "__main__":
    main()
