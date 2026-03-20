"""
evaluation/metrics.py
Evaluation utilities for all tasks.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate_sts(pred_scores: np.ndarray, true_scores: np.ndarray) -> dict:
    """
    Evaluate Semantic Textual Similarity using Spearman correlation.

    Args:
        pred_scores : Predicted similarity scores (N,)
        true_scores : Ground-truth similarity scores (N,)

    Returns:
        dict with 'spearman' key
    """
    corr, pval = spearmanr(pred_scores, true_scores)
    return {
        "spearman": round(float(corr), 4),
        "p_value":  round(float(pval), 6),
    }


def evaluate_classification(
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    average: str = "macro",
) -> dict:
    """
    Evaluate classification tasks (NLI, SST-2).

    Returns:
        dict with accuracy and f1
    """
    acc = accuracy_score(true_labels, pred_labels)
    f1  = f1_score(true_labels, pred_labels, average=average, zero_division=0)

    return {
        "accuracy": round(float(acc), 4),
        "f1":       round(float(f1), 4),
    }


def evaluate_task(task: str, predictions: np.ndarray, labels: np.ndarray) -> dict:
    """
    Unified evaluation entry point.

    Args:
        task        : 'sts', 'nli', or 'classification'
        predictions : Model predictions
        labels      : Ground-truth labels

    Returns:
        dict of metric name → value
    """
    if task == "sts":
        return evaluate_sts(predictions, labels)
    elif task in ("nli", "classification"):
        return evaluate_classification(predictions, labels)
    else:
        raise ValueError(f"Unknown task: {task}")


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between paired embeddings.

    Args:
        emb1, emb2 : Arrays of shape (N, D)

    Returns:
        Array of shape (N,) with cosine similarities
    """
    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True) + 1e-9
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True) + 1e-9
    return np.sum((emb1 / norm1) * (emb2 / norm2), axis=1)


def compute_embedding_quality(
    original: np.ndarray,
    compressed: np.ndarray,
) -> dict:
    """
    Intrinsic quality metrics comparing original vs compressed embeddings.

    Metrics:
        - Mean cosine similarity  (structural preservation)
        - Mean L2 distance        (reconstruction error proxy)
    """
    cos_sims = compute_cosine_similarity(original, compressed)
    l2_dists = np.linalg.norm(original - compressed, axis=1)

    return {
        "mean_cosine_similarity": round(float(np.mean(cos_sims)), 4),
        "std_cosine_similarity":  round(float(np.std(cos_sims)),  4),
        "mean_l2_distance":       round(float(np.mean(l2_dists)), 4),
        "std_l2_distance":        round(float(np.std(l2_dists)),  4),
    }


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # STS test
    pred  = np.array([0.8, 0.2, 0.6, 0.9])
    truth = np.array([0.9, 0.1, 0.7, 0.85])
    print("STS:", evaluate_sts(pred, truth))

    # Classification test
    pred_labels  = np.array([0, 1, 2, 1])
    true_labels  = np.array([0, 1, 1, 1])
    print("CLS:", evaluate_classification(pred_labels, true_labels))

    # Embedding quality test
    orig = np.random.randn(100, 768).astype(np.float32)
    comp = orig + 0.01 * np.random.randn(100, 768).astype(np.float32)
    print("Quality:", compute_embedding_quality(orig, comp))
