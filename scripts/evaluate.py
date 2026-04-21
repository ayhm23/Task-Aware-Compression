"""
scripts/evaluate.py
Full evaluation sweep for all compression methods, modes, tasks and dims.

Modes:
  baseline       : 768-dim ceiling — no compression, task head trained on raw embeddings
  task_agnostic  : compressor trained on reconstruction only (no task signal)
  task_aware     : compressor + task head trained jointly
  pca            : sklearn PCA as a non-learned compression baseline
  mixed          : compressor jointly trained on two tasks

Usage:
  python scripts/evaluate.py --baseline
  python scripts/evaluate.py --method linear --mode task_agnostic --dim 128
  python scripts/evaluate.py --method linear --mode task_aware --train_task sts --dim 128
  python scripts/evaluate.py --pca --pca_dim 128
  python scripts/evaluate.py --task_selector
  python scripts/evaluate.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128
  python scripts/evaluate.py --all
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (EMBEDDING_DIM, COMPRESSION_DIMS, TASKS,
                    BATCH_SIZE, SEED, EMBEDDINGS_DIR, METRICS_DIR)
from compression.linear       import LinearCompressor
from compression.autoencoder  import Autoencoder, DeepAutoencoder
from compression.distillation import StudentCompressor
from models.task_heads         import get_task_head
from evaluation.metrics        import evaluate_task, compute_embedding_quality

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Evaluator] Device: {DEVICE}")


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def ckpt_path(method, mode, task, dim):
    folder = os.path.join("models", "checkpoints", method, mode)
    tag = f"{task}_dim{dim}" if task else f"dim{dim}"
    return os.path.join(folder, f"{tag}.pt")


def mixed_ckpt_path(method, task_a, task_b, dim):
    folder = os.path.join("models", "checkpoints", method, f"mixed_{task_a}_{task_b}")
    return os.path.join(folder, f"dim{dim}.pt")


# ── Load cached test embeddings ───────────────────────────────────────────────

# GLUE SST-2 public test labels are all -1 (held out); use validation instead.
_EVAL_SPLIT = {
    "sts":            "test",
    "nli":            "test",
    "classification": "validation",
}


def load_test_embeddings(task):
    """
    Load pre-computed evaluation split embeddings from cache.
    Uses 'test' for STS/NLI and 'validation' for classification
    (GLUE SST-2 test labels are withheld).

    Returns:
        s1     : (N, 768) tensor
        s2     : (N, 768) tensor or None
        labels : (N,) tensor
    """
    def _load(name):
        path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
        return torch.tensor(np.load(path), dtype=torch.float32)

    split  = _EVAL_SPLIT[task]
    labels = _load(f"{task}_{split}_labels")
    if task in ("sts", "nli"):
        return _load(f"{task}_{split}_s1"), _load(f"{task}_{split}_s2"), labels
    else:
        return _load(f"{task}_{split}_s1"), None, labels


# ── Compressor factory ────────────────────────────────────────────────────────

def get_compressor(method, dim):
    if method == "linear":
        return LinearCompressor(input_dim=EMBEDDING_DIM, output_dim=dim)
    elif method == "autoencoder":
        return Autoencoder(input_dim=EMBEDDING_DIM, output_dim=dim) \
               if dim >= 64 else \
               DeepAutoencoder(input_dim=EMBEDDING_DIM, output_dim=dim)
    elif method == "distillation":
        return StudentCompressor(input_dim=EMBEDDING_DIM, output_dim=dim)
    raise ValueError(f"Unknown method: {method}")


# ── PCA Wrapper (thin nn.Module so it plugs into existing helpers) ─────────────

class PCAWrapper(nn.Module):
    """
    Wraps a fitted sklearn PCA model as an nn.Module so it can be used
    as a drop-in replacement for learned compressors inside _train_eval_head
    and run_task_eval.  No parameters are trained — it is always in eval mode.
    """
    def __init__(self, pca: PCA):
        super().__init__()
        self.pca    = pca
        self.output_dim = pca.n_components_
        # Store PCA arrays as buffers (non-trainable, on correct device)
        components = torch.tensor(pca.components_, dtype=torch.float32)  # (d, 768)
        mean       = torch.tensor(pca.mean_,       dtype=torch.float32)  # (768,)
        self.register_buffer("components", components)
        self.register_buffer("pca_mean",   mean)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project x (N, 768) → (N, d) using stored PCA basis."""
        return (x - self.pca_mean) @ self.components.T

    # reconstruction_loss is never called on PCAWrapper, but provide a stub
    # so _train_eval_head doesn't crash if it's ever invoked.
    def reconstruction_loss(self, x):
        z   = self.forward(x)
        rec = z @ self.components + self.pca_mean
        return F.mse_loss(rec, x)


# ── Baseline: train a fresh task head on raw 768-dim embeddings ───────────────

def train_baseline_head(task, epochs=5):
    """
    Train a task head directly on 768-dim embeddings (no compression).
    This establishes the performance ceiling.
    """
    print(f"  [Baseline] Training head for task={task} on 768-dim embeddings...")

    def _load(split, name):
        path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
        return torch.tensor(np.load(path), dtype=torch.float32)

    # Build train loader
    if task in ("sts", "nli"):
        s1     = _load("train", f"{task}_train_s1")
        s2     = _load("train", f"{task}_train_s2")
        labels = _load("train", f"{task}_train_labels")
        loader = DataLoader(TensorDataset(s1, s2, labels),
                            batch_size=BATCH_SIZE, shuffle=True)
    else:
        s1     = _load("train", f"{task}_train_s1")
        labels = _load("train", f"{task}_train_labels")
        loader = DataLoader(TensorDataset(s1, labels),
                            batch_size=BATCH_SIZE, shuffle=True)

    head      = get_task_head(task, compressed_dim=EMBEDDING_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)

    for epoch in range(1, epochs + 1):
        head.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            if task in ("sts", "nli"):
                z1, z2, labs = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
                loss = head.loss(z1, z2, labs)
            else:
                z1, labs = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = head.loss(z1, labs)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"    epoch {epoch}/{epochs}  loss={total/len(loader):.4f}")

    return head


def evaluate_baseline():
    """
    Evaluate the 768-dim baseline on all tasks.
    Trains a fresh task head (no compression) then evaluates on test set.
    """
    print("\n" + "="*60)
    print("  BASELINE EVALUATION (768-dim, no compression)")
    print("="*60)

    results = {"mode": "baseline", "dim": EMBEDDING_DIM}

    for task in TASKS:
        head = train_baseline_head(task)
        metrics = run_task_eval(task, compressor=None, task_head=head,
                                dim=EMBEDDING_DIM)
        results[task] = metrics
        _print_task_metrics(task, metrics)

    path = os.path.join(METRICS_DIR, "baseline.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {path}")
    return results


# ── Core evaluation engine ────────────────────────────────────────────────────

@torch.no_grad()
def run_task_eval(task, compressor, task_head, dim):
    """
    Compress test embeddings and evaluate task performance.

    Args:
        task        : 'sts', 'nli', or 'classification'
        compressor  : nn.Module or None (baseline: skip compression)
        task_head   : nn.Module task head
        dim         : compressed dimension (used only for embedding_quality)

    Returns:
        dict of metric name → value, plus optional embedding_quality
    """
    s1, s2, labels = load_test_embeddings(task)

    if compressor is not None:
        compressor.eval()
        # Compress in batches to avoid OOM
        s1 = _batch_compress(compressor, s1)
        if s2 is not None:
            s2 = _batch_compress(compressor, s2)

    task_head.eval()

    # Run task head
    if task in ("sts", "nli"):
        preds = _batch_predict_pair(task_head, task, s1, s2)
    else:
        preds = _batch_predict_single(task_head, s1)

    metrics = evaluate_task(task, preds, labels.numpy())
    return metrics


def _batch_compress(compressor, embeddings, batch_size=512):
    """Compress a large tensor in batches."""
    out = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size].to(DEVICE)
        out.append(compressor(batch).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def _batch_predict_pair(head, task, s1, s2, batch_size=512):
    """Run pair-input task head in batches, return numpy predictions."""
    preds = []
    for i in range(0, len(s1), batch_size):
        z1 = s1[i:i+batch_size].to(DEVICE)
        z2 = s2[i:i+batch_size].to(DEVICE)
        if task == "sts":
            out = head(z1, z2).cpu().numpy()          # (B,) floats
        else:
            out = head(z1, z2).argmax(dim=-1).cpu().numpy()  # (B,) ints
        preds.append(out)
    return np.concatenate(preds)


@torch.no_grad()
def _batch_predict_single(head, s1, batch_size=512):
    """Run single-input task head in batches, return numpy predictions."""
    preds = []
    for i in range(0, len(s1), batch_size):
        z = s1[i:i+batch_size].to(DEVICE)
        out = head(z).argmax(dim=-1).cpu().numpy()
        preds.append(out)
    return np.concatenate(preds)


def _print_task_metrics(task, metrics):
    if task == "sts":
        print(f"    STS   spearman={metrics.get('spearman', '?'):.4f}")
    else:
        print(f"    {task.upper():<4}  accuracy={metrics.get('accuracy', '?'):.4f}  "
              f"f1={metrics.get('f1', '?'):.4f}")


# ── Task-agnostic evaluation ──────────────────────────────────────────────────

def evaluate_task_agnostic(method, dim):
    """
    Load task-agnostic compressor checkpoint.
    For each eval task, train a fresh task head on compressed train embeddings,
    then evaluate on test set.
    """
    print(f"\n{'='*60}")
    print(f"  TASK-AGNOSTIC | method={method} | dim={dim}")
    print(f"{'='*60}")

    path = ckpt_path(method, "task_agnostic", None, dim)
    if not os.path.exists(path):
        print(f"  [SKIP] checkpoint not found: {path}")
        return None

    compressor = get_compressor(method, dim)
    compressor.load_state_dict(torch.load(path, map_location=DEVICE))
    compressor.to(DEVICE).eval()

    results = {"mode": "task_agnostic", "method": method, "dim": dim}

    for task in TASKS:
        head = _train_eval_head(task, compressor, dim)
        metrics = run_task_eval(task, compressor, head, dim)
        results[task] = metrics
        _print_task_metrics(task, metrics)

    tag  = f"{method}_agnostic_dim{dim}"
    path = os.path.join(METRICS_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ── Task-aware evaluation ─────────────────────────────────────────────────────

def evaluate_task_aware(method, train_task, dim):
    """
    Load task-aware compressor trained on train_task.
    Evaluate cross-task: test on ALL 3 tasks to reveal generalisation cost.
    """
    print(f"\n{'='*60}")
    print(f"  TASK-AWARE | method={method} | train_task={train_task} | dim={dim}")
    print(f"{'='*60}")

    comp_path = ckpt_path(method, "task_aware", train_task, dim)
    head_path = ckpt_path(f"{method}_head", "task_aware", train_task, dim)

    if not os.path.exists(comp_path):
        print(f"  [SKIP] compressor checkpoint not found: {comp_path}")
        return None

    compressor = get_compressor(method, dim)
    compressor.load_state_dict(torch.load(comp_path, map_location=DEVICE))
    compressor.to(DEVICE).eval()

    results = {"mode": "task_aware", "method": method,
               "train_task": train_task, "dim": dim}

    for eval_task in TASKS:
        # For the trained task, load the saved head; for cross-tasks, train fresh
        if eval_task == train_task and os.path.exists(head_path):
            head = get_task_head(eval_task, compressed_dim=dim)
            head.load_state_dict(torch.load(head_path, map_location=DEVICE))
            head.to(DEVICE)
        else:
            head = _train_eval_head(eval_task, compressor, dim)

        metrics = run_task_eval(eval_task, compressor, head, dim)
        results[eval_task] = metrics
        _print_task_metrics(eval_task, metrics)

    tag  = f"{method}_aware_{train_task}_dim{dim}"
    path = os.path.join(METRICS_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ── Helper: train a lightweight probe head on compressed embeddings ───────────

def _train_eval_head(task, compressor, dim, epochs=5):
    """
    Train a small evaluation head on compressed train embeddings.
    The compressor is frozen; only the head is trained.
    Used for task-agnostic, PCA, mixed, and cross-task evaluation.
    """
    def _load(split, key):
        return torch.tensor(
            np.load(os.path.join(EMBEDDINGS_DIR, f"{key}.npy")),
            dtype=torch.float32
        )

    compressor.eval()

    # Compress train embeddings (once, in memory)
    with torch.no_grad():
        s1_raw = _load("train", f"{task}_train_s1")
        s1     = _batch_compress(compressor, s1_raw)

        if task in ("sts", "nli"):
            s2_raw = _load("train", f"{task}_train_s2")
            s2     = _batch_compress(compressor, s2_raw)
        else:
            s2 = None

        labels = _load("train", f"{task}_train_labels")

    if s2 is not None:
        loader = DataLoader(TensorDataset(s1, s2, labels),
                            batch_size=BATCH_SIZE, shuffle=True)
    else:
        loader = DataLoader(TensorDataset(s1, labels),
                            batch_size=BATCH_SIZE, shuffle=True)

    head      = get_task_head(task, compressed_dim=dim).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-4)

    for _ in range(epochs):
        head.train()
        for batch in loader:
            optimizer.zero_grad()
            if task in ("sts", "nli"):
                z1, z2, labs = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
                loss = head.loss(z1, z2, labs)
            else:
                z1, labs = batch[0].to(DEVICE), batch[1].to(DEVICE)
                loss = head.loss(z1, labs)
            loss.backward()
            optimizer.step()

    return head


# ── PCA Baseline evaluation ───────────────────────────────────────────────────

def evaluate_pca_baseline(dim):
    """
    Fit sklearn PCA on the pooled train embeddings from all three tasks,
    then evaluate on all 3 eval tasks.  Saves results to pca_dim{dim}.json.

    This answers the question: does a learned (neural) compressor actually
    beat a classical unsupervised dimensionality-reduction method?
    """
    print(f"\n{'='*60}")
    print(f"  PCA BASELINE | dim={dim}")
    print(f"{'='*60}")

    # Fit PCA on pooled train set (all 3 tasks) for a fair, task-agnostic view
    print("  Fitting PCA on pooled train embeddings...")
    all_train = []
    for task in TASKS:
        s1 = np.load(os.path.join(EMBEDDINGS_DIR, f"{task}_train_s1.npy"))
        all_train.append(s1)
        if task in ("sts", "nli"):
            s2 = np.load(os.path.join(EMBEDDINGS_DIR, f"{task}_train_s2.npy"))
            all_train.append(s2)

    pool = np.concatenate(all_train, axis=0)
    print(f"  PCA pool shape: {pool.shape}")

    pca = PCA(n_components=dim, random_state=SEED)
    pca.fit(pool)
    explained = pca.explained_variance_ratio_.sum()
    print(f"  Explained variance ({dim} PCs): {explained:.4f}")

    # Wrap PCA as an nn.Module so it plugs into existing helpers
    wrapper = PCAWrapper(pca).to(DEVICE)

    results = {
        "mode": "pca",
        "dim":  dim,
        "explained_variance_ratio": float(explained),
    }

    for task in TASKS:
        print(f"  Evaluating on task={task}...")
        head    = _train_eval_head(task, wrapper, dim)
        metrics = run_task_eval(task, wrapper, head, dim)
        results[task] = metrics
        _print_task_metrics(task, metrics)

    out_path = os.path.join(METRICS_DIR, f"pca_dim{dim}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")
    return results


# ── Task-selector table ───────────────────────────────────────────────────────

def build_task_selector_table():
    """
    Read full_results_table.csv and find the best configuration (argmax)
    for each downstream task column.  Produces a concise "task-selector"
    recommendation table that directly answers the professor's question:
    'Which compressor should I use for my task?'

    Saves results/metrics/task_selector_table.csv and prints a summary.
    """
    csv_path = os.path.join(METRICS_DIR, "full_results_table.csv")
    if not os.path.exists(csv_path):
        print(f"[TaskSelector] {csv_path} not found — run --all first.")
        return None

    df = pd.read_csv(csv_path)

    task_cols = {
        "sts":            "STS_spearman",
        "nli":            "NLI_accuracy",
        "classification": "CLS_accuracy",
    }

    rows = []
    print(f"\n{'='*60}")
    print("  TASK-SELECTOR TABLE")
    print(f"{'='*60}")

    for task_name, col in task_cols.items():
        sub = df.dropna(subset=[col])
        best_idx = sub[col].idxmax()
        best_row = sub.loc[best_idx]
        score    = best_row[col]

        # Also collect agnostic best and PCA best for comparison
        agnostic_mask = sub["Mode"] == "task_agnostic"
        agnostic_best  = sub.loc[agnostic_mask, col].max() if agnostic_mask.any() else None

        entry = {
            "Task":          task_name,
            "Best_Method":   best_row["Method"],
            "Best_Mode":     best_row["Mode"],
            "Best_TrainTask": best_row["TrainTask"],
            "Best_Dim":      int(best_row["Dim"]),
            "Best_Score":    round(float(score), 4),
            "Agnostic_Best": round(float(agnostic_best), 4) if agnostic_best else None,
            "Metric":        col,
        }
        rows.append(entry)

        print(f"\n  Task: {task_name.upper()}")
        print(f"    Best overall : {best_row['Method']} | {best_row['Mode']} | "
              f"train_task={best_row['TrainTask']} | dim={int(best_row['Dim'])} "
              f"→ {col}={score:.4f}")
        if agnostic_best:
            delta = score - agnostic_best
            print(f"    Agnostic best: {agnostic_best:.4f}  (aware gain: {delta:+.4f})")

    selector_df = pd.DataFrame(rows)
    out_path = os.path.join(METRICS_DIR, "task_selector_table.csv")
    selector_df.to_csv(out_path, index=False)
    print(f"\n  Saved task-selector: {out_path}")
    print(selector_df.to_string(index=False))
    return selector_df


# ── Mixed-compressor evaluation ───────────────────────────────────────────────

def evaluate_mixed(method, task_a, task_b, dim):
    """
    Load a mixed compressor (jointly trained on task_a + task_b) and evaluate
    on ALL three tasks.  Each eval task gets a fresh probe head trained on the
    compressed embeddings.

    Checkpoint convention (must match train_compression.py):
        models/checkpoints/{method}/mixed_{task_a}_{task_b}/dim{dim}.pt
    """
    print(f"\n{'='*60}")
    print(f"  MIXED | method={method} | tasks=({task_a},{task_b}) | dim={dim}")
    print(f"{'='*60}")

    comp_path = mixed_ckpt_path(method, task_a, task_b, dim)
    if not os.path.exists(comp_path):
        print(f"  [SKIP] mixed compressor checkpoint not found: {comp_path}")
        return None

    compressor = get_compressor(method, dim)
    compressor.load_state_dict(torch.load(comp_path, map_location=DEVICE))
    compressor.to(DEVICE).eval()

    results = {
        "mode":   "mixed",
        "method": method,
        "task_a": task_a,
        "task_b": task_b,
        "dim":    dim,
    }

    for eval_task in TASKS:
        head    = _train_eval_head(eval_task, compressor, dim)
        metrics = run_task_eval(eval_task, compressor, head, dim)
        results[eval_task] = metrics
        _print_task_metrics(eval_task, metrics)

    tag  = f"{method}_mixed_{task_a}_{task_b}_dim{dim}"
    path = os.path.join(METRICS_DIR, f"{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {path}")
    return results


# ── Full sweep + CSV table ────────────────────────────────────────────────────

def build_results_table(all_results):
    """
    Flatten the list of result dicts into a tidy DataFrame and save as CSV.
    """
    rows = []
    for r in all_results:
        if r is None:
            continue
        mode       = r.get("mode")
        method     = r.get("method", "—")
        dim        = r.get("dim", 768)
        train_task = r.get("train_task", "agnostic" if mode == "task_agnostic" else "—")

        row = {
            "Method":     method,
            "Dim":        dim,
            "Mode":       mode,
            "TrainTask":  train_task,
            "STS_spearman":   r.get("sts",            {}).get("spearman",  None),
            "NLI_accuracy":   r.get("nli",             {}).get("accuracy", None),
            "CLS_accuracy":   r.get("classification",  {}).get("accuracy", None),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(METRICS_DIR, "full_results_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[Results] Saved full table → {csv_path}")
    print(df.to_string(index=False))
    return df


def evaluate_all():
    """
    Full evaluation sweep:
      1. Baseline (768-dim)
      2. Task-agnostic: all methods × all dims
      3. Task-aware:    all methods × all train tasks × all dims
    """
    all_results = []

    # 1. Baseline
    all_results.append(evaluate_baseline())

    # 2. Task-agnostic
    for method in ["linear", "autoencoder", "distillation"]:
        for dim in COMPRESSION_DIMS:
            all_results.append(evaluate_task_agnostic(method, dim))

    # 3. Task-aware (cross-task)
    for method in ["linear", "autoencoder", "distillation"]:
        for train_task in TASKS:
            for dim in COMPRESSION_DIMS:
                all_results.append(evaluate_task_aware(method, train_task, dim))

    build_results_table(all_results)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline",   action="store_true",
                        help="Evaluate 768-dim baseline on all tasks")
    parser.add_argument("--all",        action="store_true",
                        help="Full sweep: baseline + all methods/modes/dims")
    parser.add_argument("--method",     default="linear",
                        choices=["linear", "autoencoder", "distillation"])
    parser.add_argument("--mode",       default="task_agnostic",
                        choices=["task_agnostic", "task_aware", "mixed"])
    parser.add_argument("--train_task", default="sts",
                        choices=list(TASKS.keys()),
                        help="Which task the compressor was trained on (task_aware only)")
    parser.add_argument("--dim",        type=int, default=128,
                        choices=COMPRESSION_DIMS)
    # PCA baseline args
    parser.add_argument("--pca",        action="store_true",
                        help="Run PCA baseline across all COMPRESSION_DIMS")
    parser.add_argument("--pca_dim",    type=int, default=None,
                        choices=COMPRESSION_DIMS,
                        help="Run PCA baseline for a single dim")
    parser.add_argument("--task_selector", action="store_true",
                        help="Build task-selector table from full_results_table.csv")
    # Mixed compressor args
    parser.add_argument("--task_a",     default="sts",  choices=list(TASKS.keys()))
    parser.add_argument("--task_b",     default="nli",  choices=list(TASKS.keys()))

    args = parser.parse_args()

    if args.all:
        evaluate_all()
    elif args.baseline:
        evaluate_baseline()
    elif args.pca:
        dims = [args.pca_dim] if args.pca_dim else COMPRESSION_DIMS
        for d in dims:
            evaluate_pca_baseline(d)
        build_task_selector_table()
    elif args.task_selector:
        build_task_selector_table()
    elif args.mode == "task_agnostic":
        evaluate_task_agnostic(args.method, args.dim)
    elif args.mode == "mixed":
        evaluate_mixed(args.method, args.task_a, args.task_b, args.dim)
    else:
        evaluate_task_aware(args.method, args.train_task, args.dim)


if __name__ == "__main__":
    main()
