"""
scripts/evaluate.py
Full evaluation sweep for all compression methods, modes, tasks and dims.

Modes:
  baseline       : 768-dim ceiling — no compression, task head trained on raw embeddings
  task_agnostic  : compressor trained on reconstruction only (no task signal)
  task_aware     : compressor + task head trained jointly

Usage:
  python scripts/evaluate.py --baseline
  python scripts/evaluate.py --method linear --mode task_agnostic --dim 128
  python scripts/evaluate.py --method linear --mode task_aware --train_task sts --dim 128
  python scripts/evaluate.py --all
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import pandas as pd

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
    Used for task-agnostic and cross-task evaluation.
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
                        choices=["task_agnostic", "task_aware"])
    parser.add_argument("--train_task", default="sts",
                        choices=list(TASKS.keys()),
                        help="Which task the compressor was trained on (task_aware only)")
    parser.add_argument("--dim",        type=int, default=128,
                        choices=COMPRESSION_DIMS)
    args = parser.parse_args()

    if args.all:
        evaluate_all()
    elif args.baseline:
        evaluate_baseline()
    elif args.mode == "task_agnostic":
        evaluate_task_agnostic(args.method, args.dim)
    else:
        evaluate_task_aware(args.method, args.train_task, args.dim)


if __name__ == "__main__":
    main()
