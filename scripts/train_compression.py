"""
scripts/train_compression.py
Unified training script for all compression methods x all tasks.

Modes:
  task_agnostic : reconstruction loss only, balanced multi-source data
  task_aware    : compressor + task head trained jointly
  mixed         : one compressor shared across two tasks, combined loss

Usage:
  python scripts/train_compression.py --method linear --mode task_agnostic --dim 64
  python scripts/train_compression.py --method autoencoder --mode task_aware --task sts --dim 128
  python scripts/train_compression.py --method autoencoder --mode mixed --task_a sts --task_b nli --dim 128
  python scripts/train_compression.py --all
"""

import os, sys, argparse, json, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (EMBEDDING_DIM, COMPRESSION_DIMS, TASKS,
                    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
                    WEIGHT_DECAY, SEED, EMBEDDINGS_DIR, METRICS_DIR)
from compression.linear       import LinearCompressor
from compression.autoencoder  import Autoencoder, DeepAutoencoder
from compression.distillation import StudentCompressor, DistillationLoss
from models.task_heads         import get_task_head

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Trainer] Device: {DEVICE}")


# ── Checkpoint path ───────────────────────────────────────────────────────────

def ckpt_path(method, mode, task, dim):
    folder = os.path.join("models", "checkpoints", method, mode)
    os.makedirs(folder, exist_ok=True)
    tag = f"{task}_dim{dim}" if task else f"dim{dim}"
    return os.path.join(folder, f"{tag}.pt")


# ── Load cached embeddings ────────────────────────────────────────────────────

def load_embeddings(task, split):
    def _load(name):
        path = os.path.join(EMBEDDINGS_DIR, f"{name}.npy")
        return torch.tensor(np.load(path), dtype=torch.float32)
    labels = _load(f"{task}_{split}_labels")
    if task in ("sts", "nli"):
        return _load(f"{task}_{split}_s1"), _load(f"{task}_{split}_s2"), labels
    else:
        return _load(f"{task}_{split}_s1"), None, labels


def make_dataloader(task, split, batch_size=BATCH_SIZE, shuffle=True):
    s1, s2, labels = load_embeddings(task, split)
    if s2 is not None:
        dataset = TensorDataset(s1, s2, labels)
    else:
        dataset = TensorDataset(s1, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


# ── Balanced multi-source loader (borrowed + improved from ref repo) ──────────

def make_balanced_agnostic_loader(target_per_source=50000, batch_size=BATCH_SIZE):
    """
    Pools embeddings from ALL 3 tasks, balances each to target_per_source,
    then L2-normalises the combined pool.

    Why: ensures the agnostic compressor sees diverse linguistic content
    rather than being biased toward one task's sentence distribution.
    Borrowed from reference repo, improved with L2 normalisation.
    """
    def _load_flat(task, split="train"):
        def _npy(name):
            return torch.tensor(
                np.load(os.path.join(EMBEDDINGS_DIR, f"{name}.npy")),
                dtype=torch.float32
            )
        if task in ("sts", "nli"):
            return torch.cat([_npy(f"{task}_{split}_s1"),
                               _npy(f"{task}_{split}_s2")], dim=0)
        return _npy(f"{task}_{split}_s1")

    def _balance(tensor, target):
        n = tensor.size(0)
        idx = torch.randperm(n)[:target] if n >= target \
              else torch.randint(0, n, (target,))
        return tensor[idx]

    print("[BalancedLoader] Building balanced multi-source pool...")
    pools = []
    for task in ["sts", "nli", "classification"]:
        raw = _load_flat(task)
        bal = _balance(raw, target_per_source)
        pools.append(bal)
        print(f"  {task:>14}: raw={raw.shape[0]:>6} -> balanced={bal.shape[0]}")

    combined = torch.cat(pools, dim=0)
    combined = F.normalize(combined, p=2, dim=1)   # L2-normalise
    print(f"  Combined pool : {combined.shape}  (L2-normalised)")

    return DataLoader(TensorDataset(combined),
                      batch_size=batch_size, shuffle=True, num_workers=2)


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


# ── Task-agnostic training ────────────────────────────────────────────────────

def train_task_agnostic(method, dim, epochs=NUM_EPOCHS):
    """
    Train compressor with reconstruction loss only.
    Uses balanced multi-source pool (STS + NLI + SST2).
    """
    print(f"\n{'='*60}")
    print(f"  Task-Agnostic | method={method} | dim={dim}")
    print(f"{'='*60}")

    compressor = get_compressor(method, dim).to(DEVICE)
    optimizer  = torch.optim.Adam(compressor.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loader  = make_balanced_agnostic_loader()   # <-- balanced multi-source
    history = []

    for epoch in range(1, epochs + 1):
        compressor.train()
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss = compressor.reconstruction_loss(x)
            loss.backward()
            nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        scheduler.step()
        history.append({"epoch": epoch, "loss": avg})
        print(f"  Epoch {epoch:>2}/{epochs}  recon_loss={avg:.5f}")

    path = ckpt_path(method, "task_agnostic", None, dim)
    torch.save(compressor.state_dict(), path)
    print(f"  Saved: {path}")

    with open(os.path.join(METRICS_DIR,
              f"{method}_agnostic_dim{dim}.json"), "w") as f:
        json.dump({"method": method, "mode": "task_agnostic",
                   "dim": dim, "history": history}, f, indent=2)

    return compressor


# ── Task-aware training ───────────────────────────────────────────────────────

def train_task_aware(method, task, dim, epochs=NUM_EPOCHS):
    """
    Train compressor + task head jointly.
    Distillation: adds teacher-student alignment loss.
    Linear/AE:    adds reconstruction loss as auxiliary.
    """
    print(f"\n{'='*60}")
    print(f"  Task-Aware | method={method} | task={task} | dim={dim}")
    print(f"{'='*60}")

    compressor = get_compressor(method, dim).to(DEVICE)
    task_head  = get_task_head(task, dim).to(DEVICE)

    params    = list(compressor.parameters()) + list(task_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    distill_loss_fn = DistillationLoss() if method == "distillation" else None

    # STS uses "dev" not "validation"
    train_loader = make_dataloader(task, "train")
    history = []

    for epoch in range(1, epochs + 1):
        compressor.train()
        task_head.train()
        total_task  = 0.0
        total_aux   = 0.0
        n = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()

            if task in ("sts", "nli"):
                s1, s2, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
                z1 = compressor(s1)
                z2 = compressor(s2)
                task_loss = task_head.loss(z1, z2, labels)
            else:
                s1, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
                z1 = compressor(s1)
                task_loss = task_head.loss(z1, labels)

            if method == "distillation":
                proj = compressor.project_to_teacher_space(z1)
                total_loss, task_loss, aux_loss = distill_loss_fn(task_loss, proj, s1)
            else:
                aux_loss   = compressor.reconstruction_loss(s1)
                total_loss = 0.7 * task_loss + 0.3 * aux_loss

            total_loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_task += task_loss.item()
            total_aux  += aux_loss.item()
            n += 1

        scheduler.step()
        history.append({"epoch": epoch,
                         "task_loss": total_task / n,
                         "aux_loss":  total_aux  / n})
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"task={total_task/n:.4f}  aux={total_aux/n:.4f}")

    # Save
    comp_path = ckpt_path(method, "task_aware", task, dim)
    head_path = ckpt_path(f"{method}_head", "task_aware", task, dim)
    torch.save(compressor.state_dict(), comp_path)
    torch.save(task_head.state_dict(),  head_path)
    print(f"  Compressor : {comp_path}")
    print(f"  Task head  : {head_path}")

    with open(os.path.join(METRICS_DIR,
              f"{method}_aware_{task}_dim{dim}.json"), "w") as f:
        json.dump({"method": method, "mode": "task_aware",
                   "task": task, "dim": dim, "history": history}, f, indent=2)

    return compressor, task_head


# ── Mixed-task training ───────────────────────────────────────────────────────

def _mixed_ckpt_folder(method, task_a, task_b):
    folder = os.path.join("models", "checkpoints", method, f"mixed_{task_a}_{task_b}")
    os.makedirs(folder, exist_ok=True)
    return folder


def train_mixed(method, task_a, task_b, alpha=0.5, dim=128, epochs=NUM_EPOCHS):
    """
    Train one shared compressor jointly on two tasks.

    Loss = alpha * loss_task_a + (1-alpha) * loss_task_b

    Each task has its own head; both heads + the compressor are optimised
    together.  When the two dataloaders have different lengths we cycle the
    shorter one so every epoch sees all batches from the longer task.

    Checkpoints:
        models/checkpoints/{method}/mixed_{task_a}_{task_b}/dim{dim}.pt
        models/checkpoints/{method}_head/mixed_{task_a}_{task_b}/{task_a}_dim{dim}.pt
        models/checkpoints/{method}_head/mixed_{task_a}_{task_b}/{task_b}_dim{dim}.pt
    """
    print(f"\n{'='*60}")
    print(f"  Mixed | method={method} | tasks=({task_a},{task_b}) | dim={dim} | alpha={alpha}")
    print(f"{'='*60}")

    compressor = get_compressor(method, dim).to(DEVICE)
    head_a     = get_task_head(task_a, dim).to(DEVICE)
    head_b     = get_task_head(task_b, dim).to(DEVICE)

    params    = (list(compressor.parameters()) +
                 list(head_a.parameters()) +
                 list(head_b.parameters()))
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    distill_loss_fn = DistillationLoss() if method == "distillation" else None

    loader_a = make_dataloader(task_a, "train")
    loader_b = make_dataloader(task_b, "train")

    # Pair up batches; cycle the shorter loader
    len_a, len_b = len(loader_a), len(loader_b)
    history = []

    for epoch in range(1, epochs + 1):
        compressor.train()
        head_a.train()
        head_b.train()

        total_loss_a = 0.0
        total_loss_b = 0.0
        n = 0

        # zip the longer loader with a cycling version of the shorter
        if len_a >= len_b:
            paired = zip(loader_a, itertools.cycle(loader_b))
            n_steps = len_a
        else:
            paired = zip(itertools.cycle(loader_a), loader_b)
            n_steps = len_b

        for batch_a, batch_b in tqdm(paired, total=n_steps,
                                      desc=f"Epoch {epoch}/{epochs}", leave=False):
            optimizer.zero_grad()

            # ── Task A loss ──────────────────────────────────────
            if task_a in ("sts", "nli"):
                s1a, s2a, labs_a = (batch_a[0].to(DEVICE),
                                    batch_a[1].to(DEVICE),
                                    batch_a[2].to(DEVICE))
                z1a = compressor(s1a)
                z2a = compressor(s2a)
                loss_a = head_a.loss(z1a, z2a, labs_a)
                if method == "distillation":
                    proj_a = compressor.project_to_teacher_space(z1a)
                    loss_a, _, _ = distill_loss_fn(loss_a, proj_a, s1a)
                else:
                    aux_a  = compressor.reconstruction_loss(s1a)
                    loss_a = 0.7 * loss_a + 0.3 * aux_a
            else:  # classification
                s1a, labs_a = batch_a[0].to(DEVICE), batch_a[1].to(DEVICE)
                z1a    = compressor(s1a)
                loss_a = head_a.loss(z1a, labs_a)
                if method == "distillation":
                    proj_a = compressor.project_to_teacher_space(z1a)
                    loss_a, _, _ = distill_loss_fn(loss_a, proj_a, s1a)
                else:
                    aux_a  = compressor.reconstruction_loss(s1a)
                    loss_a = 0.7 * loss_a + 0.3 * aux_a

            # ── Task B loss ──────────────────────────────────────
            if task_b in ("sts", "nli"):
                s1b, s2b, labs_b = (batch_b[0].to(DEVICE),
                                    batch_b[1].to(DEVICE),
                                    batch_b[2].to(DEVICE))
                z1b = compressor(s1b)
                z2b = compressor(s2b)
                loss_b = head_b.loss(z1b, z2b, labs_b)
                if method == "distillation":
                    proj_b = compressor.project_to_teacher_space(z1b)
                    loss_b, _, _ = distill_loss_fn(loss_b, proj_b, s1b)
                else:
                    aux_b  = compressor.reconstruction_loss(s1b)
                    loss_b = 0.7 * loss_b + 0.3 * aux_b
            else:  # classification
                s1b, labs_b = batch_b[0].to(DEVICE), batch_b[1].to(DEVICE)
                z1b    = compressor(s1b)
                loss_b = head_b.loss(z1b, labs_b)
                if method == "distillation":
                    proj_b = compressor.project_to_teacher_space(z1b)
                    loss_b, _, _ = distill_loss_fn(loss_b, proj_b, s1b)
                else:
                    aux_b  = compressor.reconstruction_loss(s1b)
                    loss_b = 0.7 * loss_b + 0.3 * aux_b

            combined = alpha * loss_a + (1 - alpha) * loss_b
            combined.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss_a += loss_a.item()
            total_loss_b += loss_b.item()
            n += 1

        scheduler.step()
        history.append({
            "epoch":  epoch,
            "loss_a": total_loss_a / n,
            "loss_b": total_loss_b / n,
        })
        print(f"  Epoch {epoch:>2}/{epochs}  "
              f"loss_{task_a}={total_loss_a/n:.4f}  "
              f"loss_{task_b}={total_loss_b/n:.4f}")

    # ── Save checkpoints ────────────────────────────────────────────
    comp_folder  = _mixed_ckpt_folder(method, task_a, task_b)
    comp_path    = os.path.join(comp_folder, f"dim{dim}.pt")
    torch.save(compressor.state_dict(), comp_path)
    print(f"  Compressor: {comp_path}")

    head_folder = _mixed_ckpt_folder(f"{method}_head", task_a, task_b)
    for head, tag in [(head_a, task_a), (head_b, task_b)]:
        hp = os.path.join(head_folder, f"{tag}_dim{dim}.pt")
        torch.save(head.state_dict(), hp)
        print(f"  Head ({tag}): {hp}")

    hist_path = os.path.join(METRICS_DIR,
                             f"{method}_mixed_{task_a}_{task_b}_dim{dim}_train.json")
    with open(hist_path, "w") as f:
        json.dump({"method": method, "mode": "mixed",
                   "task_a": task_a, "task_b": task_b,
                   "dim": dim, "alpha": alpha,
                   "history": history}, f, indent=2)
    print(f"  History : {hist_path}")

    return compressor, head_a, head_b


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="linear",
                        choices=["linear", "autoencoder", "distillation"])
    parser.add_argument("--mode",   default="task_aware",
                        choices=["task_agnostic", "task_aware", "mixed"])
    parser.add_argument("--task",   default="sts",
                        choices=list(TASKS.keys()))
    parser.add_argument("--task_a", default="sts",
                        choices=list(TASKS.keys()),
                        help="First task for mixed mode")
    parser.add_argument("--task_b", default="nli",
                        choices=list(TASKS.keys()),
                        help="Second task for mixed mode")
    parser.add_argument("--alpha",  type=float, default=0.5,
                        help="Weight for task_a loss in mixed mode")
    parser.add_argument("--dim",    type=int, default=128,
                        choices=COMPRESSION_DIMS)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--all",    action="store_true",
                        help="Train all method x task x dim combinations")
    args = parser.parse_args()

    if args.all:
        methods = ["linear", "autoencoder", "distillation"]
        tasks   = list(TASKS.keys())
        dims    = COMPRESSION_DIMS

        print("\n[Train All] Task-agnostic runs...")
        for method in methods:
            for dim in dims:
                train_task_agnostic(method, dim, epochs=args.epochs)

        print("\n[Train All] Task-aware runs...")
        for method in methods:
            for task in tasks:
                for dim in dims:
                    train_task_aware(method, task, dim, epochs=args.epochs)

        print("\n[Train All] Mixed runs...")
        for method in ["autoencoder"]:      # extend to linear if time permits
            for pair in [("sts","nli"), ("nli","classification")]:
                for dim in [128, 256]:
                    train_mixed(method, pair[0], pair[1], dim=dim, epochs=args.epochs)

    elif args.mode == "task_agnostic":
        train_task_agnostic(args.method, args.dim, epochs=args.epochs)
    elif args.mode == "mixed":
        train_mixed(args.method, args.task_a, args.task_b,
                    alpha=args.alpha, dim=args.dim, epochs=args.epochs)
    else:
        train_task_aware(args.method, args.task, args.dim, epochs=args.epochs)


if __name__ == "__main__":
    main()
