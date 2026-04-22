"""
scripts/multi_seed_eval.py
Run the best NLI-aware autoencoder config (dim=256) across multiple seeds
to produce mean ± std confidence intervals.

Also runs the best agnostic baseline (autoencoder agnostic dim=256)
for a fair comparison.

Usage:
    python scripts/multi_seed_eval.py
    python scripts/multi_seed_eval.py --seeds 42 123 7 99 2024
"""

import os, sys, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (EMBEDDING_DIM, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
                    WEIGHT_DECAY, EMBEDDINGS_DIR, METRICS_DIR)
from compression.autoencoder import Autoencoder
from models.task_heads       import get_task_head
from evaluation.metrics      import evaluate_task

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM    = 256
METHOD = "autoencoder"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_embeddings(task, split):
    def _load(name):
        return torch.tensor(
            np.load(os.path.join(EMBEDDINGS_DIR, f"{name}.npy")),
            dtype=torch.float32)
    labels = _load(f"{task}_{split}_labels")
    if task in ("sts", "nli"):
        return _load(f"{task}_{split}_s1"), _load(f"{task}_{split}_s2"), labels
    return _load(f"{task}_{split}_s1"), None, labels


def make_dataloader(task, split, batch_size=BATCH_SIZE, shuffle=True):
    s1, s2, labels = load_embeddings(task, split)
    if s2 is not None:
        dataset = TensorDataset(s1, s2, labels)
    else:
        dataset = TensorDataset(s1, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def batch_compress(compressor, embeddings, batch_size=512):
    out = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size].to(DEVICE)
        out.append(compressor(batch).cpu())
    return torch.cat(out, dim=0)


@torch.no_grad()
def evaluate_nli(compressor, head):
    """Evaluate on NLI test set, return accuracy."""
    s1, s2, labels = load_embeddings("nli", "test")
    compressor.eval()
    head.eval()
    z1 = batch_compress(compressor, s1)
    z2 = batch_compress(compressor, s2)
    preds = []
    for i in range(0, len(z1), 512):
        a = z1[i:i+512].to(DEVICE)
        b = z2[i:i+512].to(DEVICE)
        preds.append(head(a, b).argmax(dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    metrics = evaluate_task("nli", preds, labels.numpy())
    return metrics["accuracy"]


@torch.no_grad()
def evaluate_sts(compressor, head):
    """Evaluate on STS test set, return spearman."""
    s1, s2, labels = load_embeddings("sts", "test")
    compressor.eval()
    head.eval()
    z1 = batch_compress(compressor, s1)
    z2 = batch_compress(compressor, s2)
    preds = []
    for i in range(0, len(z1), 512):
        a = z1[i:i+512].to(DEVICE)
        b = z2[i:i+512].to(DEVICE)
        preds.append(head(a, b).cpu().numpy())
    preds = np.concatenate(preds)
    metrics = evaluate_task("sts", preds, labels.numpy())
    return metrics["spearman"]


# ── Train + Eval for one seed ────────────────────────────────────────────────

def train_and_eval_aware(seed):
    """Train NLI-aware autoencoder at dim=256 with given seed, eval on NLI."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    compressor = Autoencoder(input_dim=EMBEDDING_DIM, output_dim=DIM).to(DEVICE)
    task_head  = get_task_head("nli", DIM).to(DEVICE)

    params    = list(compressor.parameters()) + list(task_head.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    loader = make_dataloader("nli", "train")

    for epoch in range(1, NUM_EPOCHS + 1):
        compressor.train()
        task_head.train()
        total_loss = 0.0
        n = 0
        for batch in tqdm(loader, desc=f"  Seed {seed} Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            s1, s2, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            optimizer.zero_grad()
            z1 = compressor(s1)
            z2 = compressor(s2)
            task_loss = task_head.loss(z1, z2, labels)
            aux_loss  = compressor.reconstruction_loss(s1)
            loss      = 0.7 * task_loss + 0.3 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            total_loss += loss.item()
            n += 1
        scheduler.step()
        print(f"    Epoch {epoch:>2}/{NUM_EPOCHS}  loss={total_loss/n:.4f}")

    nli_acc = evaluate_nli(compressor, task_head)
    return nli_acc


def train_and_eval_agnostic(seed):
    """Train agnostic autoencoder at dim=256 with given seed, eval on NLI."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    compressor = Autoencoder(input_dim=EMBEDDING_DIM, output_dim=DIM).to(DEVICE)
    optimizer  = torch.optim.Adam(compressor.parameters(),
                                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # Build balanced multi-source pool (same as train_compression.py)
    def _load_flat(task, split="train"):
        def _npy(name):
            return torch.tensor(
                np.load(os.path.join(EMBEDDINGS_DIR, f"{name}.npy")),
                dtype=torch.float32)
        if task in ("sts", "nli"):
            return torch.cat([_npy(f"{task}_{split}_s1"),
                              _npy(f"{task}_{split}_s2")], dim=0)
        return _npy(f"{task}_{split}_s1")

    pools = []
    for task in ["sts", "nli", "classification"]:
        raw = _load_flat(task)
        idx = torch.randperm(raw.size(0))[:50000]
        pools.append(raw[idx])
    combined = torch.cat(pools, dim=0)
    combined = F.normalize(combined, p=2, dim=1)
    loader = DataLoader(TensorDataset(combined), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        compressor.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"  Seed {seed} Epoch {epoch}/{NUM_EPOCHS}", leave=False):
            x = batch[0].to(DEVICE)
            optimizer.zero_grad()
            loss = compressor.reconstruction_loss(x)
            loss.backward()
            nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"    Epoch {epoch:>2}/{NUM_EPOCHS}  loss={total_loss/len(loader):.5f}")

    # Train a fresh NLI probe head on the compressed embeddings
    compressor.eval()
    with torch.no_grad():
        s1_raw = torch.tensor(np.load(os.path.join(EMBEDDINGS_DIR, "nli_train_s1.npy")),
                              dtype=torch.float32)
        s2_raw = torch.tensor(np.load(os.path.join(EMBEDDINGS_DIR, "nli_train_s2.npy")),
                              dtype=torch.float32)
        labels = torch.tensor(np.load(os.path.join(EMBEDDINGS_DIR, "nli_train_labels.npy")),
                              dtype=torch.float32)
        s1_comp = batch_compress(compressor, s1_raw)
        s2_comp = batch_compress(compressor, s2_raw)

    probe_loader = DataLoader(TensorDataset(s1_comp, s2_comp, labels),
                              batch_size=BATCH_SIZE, shuffle=True)
    head = get_task_head("nli", DIM).to(DEVICE)
    opt  = torch.optim.Adam(head.parameters(), lr=3e-4)

    for _ in range(5):
        head.train()
        for batch in probe_loader:
            z1, z2, labs = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            opt.zero_grad()
            loss = head.loss(z1, z2, labs)
            loss.backward()
            opt.step()

    nli_acc = evaluate_nli(compressor, head)
    return nli_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 7])
    args = parser.parse_args()

    seeds = args.seeds
    print(f"\n{'='*60}")
    print(f"  MULTI-SEED EVALUATION — NLI-Aware AE dim=256")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*60}\n")

    # ── Task-Aware (NLI) ──────────────────────────────────────────
    aware_scores = []
    for seed in seeds:
        print(f"\n--- Seed {seed}: NLI-Aware Autoencoder ---")
        acc = train_and_eval_aware(seed)
        aware_scores.append(acc)
        print(f"  => NLI accuracy = {acc:.4f}")

    # ── Task-Agnostic ─────────────────────────────────────────────
    agnostic_scores = []
    for seed in seeds:
        print(f"\n--- Seed {seed}: Agnostic Autoencoder ---")
        acc = train_and_eval_agnostic(seed)
        agnostic_scores.append(acc)
        print(f"  => NLI accuracy = {acc:.4f}")

    # ── Summary ───────────────────────────────────────────────────
    aware_mean = np.mean(aware_scores)
    aware_std  = np.std(aware_scores)
    agnostic_mean = np.mean(agnostic_scores)
    agnostic_std  = np.std(agnostic_scores)
    delta_mean    = aware_mean - agnostic_mean

    print(f"\n{'='*60}")
    print(f"  MULTI-SEED RESULTS (NLI Accuracy, AE dim=256)")
    print(f"{'='*60}")
    print(f"  Task-Aware  (NLI):  {aware_mean:.4f} ± {aware_std:.4f}  {aware_scores}")
    print(f"  Task-Agnostic:      {agnostic_mean:.4f} ± {agnostic_std:.4f}  {agnostic_scores}")
    print(f"  Delta (aware-agn):  {delta_mean:+.4f}")
    print(f"{'='*60}\n")

    # Save results
    result = {
        "config": {"method": "autoencoder", "dim": 256, "task": "nli",
                   "epochs": NUM_EPOCHS, "seeds": seeds},
        "task_aware_nli": {
            "scores": aware_scores,
            "mean": round(aware_mean, 4),
            "std": round(aware_std, 4),
        },
        "task_agnostic": {
            "scores": agnostic_scores,
            "mean": round(agnostic_mean, 4),
            "std": round(agnostic_std, 4),
        },
        "delta_mean": round(delta_mean, 4),
    }

    out_path = os.path.join(METRICS_DIR, "multi_seed_nli_confidence.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
