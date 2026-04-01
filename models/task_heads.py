"""
models/task_heads.py
─────────────────────────────────────────────────────────────────────────────
Task-specific classifier heads that sit on top of compressed embeddings.

For each task, a small MLP takes the compressed embedding(s) as input
and outputs task predictions.

    STS:            [z1 ; z2 ; |z1-z2| ; z1*z2]  →  similarity score (float)
    NLI:            [z1 ; z2 ; |z1-z2| ; z1*z2]  →  3-class logits
    Classification: [z]                           →  2-class logits
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TASKS


class STSHead(nn.Module):
    """
    Regression head for Semantic Textual Similarity.

    Input:  concatenation of [z1, z2, |z1-z2|, z1*z2]  → 4 × compressed_dim
    Output: scalar similarity score in [0, 1]
    """

    def __init__(self, compressed_dim: int, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        input_dim = 4 * compressed_dim   # [z1 ; z2 ; |z1-z2| ; z1*z2]

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),                 # output in [0, 1]
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1, z2 : compressed embeddings (batch_size, compressed_dim)
        Returns:
            scores : (batch_size,) — similarity scores in [0, 1]
        """
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.net(combined).squeeze(-1)

    def loss(self, z1, z2, labels):
        """MSE loss for regression."""
        preds = self.forward(z1, z2)
        return F.mse_loss(preds, labels.float())


class NLIHead(nn.Module):
    """
    3-class classification head for Natural Language Inference.

    Input:  [z1 ; z2 ; |z1-z2| ; z1*z2]  → 4 × compressed_dim
    Output: logits over {entailment, neutral, contradiction}
    """

    def __init__(self, compressed_dim: int, hidden_dim: int = 256,
                 num_labels: int = 3, dropout: float = 0.1):
        super().__init__()
        input_dim = 4 * compressed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits : (batch_size, num_labels)
        """
        combined = torch.cat([z1, z2, torch.abs(z1 - z2), z1 * z2], dim=-1)
        return self.net(combined)

    def loss(self, z1, z2, labels):
        """Cross-entropy loss."""
        logits = self.forward(z1, z2)
        return F.cross_entropy(logits, labels.long())


class ClassificationHead(nn.Module):
    """
    Binary classification head for sentiment (SST-2).

    Input:  z  → compressed_dim
    Output: logits over {negative, positive}
    """

    def __init__(self, compressed_dim: int, hidden_dim: int = 128,
                 num_labels: int = 2, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(compressed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            logits : (batch_size, num_labels)
        """
        return self.net(z)

    def loss(self, z, labels):
        """Cross-entropy loss."""
        logits = self.forward(z)
        return F.cross_entropy(logits, labels.long())


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_task_head(task: str, compressed_dim: int) -> nn.Module:
    """
    Returns the appropriate task head for a given task.

    Args:
        task           : 'sts', 'nli', or 'classification'
        compressed_dim : output dimension of the compressor

    Returns:
        nn.Module task head
    """
    heads = {
        "sts":            STSHead,
        "nli":            NLIHead,
        "classification": ClassificationHead,
    }

    if task not in heads:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(heads.keys())}")

    return heads[task](compressed_dim=compressed_dim)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch = 16

    print("── Task Heads ──")
    for dim in [32, 64, 128, 256]:

        # STS
        head   = STSHead(compressed_dim=dim)
        z1, z2 = torch.randn(batch, dim), torch.randn(batch, dim)
        labels = torch.rand(batch)
        loss   = head.loss(z1, z2, labels)
        print(f"  STS   dim={dim}  loss={loss.item():.4f}  "
              f"params={sum(p.numel() for p in head.parameters()):,}")

        # NLI
        head   = NLIHead(compressed_dim=dim)
        labels = torch.randint(0, 3, (batch,))
        loss   = head.loss(z1, z2, labels)
        print(f"  NLI   dim={dim}  loss={loss.item():.4f}  "
              f"params={sum(p.numel() for p in head.parameters()):,}")

        # Classification
        head   = ClassificationHead(compressed_dim=dim)
        z      = torch.randn(batch, dim)
        labels = torch.randint(0, 2, (batch,))
        loss   = head.loss(z, labels)
        print(f"  CLS   dim={dim}  loss={loss.item():.4f}  "
              f"params={sum(p.numel() for p in head.parameters()):,}")
        print()
