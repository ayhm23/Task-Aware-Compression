"""
compression/linear.py
─────────────────────────────────────────────────────────────────────────────
Linear Projection Compression.

Two variants:
  1. LinearCompressor        — single linear layer  (768 → d)
  2. LinearCompressorWithBN  — linear + batch norm  (more stable training)

Loss modes:
  - task_agnostic : MSE reconstruction loss (via a decoder projection)
  - task_aware    : task loss passed in from trainer (no change to architecture)

The same module is used for both modes — the difference is only in
how the training loss is computed (handled in train_compression.py).
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.base import BaseCompressor
from config import EMBEDDING_DIM


class LinearCompressor(BaseCompressor):
    """
    Single linear projection: 768 → output_dim

    The simplest possible compression — a learned weight matrix.
    No activation, no non-linearity.

    Parameters: input_dim × output_dim  (e.g. 768×64 = 49,152)
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, output_dim: int = 128):
        super().__init__(input_dim, output_dim)

        self.projection = nn.Linear(input_dim, output_dim, bias=True)

        # Decoder used only for task-agnostic (reconstruction) training
        self.decoder = nn.Linear(output_dim, input_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init for stable training."""
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (batch_size, input_dim)
        Returns:
            compressed : (batch_size, output_dim)
        """
        return self.projection(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original embedding from compressed form.
        Used during task-agnostic training.

        Args:
            z : (batch_size, output_dim)
        Returns:
            x_hat : (batch_size, input_dim)
        """
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        MSE loss between original and reconstructed embeddings.
        Used for task-agnostic training objective.
        """
        z     = self.forward(x)
        x_hat = self.reconstruct(z)
        return F.mse_loss(x_hat, x)


class LinearCompressorWithBN(BaseCompressor):
    """
    Linear projection + Batch Normalization + optional L2 normalisation.

    Slightly more powerful than plain linear — BN stabilises
    the compressed representation's distribution.

    Parameters: input_dim × output_dim + BN params
    """

    def __init__(
        self,
        input_dim:  int  = EMBEDDING_DIM,
        output_dim: int  = 128,
        normalize:  bool = True,
    ):
        super().__init__(input_dim, output_dim)
        self.normalize = normalize

        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        self.bn         = nn.BatchNorm1d(output_dim)
        self.decoder    = nn.Linear(output_dim, input_dim, bias=True)

        nn.init.xavier_uniform_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projection(x)
        z = self.bn(z)
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        return z

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.forward(x)
        x_hat = self.reconstruct(z)
        return F.mse_loss(x_hat, x)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch = torch.randn(32, 768)

    for dim in [32, 64, 128, 256]:
        model = LinearCompressor(input_dim=768, output_dim=dim)
        out   = model(batch)
        loss  = model.reconstruction_loss(batch)
        print(f"  LinearCompressor  → dim={dim:>3}  out={tuple(out.shape)}  "
              f"recon_loss={loss.item():.4f}  {model}")

    print()
    for dim in [32, 64, 128, 256]:
        model = LinearCompressorWithBN(input_dim=768, output_dim=dim)
        out   = model(batch)
        print(f"  LinearCompressorBN → dim={dim:>3}  out={tuple(out.shape)}  {model}")
