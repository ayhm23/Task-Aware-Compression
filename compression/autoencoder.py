"""
compression/autoencoder.py
─────────────────────────────────────────────────────────────────────────────
Autoencoder-based Compression.

Architecture:
    Encoder: 768 → 512 → output_dim   (compress)
    Decoder: output_dim → 512 → 768   (reconstruct)

The bottleneck (output_dim) IS the compressed embedding.

Two variants:
  1. Autoencoder       — standard AE with ReLU activations
  2. VariationalAE     — VAE with reparameterisation trick (bonus variant)

Loss:
  - task_agnostic : reconstruction loss (MSE) only
  - task_aware    : reconstruction loss + task loss (weighted sum)
─────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from compression.base import BaseCompressor
from config import EMBEDDING_DIM, AE_HIDDEN_DIM, AE_DROPOUT


class Autoencoder(BaseCompressor):
    """
    Standard Autoencoder.

    Encoder: input_dim → hidden_dim → output_dim
    Decoder: output_dim → hidden_dim → input_dim

    The compressed representation is the encoder's output (bottleneck).
    """

    def __init__(
        self,
        input_dim:  int   = EMBEDDING_DIM,
        output_dim: int   = 128,
        hidden_dim: int   = AE_HIDDEN_DIM,
        dropout:    float = AE_DROPOUT,
    ):
        super().__init__(input_dim, output_dim)
        self.hidden_dim = hidden_dim

        # ── Encoder ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress: returns bottleneck (compressed) representation.

        Args:
            x : (batch_size, input_dim)
        Returns:
            z : (batch_size, output_dim)
        """
        return self.encoder(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct original embedding from compressed form.

        Args:
            z : (batch_size, output_dim)
        Returns:
            x_hat : (batch_size, input_dim)
        """
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full forward pass + MSE reconstruction loss.
        Primary loss for task-agnostic training.
        """
        z     = self.forward(x)
        x_hat = self.reconstruct(z)
        return F.mse_loss(x_hat, x)

    def full_forward(self, x: torch.Tensor):
        """
        Returns both compressed z and reconstructed x_hat.
        Useful during task-aware training where we need both.
        """
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class DeepAutoencoder(BaseCompressor):
    """
    Deeper Autoencoder with 3 hidden layers.

    Encoder: 768 → 512 → 256 → output_dim
    Decoder: output_dim → 256 → 512 → 768

    More expressive than standard AE — better for very small output dims (32, 64).
    """

    def __init__(
        self,
        input_dim:  int   = EMBEDDING_DIM,
        output_dim: int   = 64,
        dropout:    float = AE_DROPOUT,
    ):
        super().__init__(input_dim, output_dim)

        # ── Encoder ──────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, output_dim),
        )

        # ── Decoder ──────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(512, input_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        z     = self.forward(x)
        x_hat = self.reconstruct(z)
        return F.mse_loss(x_hat, x)

    def full_forward(self, x: torch.Tensor):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    batch = torch.randn(32, 768)

    print("── Standard Autoencoder ──")
    for dim in [32, 64, 128, 256]:
        model = Autoencoder(input_dim=768, output_dim=dim)
        z     = model(batch)
        loss  = model.reconstruction_loss(batch)
        print(f"  dim={dim:>3}  z={tuple(z.shape)}  "
              f"recon_loss={loss.item():.4f}  {model}")

    print("\n── Deep Autoencoder ──")
    for dim in [32, 64, 128, 256]:
        model = DeepAutoencoder(input_dim=768, output_dim=dim)
        z     = model(batch)
        loss  = model.reconstruction_loss(batch)
        print(f"  dim={dim:>3}  z={tuple(z.shape)}  "
              f"recon_loss={loss.item():.4f}  {model}")
