"""
compression/base.py
Abstract base class that all compression methods inherit from.
Enforces a consistent interface across Linear, Autoencoder, Distillation.
"""

import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EMBEDDING_DIM, COMPRESSION_DIMS


class BaseCompressor(nn.Module, ABC):
    """
    Abstract base for all compression modules.

    Every compressor must implement:
        - forward(x)        → compressed embedding
        - compress(x)       → alias for forward (inference)
        - get_output_dim()  → returns compressed dimension size
    """

    def __init__(self, input_dim: int = EMBEDDING_DIM, output_dim: int = 128):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress input embeddings.

        Args:
            x : Tensor of shape (batch_size, input_dim)

        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        pass

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward — used during inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def get_output_dim(self) -> int:
        return self.output_dim

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"[Compressor] Saved to {path}")

    def load(self, path: str):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location="cpu"))
        print(f"[Compressor] Loaded from {path}")
        return self

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}, "
            f"params={self.count_parameters():,})"
        )
