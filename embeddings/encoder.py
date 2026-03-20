"""
embeddings/encoder.py
Wraps a pretrained sentence transformer to generate embeddings.
Supports batch encoding and caching to disk.
"""

import os
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENCODER_MODEL, EMBEDDING_DIM, EMBEDDINGS_DIR


class SentenceEncoder:
    """
    Wrapper around a pretrained SentenceTransformer model.

    Usage:
        encoder = SentenceEncoder()
        embeddings = encoder.encode(["Hello world", "Another sentence"])
    """

    def __init__(self, model_name: str = ENCODER_MODEL, device: str = None):
        self.model_name = model_name
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[Encoder] Loading model: {model_name}")
        print(f"[Encoder] Device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        print(f"[Encoder] Embedding dimension: {self.embedding_dim}")

    def encode(
        self,
        sentences: list,
        batch_size: int = 64,
        show_progress: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of sentences into embeddings.

        Args:
            sentences     : List of strings
            batch_size    : Number of sentences per batch
            show_progress : Show tqdm progress bar
            normalize     : L2-normalize embeddings

        Returns:
            np.ndarray of shape (N, embedding_dim)
        """
        embeddings = self.model.encode(
            sentences,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def encode_pairs(
        self,
        sentences1: list,
        sentences2: list,
        batch_size: int = 64,
        normalize: bool = False,
    ):
        """
        Encode two lists of sentences (for pair tasks like STS / NLI).

        Returns:
            Tuple of (embeddings1, embeddings2), each np.ndarray of shape (N, D)
        """
        print("[Encoder] Encoding sentence1 list...")
        emb1 = self.encode(sentences1, batch_size=batch_size, normalize=normalize)

        print("[Encoder] Encoding sentence2 list...")
        emb2 = self.encode(sentences2, batch_size=batch_size, normalize=normalize)

        return emb1, emb2

    def save_embeddings(self, embeddings: np.ndarray, filename: str):
        """Save embeddings array to disk as .npy file."""
        path = os.path.join(EMBEDDINGS_DIR, filename)
        np.save(path, embeddings)
        print(f"[Encoder] Saved embeddings to: {path}  shape={embeddings.shape}")

    def load_embeddings(self, filename: str) -> np.ndarray:
        """Load embeddings from disk."""
        path = os.path.join(EMBEDDINGS_DIR, filename)
        embeddings = np.load(path)
        print(f"[Encoder] Loaded embeddings from: {path}  shape={embeddings.shape}")
        return embeddings

    def encode_and_cache(
        self,
        sentences: list,
        cache_name: str,
        force_recompute: bool = False,
        **encode_kwargs,
    ) -> np.ndarray:
        """
        Encode sentences, but use cached result if it already exists.

        Args:
            sentences       : List of strings
            cache_name      : Filename (without extension) for the cache file
            force_recompute : If True, ignore existing cache and recompute

        Returns:
            np.ndarray of embeddings
        """
        cache_path = os.path.join(EMBEDDINGS_DIR, f"{cache_name}.npy")

        if not force_recompute and os.path.exists(cache_path):
            print(f"[Encoder] Cache hit — loading from {cache_path}")
            return np.load(cache_path)

        print(f"[Encoder] Cache miss — computing embeddings for '{cache_name}'...")
        embeddings = self.encode(sentences, **encode_kwargs)
        np.save(cache_path, embeddings)
        print(f"[Encoder] Cached to: {cache_path}")
        return embeddings


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    encoder = SentenceEncoder()

    sample_sentences = [
        "The cat sat on the mat.",
        "A dog is running in the park.",
        "Natural language processing is a subfield of AI.",
        "Deep learning has revolutionized NLP tasks.",
    ]

    embs = encoder.encode(sample_sentences, show_progress=True)
    print(f"\nEmbedding shape: {embs.shape}")           # (4, 768)
    print(f"Sample norm: {np.linalg.norm(embs[0]):.4f}")

    # Test caching
    cached = encoder.encode_and_cache(sample_sentences, cache_name="test_sample")
    print(f"Cached embeddings match: {np.allclose(embs, cached)}")
