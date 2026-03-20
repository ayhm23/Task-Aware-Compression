"""
config.py — Central configuration for Task-Aware Compression project.
All hyperparameters and paths are defined here.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
EMBEDDINGS_DIR  = os.path.join(BASE_DIR, "embeddings", "cache")
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
PLOTS_DIR       = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR     = os.path.join(RESULTS_DIR, "metrics")

for _dir in [EMBEDDINGS_DIR, PLOTS_DIR, METRICS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ─── Encoder ──────────────────────────────────────────────────────────────────
ENCODER_MODEL       = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM       = 768          # Output dimension of the base encoder

# ─── Compression ──────────────────────────────────────────────────────────────
# Target compressed dimensions to experiment with
COMPRESSION_DIMS    = [32, 64, 128, 256]

# ─── Tasks ────────────────────────────────────────────────────────────────────
TASKS = {
    "sts": {
        "dataset": "stsb_multi_mt",       # HuggingFace dataset name
        "dataset_name": "en",
        "metric": "spearman",
        "num_labels": 1,
        "description": "Semantic Textual Similarity"
    },
    "nli": {
        "dataset": "snli",
        "dataset_name": None,
        "metric": "accuracy",
        "num_labels": 3,
        "description": "Natural Language Inference"
    },
    "classification": {
        "dataset": "sst2",
        "dataset_name": None,
        "metric": "accuracy",
        "num_labels": 2,
        "description": "Sentiment Classification"
    }
}

# ─── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE          = 64
NUM_EPOCHS          = 10
LEARNING_RATE       = 3e-4
WEIGHT_DECAY        = 1e-5
SEED                = 42

# ─── Compression Methods ──────────────────────────────────────────────────────
COMPRESSION_METHODS = ["linear", "autoencoder", "distillation"]

# ─── Autoencoder specific ─────────────────────────────────────────────────────
AE_HIDDEN_DIM       = 512          # Intermediate hidden layer in autoencoder
AE_DROPOUT          = 0.1

# ─── Distillation specific ────────────────────────────────────────────────────
DISTILL_TEMPERATURE = 4.0
DISTILL_ALPHA       = 0.7          # Weight for distillation loss vs task loss
