"""
scripts/generate_embeddings.py
-----------------------------------------------------------------------
Step 1: Load each dataset split and generate sentence embeddings
        using the pretrained encoder. Saves them to embeddings/cache/.

Run:
    python scripts/generate_embeddings.py
    python scripts/generate_embeddings.py --task sts --split train
-----------------------------------------------------------------------
"""

import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TASKS, ENCODER_MODEL
from data.dataset_loader import get_dataset
from embeddings.encoder import SentenceEncoder


def generate_for_task(encoder: SentenceEncoder, task: str, split: str):
    """
    Load dataset for a task/split, encode sentences, and save to cache.
    """
    print(f"\n{'='*60}")
    print(f"  Task: {task.upper()} | Split: {split}")
    print(f"{'='*60}")

    dataset = get_dataset(task, split=split)

    # ── Collect sentences ────────────────────────────────────────
    if task in ("sts", "nli"):
        sentences1 = [dataset[i]["sentence1"] for i in range(len(dataset))]
        sentences2 = [dataset[i]["sentence2"] for i in range(len(dataset))]
        labels     = [dataset[i]["label"]     for i in range(len(dataset))]

        emb1 = encoder.encode_and_cache(
            sentences1,
            cache_name=f"{task}_{split}_s1",
            show_progress=True,
        )
        emb2 = encoder.encode_and_cache(
            sentences2,
            cache_name=f"{task}_{split}_s2",
            show_progress=True,
        )
        np.save(
            os.path.join(encoder.model.embedding_dim if False else
                         __import__("config").EMBEDDINGS_DIR,
                         f"{task}_{split}_labels.npy"),
            np.array(labels)
        )
        print(f"  ✓ s1: {emb1.shape}  s2: {emb2.shape}  labels: {len(labels)}")

    else:  # classification (single sentence)
        sentences = [dataset[i]["sentence"] for i in range(len(dataset))]
        labels    = [dataset[i]["label"]    for i in range(len(dataset))]

        embs = encoder.encode_and_cache(
            sentences,
            cache_name=f"{task}_{split}_s1",
            show_progress=True,
        )
        from config import EMBEDDINGS_DIR
        np.save(
            os.path.join(EMBEDDINGS_DIR, f"{task}_{split}_labels.npy"),
            np.array(labels)
        )
        print(f"  ✓ embeddings: {embs.shape}  labels: {len(labels)}")


def main():
    parser = argparse.ArgumentParser(description="Generate & cache sentence embeddings")
    parser.add_argument("--task",  type=str, default="all",
                        choices=["all"] + list(TASKS.keys()),
                        help="Which task to generate embeddings for")
    parser.add_argument("--split", type=str, default="all",
                        choices=["all", "train", "validation", "test"],
                        help="Which split to process")
    args = parser.parse_args()

    # ── Initialise encoder ──────────────────────────────────────
    encoder = SentenceEncoder(model_name=ENCODER_MODEL)

    tasks  = list(TASKS.keys()) if args.task  == "all" else [args.task]
    splits = ["train", "validation", "test"]  if args.split == "all" else [args.split]

    for task in tasks:
        for split in splits:
            try:
                generate_for_task(encoder, task, split)
            except Exception as e:
                print(f"  ✗ Failed [{task}/{split}]: {e}")

    print("\n✅ Embedding generation complete.")


if __name__ == "__main__":
    main()
