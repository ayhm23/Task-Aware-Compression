"""
data/dataset_loader.py
Handles loading and preprocessing of all datasets used in the project.
Datasets: STS-B (similarity), SNLI (NLI), SST-2 (classification)
"""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch
from config import TASKS, BATCH_SIZE, SEED


# ─── Generic Sentence Dataset ─────────────────────────────────────────────────

class SentencePairDataset(Dataset):
    """
    Generic dataset for sentence pairs with labels.
    Works for STS (float labels) and NLI / classification (int labels).
    """

    def __init__(self, sentences1, sentences2, labels):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels     = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sentence1": self.sentences1[idx],
            "sentence2": self.sentences2[idx],
            "label":     self.labels[idx],
        }


class SingleSentenceDataset(Dataset):
    """Dataset for single-sentence tasks like SST-2."""

    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "sentence": self.sentences[idx],
            "label":    self.labels[idx],
        }


# ─── Task-Specific Loaders ────────────────────────────────────────────────────

def load_sts(split="train"):
    """
    Load STS-B dataset.
    Labels are float similarity scores in [0, 5], normalized to [0, 1].
    Note: STS-B uses 'dev' instead of 'validation'.
    """
    if split == "validation":
        split = "dev"
    dataset = load_dataset("stsb_multi_mt", "en", split=split)

    sentences1 = dataset["sentence1"]
    sentences2 = dataset["sentence2"]
    labels     = [score / 5.0 for score in dataset["similarity_score"]]

    return SentencePairDataset(sentences1, sentences2, labels)


def load_nli(split="train", max_samples=50000):
    """
    Load SNLI dataset.
    Labels: 0=entailment, 1=neutral, 2=contradiction. Skips -1 (no label).
    """
    dataset = load_dataset("snli", split=split)

    # Filter out unlabeled examples
    dataset = dataset.filter(lambda x: x["label"] != -1)

    if max_samples and len(dataset) > max_samples:
        dataset = dataset.shuffle(seed=SEED).select(range(max_samples))

    sentences1 = dataset["premise"]
    sentences2 = dataset["hypothesis"]
    labels     = dataset["label"]

    return SentencePairDataset(sentences1, sentences2, labels)


def load_sst2(split="train"):
    """
    Load SST-2 sentiment classification dataset.
    Labels: 0=negative, 1=positive.
    """
    dataset = load_dataset("glue", "sst2", split=split)

    sentences = dataset["sentence"]
    labels    = dataset["label"]

    return SingleSentenceDataset(sentences, labels)


# ─── Unified Loader ───────────────────────────────────────────────────────────

def get_dataset(task: str, split: str = "train"):
    """
    Returns the appropriate dataset object for a given task name and split.

    Args:
        task  : One of 'sts', 'nli', 'classification'
        split : 'train', 'validation', or 'test'

    Returns:
        A torch Dataset object.
    """
    loaders = {
        "sts":            load_sts,
        "nli":            load_nli,
        "classification": load_sst2,
    }

    if task not in loaders:
        raise ValueError(f"Unknown task '{task}'. Choose from: {list(loaders.keys())}")

    print(f"[DataLoader] Loading '{task}' — split: {split}")
    return loaders[task](split=split)


def get_dataloader(task: str, split: str = "train", batch_size: int = BATCH_SIZE, shuffle: bool = True):
    """
    Returns a DataLoader for the specified task and split.
    """
    dataset = get_dataset(task, split)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)


# ─── Quick Test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for task in ["sts", "nli", "classification"]:
        ds = get_dataset(task, split="validation")
        print(f"  {task}: {len(ds)} examples | sample: {ds[0]}")
