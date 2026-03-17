from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        max_length: int,
        labels: Optional[np.ndarray] = None,
        sample_weights: Optional[np.ndarray] = None,
    ):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.sample_weights = sample_weights

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        if self.sample_weights is not None:
            item["sample_weight"] = torch.tensor(float(self.sample_weights[idx]), dtype=torch.float)
        return item
