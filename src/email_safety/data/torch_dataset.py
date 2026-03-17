from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class FusionDataset(Dataset):
    def __init__(
        self,
        texts,
        structured_features: np.ndarray,
        tokenizer,
        max_length: int,
        labels: Optional[np.ndarray] = None,
    ):
        self.texts = list(texts)
        self.structured_features = structured_features
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "structured_features": torch.tensor(self.structured_features[idx], dtype=torch.float),
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
