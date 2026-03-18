from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LLMGuidedTextDataset(Dataset):
    def __init__(
        self,
        texts,
        tokenizer,
        max_length: int,
        labels: Optional[np.ndarray] = None,
        risk_hints: Optional[list[str]] = None,
        soft_targets: Optional[np.ndarray] = None,
    ):
        self.texts = list(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        self.risk_hints = list(risk_hints) if risk_hints is not None else None
        self.soft_targets = soft_targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_pair = None
        if self.risk_hints is not None:
            hint = self.risk_hints[idx]
            text_pair = hint if hint else None
        encoded = self.tokenizer(
            self.texts[idx],
            text_pair=text_pair,
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
        if self.soft_targets is not None:
            item["soft_targets"] = torch.tensor(self.soft_targets[idx], dtype=torch.float)
        return item
