#!/usr/bin/env python
"""Run local-path-only domain adaptive pretraining with masked language modeling.
Inputs: local model dir and MLM corpus. Outputs: DAPT checkpoints, final model, and training state."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.utils.logger import get_logger

LOGGER = get_logger(__name__)


class MLMDataset(Dataset):
    def __init__(self, input_ids_chunks):
        self.input_ids_chunks = input_ids_chunks

    def __len__(self):
        return len(self.input_ids_chunks)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids_chunks[idx]}


def parse_args():
    parser = argparse.ArgumentParser(description="Run domain adaptive pretraining with MLM")
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--corpus-txt", type=str, default="data/mlm_corpus/mail_corpus.txt")
    parser.add_argument("--output-dir", type=str, default="outputs/dapt_multilingual_bert")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--mlm-probability", type=float, default=0.15)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_dataset(tokenizer, corpus_path: str, block_size: int):
    text = Path(corpus_path).read_text(encoding="utf-8")
    tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    chunks = []
    for start in range(0, len(tokens) - block_size + 1, block_size):
        chunk = tokens[start : start + block_size]
        chunks.append(torch.tensor(chunk, dtype=torch.long))
    return MLMDataset(chunks)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _select_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir, local_files_only=True).to(device)
    dataset = _build_dataset(tokenizer, args.corpus_txt, args.block_size)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    state_path = output_dir / "trainer_state.pt"
    start_epoch = 0
    training_log = []
    if args.resume and state_path.exists():
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0))
        training_log = state.get("training_log", [])

    run_config = {
        "model_dir": args.model_dir,
        "corpus_txt": args.corpus_txt,
        "output_dir": args.output_dir,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "mlm_probability": args.mlm_probability,
        "device": device.type,
        "dataset_size": len(dataset),
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    training_log_path = output_dir / "training_log.jsonl"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(loader))
        perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
        entry = {"epoch": epoch + 1, "avg_loss": avg_loss, "perplexity": perplexity}
        training_log.append(entry)
        with training_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_log": training_log,
            },
            state_path,
        )
        with (output_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump({"last_epoch": epoch + 1, "training_log": training_log}, f, ensure_ascii=False, indent=2)
        LOGGER.info("Epoch %d/%d avg_loss=%.4f perplexity=%.4f", epoch + 1, args.epochs, avg_loss, perplexity)

    final_dir = output_dir / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
