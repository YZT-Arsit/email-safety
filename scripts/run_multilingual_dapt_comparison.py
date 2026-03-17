#!/usr/bin/env python
"""Compare local multilingual BERT and DAPT multilingual BERT on Gold v2 text classification.
Inputs: Gold v2 CSV and local model directories. Outputs: split files, per-run metrics, and results summary."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.text_dataset import TextClassificationDataset
from email_safety.evaluation.metrics import dump_eval_results, evaluate_multiclass
from email_safety.models.text_transformer import TextOnlyTransformerClassifier
from email_safety.preprocessing.text_clean import build_concat_text
from email_safety.utils.logger import get_logger
from email_safety.utils.seed import set_seed

LOGGER = get_logger(__name__)


EXPERIMENTS = [
    ("multilingual_bert_base", False),
    ("multilingual_bert_dapt", True),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Compare local multilingual BERT before/after DAPT on gold_v2")
    parser.add_argument("--gold-csv", type=str, default="data/annotation/gold/gold_v2.csv")
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--dapt-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/multilingual_dapt_comparison")
    parser.add_argument("--processed-dir", type=str, default="data/processed/multilingual_dapt_comparison")
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _prepare(path: str, id_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in [id_column, label_column, "subject", "content", "doccontent"]:
        if col not in df.columns:
            df[col] = ""
    df[id_column] = df[id_column].map(_safe_text)
    df[label_column] = df[label_column].map(_safe_text)
    for col in ["subject", "content", "doccontent"]:
        df[col] = df[col].map(_safe_text)
    return df[df[id_column].ne("") & df[label_column].ne("")].drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)


def _split(df: pd.DataFrame, label_column: str, seed: int):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[label_column])
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df[label_column])
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _save_split(train_df, valid_df, test_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def _evaluate(model, loader, device, id_to_label):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend([id_to_label[i] for i in preds])
            y_true.extend([id_to_label[i] for i in batch["labels"].cpu().numpy().tolist()])
    return evaluate_multiclass(y_true, y_pred)


def _train_one(model_dir: str, experiment_name: str, train_df, valid_df, test_df, args, output_root: Path):
    labels = sorted(pd.concat([train_df[args.label_column], valid_df[args.label_column], test_df[args.label_column]]).unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    train_texts = build_concat_text(train_df, ["subject", "content", "doccontent"], lowercase=True, remove_urls=False, max_text_length=5000)
    valid_texts = build_concat_text(valid_df, ["subject", "content", "doccontent"], lowercase=True, remove_urls=False, max_text_length=5000)
    test_texts = build_concat_text(test_df, ["subject", "content", "doccontent"], lowercase=True, remove_urls=False, max_text_length=5000)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_ds = TextClassificationDataset(train_texts, tokenizer, args.max_length, train_df[args.label_column].map(label_to_id).values)
    valid_ds = TextClassificationDataset(valid_texts, tokenizer, args.max_length, valid_df[args.label_column].map(label_to_id).values)
    test_ds = TextClassificationDataset(test_texts, tokenizer, args.max_length, test_df[args.label_column].map(label_to_id).values)

    loader_kwargs = {"batch_size": args.batch_size, "num_workers": 0}
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    device = _select_device()
    model = TextOnlyTransformerClassifier(model_dir, num_labels=len(label_to_id)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_valid_f1 = -1.0
    best_epoch = 0
    metrics_dir = output_root / experiment_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = metrics_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        valid_result = _evaluate(model, valid_loader, device, id_to_label)
        LOGGER.info(
            "%s epoch=%d train_loss=%.4f valid_macro_f1=%.4f",
            experiment_name,
            epoch,
            running_loss / max(1, len(train_loader)),
            valid_result["macro_f1"],
        )
        if valid_result["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_result["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_dir": model_dir,
                    "pretrained_model_name": model_dir,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "max_length": args.max_length,
                },
                ckpt_path,
            )

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    valid_result = _evaluate(model, valid_loader, device, id_to_label)
    test_result = _evaluate(model, test_loader, device, id_to_label)

    dump_eval_results(test_result, metrics_dir)
    with (metrics_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "valid": {k: valid_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
                "test": {k: test_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
                "base_checkpoint": model_dir,
                "whether_dapt": "dapt" in experiment_name,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return best_epoch, valid_result, test_result


def main():
    args = parse_args()
    set_seed(args.seed)
    df = _prepare(args.gold_csv, args.id_column, args.label_column)
    train_df, valid_df, test_df = _split(df, args.label_column, args.seed)
    split_dir = Path(args.processed_dir)
    _save_split(train_df, valid_df, test_df, split_dir)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for experiment_name, whether_dapt in EXPERIMENTS:
        model_dir = args.dapt_model_dir if whether_dapt else args.base_model_dir
        best_epoch, valid_result, test_result = _train_one(model_dir, experiment_name, train_df, valid_df, test_df, args, output_root)
        records.append(
            {
                "experiment": experiment_name,
                "model_name": Path(model_dir).name,
                "base_checkpoint": args.base_model_dir,
                "whether_dapt": whether_dapt,
                "dapt_checkpoint_path": args.dapt_model_dir if whether_dapt else "",
                "best_epoch": best_epoch,
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_precision": valid_result["macro_precision"],
                "valid_macro_recall": valid_result["macro_recall"],
                "valid_macro_f1": valid_result["macro_f1"],
                "test_accuracy": test_result["accuracy"],
                "test_macro_precision": test_result["macro_precision"],
                "test_macro_recall": test_result["macro_recall"],
                "test_macro_f1": test_result["macro_f1"],
            }
        )

    pd.DataFrame(records).to_csv(output_root / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
