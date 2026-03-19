#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
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
from email_safety.utils.config import load_config
from email_safety.utils.logger import get_logger
from email_safety.utils.seed import set_seed

LOGGER = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train text-only Transformer on gold_v2")
    parser.add_argument("--config", type=str, default="configs/text_transformer.yaml")
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


def _resolve_batch_size(device: torch.device, configured_batch_size: int) -> int:
    if configured_batch_size and configured_batch_size > 0:
        return configured_batch_size
    return 16 if device.type == "cuda" else 8


def _prepare_dataframe(path: str, label_column: str, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")
    if id_column not in df.columns:
        raise ValueError(f"Missing id column: {id_column}")
    df[id_column] = df[id_column].map(_safe_text)
    df[label_column] = df[label_column].map(_safe_text)
    df = df[df[id_column].ne("") & df[label_column].ne("")].copy()
    df = df.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)
    return df


def _stratified_split(df: pd.DataFrame, label_column: str, seed: int):
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df[label_column],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df[label_column],
    )
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _save_split_files(train_df, valid_df, test_df, processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    valid_df.to_csv(processed_dir / "valid.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)


def _evaluate_model(model, loader, device, id_to_label):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend([id_to_label[idx] for idx in pred])
            y_true.extend([id_to_label[idx] for idx in batch["labels"].cpu().numpy().tolist()])
    return evaluate_multiclass(y_true, y_pred)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"].get("seed", 42))

    data_cfg = cfg["data"]
    preprocess_cfg = cfg["preprocess"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    fields_cfg = cfg["fields"]

    output_dir = Path(cfg["project"]["output_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_dataframe(
        path=data_cfg["input_csv"],
        label_column=data_cfg["label_column"],
        id_column=data_cfg["id_column"],
    )
    train_df, valid_df, test_df = _stratified_split(df, data_cfg["label_column"], cfg["project"].get("seed", 42))
    _save_split_files(train_df, valid_df, test_df, processed_dir)

    labels = sorted(df[data_cfg["label_column"]].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_to_id, f, ensure_ascii=False, indent=2)

    run_cfg = copy.deepcopy(cfg)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    train_texts = build_concat_text(train_df, fields_cfg["text_fields"], **preprocess_cfg)
    valid_texts = build_concat_text(valid_df, fields_cfg["text_fields"], **preprocess_cfg)
    test_texts = build_concat_text(test_df, fields_cfg["text_fields"], **preprocess_cfg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])
    except OSError as exc:
        raise RuntimeError(
            "Failed to load tokenizer/model config. "
            "Please check network access to HuggingFace, or set "
            "`model.pretrained_model_name` to a local model directory."
        ) from exc
    train_labels = train_df[data_cfg["label_column"]].map(label_to_id).values
    valid_labels = valid_df[data_cfg["label_column"]].map(label_to_id).values
    test_labels = test_df[data_cfg["label_column"]].map(label_to_id).values

    train_ds = TextClassificationDataset(train_texts, tokenizer, model_cfg["max_length"], train_labels)
    valid_ds = TextClassificationDataset(valid_texts, tokenizer, model_cfg["max_length"], valid_labels)
    test_ds = TextClassificationDataset(test_texts, tokenizer, model_cfg["max_length"], test_labels)

    device = _select_device()
    batch_size = _resolve_batch_size(device, int(train_cfg.get("batch_size", 8)))
    LOGGER.info("Using device=%s batch_size=%d", device.type, batch_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    try:
        model = TextOnlyTransformerClassifier(
            pretrained_model_name=model_cfg["pretrained_model_name"],
            num_labels=model_cfg["num_labels"],
        ).to(device)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load pretrained weights. "
            "Please check network access to HuggingFace, or set "
            "`model.pretrained_model_name` to a local model directory."
        ) from exc
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    best_valid_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pt"
    train_logs = []

    for epoch in range(1, int(train_cfg.get("epochs", 3)) + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = criterion(logits, batch["labels"].to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        valid_result = _evaluate_model(model, valid_loader, device, id_to_label)
        train_logs.append(
            {
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_f1": valid_result["macro_f1"],
            }
        )
        LOGGER.info(
            "Epoch %d/%d train_loss=%.4f valid_acc=%.4f valid_macro_f1=%.4f",
            epoch,
            int(train_cfg.get("epochs", 3)),
            avg_train_loss,
            valid_result["accuracy"],
            valid_result["macro_f1"],
        )

        if valid_result["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_result["macro_f1"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "pretrained_model_name": model_cfg["pretrained_model_name"],
                    "max_length": model_cfg["max_length"],
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "config": cfg,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    valid_result = _evaluate_model(model, valid_loader, device, id_to_label)
    test_result = _evaluate_model(model, test_loader, device, id_to_label)

    dump_eval_results(test_result, output_dir)

    metrics_summary = {
        "device": device.type,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "valid": {
            "accuracy": valid_result["accuracy"],
            "macro_precision": valid_result["macro_precision"],
            "macro_recall": valid_result["macro_recall"],
            "macro_f1": valid_result["macro_f1"],
        },
        "test": {
            "accuracy": test_result["accuracy"],
            "macro_precision": test_result["macro_precision"],
            "macro_recall": test_result["macro_recall"],
            "macro_f1": test_result["macro_f1"],
        },
        "train_logs": train_logs,
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame(
        [
            {
                "experiment": "text_only_bert_base_chinese",
                "model_name": model_cfg["pretrained_model_name"],
                "best_epoch": best_epoch,
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "test_rows": len(test_df),
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_precision": valid_result["macro_precision"],
                "valid_macro_recall": valid_result["macro_recall"],
                "valid_macro_f1": valid_result["macro_f1"],
                "test_accuracy": test_result["accuracy"],
                "test_macro_precision": test_result["macro_precision"],
                "test_macro_recall": test_result["macro_recall"],
                "test_macro_f1": test_result["macro_f1"],
            }
        ]
    ).to_csv(output_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
