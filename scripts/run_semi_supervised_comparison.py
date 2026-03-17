#!/usr/bin/env python
"""Compare Gold-only and Gold-plus-Silver downstream training with local multilingual BERT backbones.
Inputs: Gold v2, trusted silver, and local model dirs. Outputs: per-run metrics and summary CSV."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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
    ("gold_v2_only__multilingual_bert", False, False),
    ("gold_v2_only__dapt_multilingual_bert", True, False),
    ("gold_v2_plus_silver__multilingual_bert", False, True),
    ("gold_v2_plus_silver__dapt_multilingual_bert", True, True),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semi-supervised comparison with local multilingual BERT / DAPT BERT")
    parser.add_argument("--gold-csv", type=str, default="data/annotation/gold/gold_v2.csv")
    parser.add_argument("--silver-csv", type=str, default="data/annotation/silver/consensus_trusted_silver.csv")
    parser.add_argument("--base-model-dir", type=str, required=True)
    parser.add_argument("--dapt-model-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/semi_supervised_comparison")
    parser.add_argument("--processed-dir", type=str, default="data/processed/semi_supervised_comparison")
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--silver-weight", type=float, default=0.5)
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


def _prepare_gold(path: str, id_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in [id_column, label_column, "subject", "content", "doccontent"]:
        if col not in df.columns:
            df[col] = ""
    for col in [id_column, label_column, "subject", "content", "doccontent"]:
        df[col] = df[col].map(_safe_text)
    return df[df[id_column].ne("") & df[label_column].ne("")].drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)


def _prepare_silver(path: str) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["id", "label", "subject_summary", "content_summary"])
    df = pd.read_csv(path).copy()
    for col in ["id", "label", "subject_summary", "content_summary"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_safe_text)
    return df[df["id"].ne("") & df["label"].ne("")].drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)


def _split(df: pd.DataFrame, label_column: str, seed: int):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[label_column])
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df[label_column])
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _save_split(train_df, valid_df, test_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "gold_train.csv", index=False)
    valid_df.to_csv(out_dir / "gold_valid.csv", index=False)
    test_df.to_csv(out_dir / "gold_test.csv", index=False)


def _gold_text(df: pd.DataFrame) -> pd.Series:
    return build_concat_text(df, ["subject", "content", "doccontent"], lowercase=True, remove_urls=False, max_text_length=5000)


def _silver_text(df: pd.DataFrame) -> pd.Series:
    return (df["subject_summary"] + " [SEP] " + df["content_summary"]).str.replace(r"\s+", " ", regex=True).str.strip()


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


def _class_weight_tensor(label_ids: np.ndarray, num_labels: int, device) -> torch.Tensor:
    counts = np.bincount(label_ids, minlength=num_labels)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_labels * counts)
    return torch.tensor(weights, dtype=torch.float, device=device)


def _train_one(experiment_name: str, model_dir: str, train_texts, train_labels, train_weights, valid_texts, valid_labels, test_texts, test_labels, label_to_id, args, output_root: Path):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    train_ds = TextClassificationDataset(train_texts, tokenizer, args.max_length, train_labels, train_weights)
    valid_ds = TextClassificationDataset(valid_texts, tokenizer, args.max_length, valid_labels)
    test_ds = TextClassificationDataset(test_texts, tokenizer, args.max_length, test_labels)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = _select_device()
    model = TextOnlyTransformerClassifier(model_dir, num_labels=len(label_to_id)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = _class_weight_tensor(np.array(train_labels), len(label_to_id), device)

    best_valid_f1 = -1.0
    best_epoch = 0
    out_dir = output_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "best_model.pt"
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            per_sample_loss = nn.functional.cross_entropy(
                logits,
                batch["labels"].to(device),
                weight=class_weights,
                reduction="none",
            )
            loss = (per_sample_loss * batch["sample_weight"].to(device)).mean()
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
    dump_eval_results(test_result, out_dir)
    with (out_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "valid": {k: valid_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
                "test": {k: test_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
                "sample_weight_setting": {"gold": 1.0, "silver": args.silver_weight},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return best_epoch, valid_result, test_result, out_dir


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    gold_df = _prepare_gold(args.gold_csv, args.id_column, args.label_column)
    silver_df = _prepare_silver(args.silver_csv)

    train_gold, valid_gold, test_gold = _split(gold_df, args.label_column, args.seed)
    _save_split(train_gold, valid_gold, test_gold, Path(args.processed_dir))

    labels = sorted(gold_df[args.label_column].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    gold_train_texts = _gold_text(train_gold)
    valid_texts = _gold_text(valid_gold)
    test_texts = _gold_text(test_gold)
    valid_labels = valid_gold[args.label_column].map(label_to_id).values
    test_labels = test_gold[args.label_column].map(label_to_id).values

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    baseline_by_model = {}
    for experiment_name, whether_dapt, whether_use_silver in EXPERIMENTS:
        model_dir = args.dapt_model_dir if whether_dapt else args.base_model_dir
        train_texts = gold_train_texts.copy()
        train_labels = train_gold[args.label_column].map(label_to_id).values.tolist()
        train_weights = [1.0] * len(train_gold)
        silver_size = 0
        if whether_use_silver and not silver_df.empty:
            silver_subset = silver_df[silver_df["label"].isin(label_to_id)].copy()
            silver_size = int(len(silver_subset))
            if silver_size > 0:
                train_texts = pd.concat([train_texts, _silver_text(silver_subset)], ignore_index=True)
                train_labels.extend(silver_subset["label"].map(label_to_id).values.tolist())
                train_weights.extend([float(args.silver_weight)] * silver_size)
        best_epoch, valid_result, test_result, metrics_dir = _train_one(
            experiment_name,
            model_dir,
            train_texts,
            np.array(train_labels),
            np.array(train_weights),
            valid_texts,
            valid_labels,
            test_texts,
            test_labels,
            label_to_id,
            args,
            output_root,
        )
        key = Path(model_dir).name
        baseline_key = (whether_dapt, False)
        if not whether_use_silver:
            baseline_by_model[baseline_key] = test_result["macro_f1"]
        delta = test_result["macro_f1"] - baseline_by_model.get((whether_dapt, False), test_result["macro_f1"])
        records.append(
            {
                "experiment": experiment_name,
                "base_model": key,
                "whether_dapt": whether_dapt,
                "whether_use_silver": whether_use_silver,
                "silver_size": silver_size,
                "sample_weight_setting": json.dumps({"gold": 1.0, "silver": args.silver_weight}, ensure_ascii=False),
                "train_rows": len(train_labels),
                "valid_rows": len(valid_gold),
                "test_rows": len(test_gold),
                "best_epoch": best_epoch,
                "valid_accuracy": valid_result["accuracy"],
                "valid_macro_precision": valid_result["macro_precision"],
                "valid_macro_recall": valid_result["macro_recall"],
                "valid_macro_f1": valid_result["macro_f1"],
                "test_accuracy": test_result["accuracy"],
                "test_macro_precision": test_result["macro_precision"],
                "test_macro_recall": test_result["macro_recall"],
                "test_macro_f1": test_result["macro_f1"],
                "delta_vs_gold_only_same_backbone_macro_f1": delta,
                "metrics_json": str(metrics_dir / "metrics_summary.json"),
                "classification_report_csv": str(metrics_dir / "classification_report.csv"),
                "confusion_matrix_csv": str(metrics_dir / "confusion_matrix.csv"),
            }
        )
        report = test_result.get("classification_report", {})
        for label in labels:
            if label in report:
                records[-1][f"{label}__precision"] = report[label].get("precision", 0.0)
                records[-1][f"{label}__recall"] = report[label].get("recall", 0.0)
                records[-1][f"{label}__f1"] = report[label].get("f1-score", 0.0)

    pd.DataFrame(records).to_csv(output_root / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
