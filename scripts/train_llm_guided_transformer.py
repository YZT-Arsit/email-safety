#!/usr/bin/env python
"""Train a text-only BERT classifier with optional LLM soft-target distillation and risk-hint augmentation.
Inputs: labeled CSV/JSONL with optional class_probs/reasoning columns. Outputs: checkpoint, metrics, reports, and saved split files."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
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

from email_safety.data.io import ensure_columns, load_dataframe
from email_safety.data.llm_guided_text_dataset import LLMGuidedTextDataset
from email_safety.evaluation.metrics import dump_eval_results, evaluate_multiclass
from email_safety.models.text_transformer import TextOnlyTransformerClassifier
from email_safety.preprocessing.text_clean import build_concat_text, normalize_text
from email_safety.utils.config import load_config
from email_safety.utils.logger import get_logger
from email_safety.utils.seed import set_seed

LOGGER = get_logger(__name__)
SENT_RE = re.compile(r"(?<=[。！？!?\.])\s+")


def parse_args():
    parser = argparse.ArgumentParser(description="Train pure BERT or LLM-guided BERT with distillation")
    parser.add_argument("--config", type=str, default="configs/llm_guided_transformer.yaml")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--use-risk-hint", action="store_true")
    parser.add_argument("--no-use-risk-hint", dest="use_risk_hint", action="store_false")
    parser.set_defaults(use_risk_hint=None)
    parser.add_argument("--use-soft-targets", action="store_true")
    parser.add_argument("--no-use-soft-targets", dest="use_soft_targets", action="store_false")
    parser.set_defaults(use_soft_targets=None)
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


def _prepare_dataframe(path: str, raw_format: str, label_column: str, id_column: str) -> pd.DataFrame:
    df = load_dataframe(path, raw_format=raw_format, id_column=id_column).copy()
    df = ensure_columns(df, [id_column, label_column, "subject", "content", "doccontent", "reasoning", "reasoning_summary", "class_probs"])
    df[id_column] = df[id_column].map(_safe_text)
    df[label_column] = df[label_column].map(_safe_text)
    for col in ["subject", "content", "doccontent", "reasoning", "reasoning_summary", "class_probs"]:
        df[col] = df[col].map(_safe_text)
    df = df[df[id_column].ne("") & df[label_column].ne("")].copy()
    df = df.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)
    return df


def _stratified_split(df: pd.DataFrame, label_column: str, seed: int):
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=df[label_column])
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_df[label_column])
    return train_df.reset_index(drop=True), valid_df.reset_index(drop=True), test_df.reset_index(drop=True)


def _save_split_files(train_df, valid_df, test_df, processed_dir: Path):
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index=False)
    valid_df.to_csv(processed_dir / "valid.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)


def _summarize_reasoning(text: str, max_chars: int) -> str:
    text = normalize_text(text, lowercase=False, remove_urls=False)
    if not text:
        return ""
    pieces = [seg.strip() for seg in SENT_RE.split(text) if seg.strip()]
    if not pieces:
        return text[:max_chars]
    summary = " ".join(pieces[:2])
    return summary[:max_chars]


def _build_risk_hints(df: pd.DataFrame, reasoning_column: str, summary_column: str, max_chars: int) -> list[str]:
    hints = []
    for _, row in df.iterrows():
        hint = _safe_text(row.get(summary_column, ""))
        if not hint:
            hint = _summarize_reasoning(_safe_text(row.get(reasoning_column, "")), max_chars=max_chars)
        hints.append(hint[:max_chars])
    return hints


def _parse_soft_targets(series: pd.Series, label_to_id: dict[str, int]) -> np.ndarray:
    vectors = []
    for raw in series.tolist():
        if not raw:
            vectors.append(np.zeros(len(label_to_id), dtype=np.float32))
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            payload = {}
        vec = np.zeros(len(label_to_id), dtype=np.float32)
        if isinstance(payload, dict):
            total = 0.0
            for label, idx in label_to_id.items():
                try:
                    val = max(0.0, float(payload.get(label, 0.0)))
                except Exception:
                    val = 0.0
                vec[idx] = val
                total += val
            if total > 0:
                vec = vec / total
        vectors.append(vec)
    return np.stack(vectors, axis=0)


def _evaluate_model(model, loader, device, id_to_label):
    model.eval()
    y_true, y_pred = [], []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            y_pred.extend([id_to_label[idx] for idx in pred])
            y_true.extend([id_to_label[idx] for idx in batch["labels"].cpu().numpy().tolist()])
            probs_all.append(probs.cpu().numpy())
    result = evaluate_multiclass(y_true, y_pred)
    result["probabilities"] = np.concatenate(probs_all, axis=0) if probs_all else np.zeros((0, len(id_to_label)))
    result["y_true"] = y_true
    result["y_pred"] = y_pred
    return result


def _distillation_loss(logits: torch.Tensor, soft_targets: torch.Tensor, temperature: float) -> torch.Tensor:
    log_probs = torch.log_softmax(logits / temperature, dim=-1)
    return torch.sum(-soft_targets * log_probs, dim=-1).mean() * (temperature ** 2)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"].get("seed", 42))

    data_cfg = cfg["data"]
    preprocess_cfg = cfg["preprocess"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    fields_cfg = cfg["fields"]

    use_risk_hint = train_cfg.get("use_risk_hint", True) if args.use_risk_hint is None else args.use_risk_hint
    use_soft_targets = train_cfg.get("use_soft_targets", True) if args.use_soft_targets is None else args.use_soft_targets

    output_dir = Path(args.output_dir or cfg["project"]["output_dir"])
    processed_dir = Path(data_cfg["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = _prepare_dataframe(
        path=data_cfg["input_path"],
        raw_format=data_cfg.get("raw_format", "csv"),
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
    run_cfg["train"]["use_risk_hint"] = bool(use_risk_hint)
    run_cfg["train"]["use_soft_targets"] = bool(use_soft_targets)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_cfg, f, ensure_ascii=False, indent=2)

    train_texts = build_concat_text(train_df, fields_cfg["text_fields"], **preprocess_cfg)
    valid_texts = build_concat_text(valid_df, fields_cfg["text_fields"], **preprocess_cfg)
    test_texts = build_concat_text(test_df, fields_cfg["text_fields"], **preprocess_cfg)

    train_hints = _build_risk_hints(train_df, data_cfg["reasoning_column"], data_cfg["reasoning_summary_column"], int(train_cfg.get("hint_max_chars", 160))) if use_risk_hint else None
    valid_hints = _build_risk_hints(valid_df, data_cfg["reasoning_column"], data_cfg["reasoning_summary_column"], int(train_cfg.get("hint_max_chars", 160))) if use_risk_hint else None
    test_hints = _build_risk_hints(test_df, data_cfg["reasoning_column"], data_cfg["reasoning_summary_column"], int(train_cfg.get("hint_max_chars", 160))) if use_risk_hint else None

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])
    train_labels = train_df[data_cfg["label_column"]].map(label_to_id).values
    valid_labels = valid_df[data_cfg["label_column"]].map(label_to_id).values
    test_labels = test_df[data_cfg["label_column"]].map(label_to_id).values

    train_soft = _parse_soft_targets(train_df[data_cfg["soft_targets_column"]], label_to_id) if use_soft_targets else None
    valid_soft = _parse_soft_targets(valid_df[data_cfg["soft_targets_column"]], label_to_id) if use_soft_targets else None
    test_soft = _parse_soft_targets(test_df[data_cfg["soft_targets_column"]], label_to_id) if use_soft_targets else None

    train_ds = LLMGuidedTextDataset(train_texts, tokenizer, model_cfg["max_length"], train_labels, train_hints, train_soft)
    valid_ds = LLMGuidedTextDataset(valid_texts, tokenizer, model_cfg["max_length"], valid_labels, valid_hints, valid_soft)
    test_ds = LLMGuidedTextDataset(test_texts, tokenizer, model_cfg["max_length"], test_labels, test_hints, test_soft)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    batch_size = int(train_cfg.get("batch_size", 8))
    LOGGER.info("Using device=%s batch_size=%d use_risk_hint=%s use_soft_targets=%s", device.type, batch_size, use_risk_hint, use_soft_targets)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=int(train_cfg.get("num_workers", 0)))
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=int(train_cfg.get("num_workers", 0)))

    model = TextOnlyTransformerClassifier(model_cfg["pretrained_model_name"], num_labels=model_cfg["num_labels"]).to(device)
    optimizer = AdamW(model.parameters(), lr=float(train_cfg.get("lr", 2e-5)), weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    criterion = nn.CrossEntropyLoss()

    best_valid_f1 = -1.0
    best_epoch = 0
    checkpoint_path = output_dir / "best_model.pt"
    train_logs = []

    distill_alpha = float(train_cfg.get("distill_alpha", 0.5))
    temperature = float(train_cfg.get("temperature", 2.0))

    for epoch in range(1, int(train_cfg.get("epochs", 3)) + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            hard_loss = criterion(logits, batch["labels"].to(device))
            if use_soft_targets and "soft_targets" in batch and torch.sum(batch["soft_targets"]).item() > 0:
                soft_loss = _distillation_loss(logits, batch["soft_targets"].to(device), temperature)
                loss = (1.0 - distill_alpha) * hard_loss + distill_alpha * soft_loss
            else:
                loss = hard_loss
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        avg_train_loss = running_loss / max(1, len(train_loader))
        valid_result = _evaluate_model(model, valid_loader, device, id_to_label)
        train_logs.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "valid_accuracy": valid_result["accuracy"],
            "valid_macro_f1": valid_result["macro_f1"],
        })
        LOGGER.info("Epoch %d/%d train_loss=%.4f valid_macro_f1=%.4f", epoch, int(train_cfg.get("epochs", 3)), avg_train_loss, valid_result["macro_f1"])

        if valid_result["macro_f1"] > best_valid_f1:
            best_valid_f1 = valid_result["macro_f1"]
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "pretrained_model_name": model_cfg["pretrained_model_name"],
                "max_length": model_cfg["max_length"],
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "config": run_cfg,
                "use_risk_hint": use_risk_hint,
                "use_soft_targets": use_soft_targets,
            }, checkpoint_path)

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
        "use_risk_hint": bool(use_risk_hint),
        "use_soft_targets": bool(use_soft_targets),
        "valid": {k: valid_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
        "test": {k: test_result[k] for k in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]},
        "train_logs": train_logs,
    }
    with (output_dir / "metrics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)

    pd.DataFrame([
        {
            "experiment": "llm_guided_bert" if (use_risk_hint or use_soft_targets) else "plain_bert",
            "model_name": model_cfg["pretrained_model_name"],
            "best_epoch": best_epoch,
            "use_risk_hint": bool(use_risk_hint),
            "use_soft_targets": bool(use_soft_targets),
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
    ]).to_csv(output_dir / "results_summary.csv", index=False)


if __name__ == "__main__":
    main()
