from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from email_safety.data.io import coerce_string_columns, ensure_columns, load_dataframe
from email_safety.data.split import make_train_valid_split
from email_safety.data.torch_dataset import FusionDataset
from email_safety.evaluation.metrics import dump_eval_results, evaluate_multiclass
from email_safety.features.structured_features import StructuredFeatureProcessor
from email_safety.models.fusion_model import TextStructuredFusionModel
from email_safety.preprocessing.text_clean import build_concat_text
from email_safety.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _load_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    fields_cfg = config["fields"]
    id_col = data_cfg["id_column"]
    label_col = data_cfg["label_column"]
    raw_format = data_cfg.get("raw_format", "auto")

    train_df = load_dataframe(data_cfg["train_path"], raw_format=raw_format, id_column=id_col)
    valid_path = data_cfg.get("valid_path")
    test_path = data_cfg.get("test_path")

    text_fields = fields_cfg["text_fields"]
    train_df = ensure_columns(train_df, [id_col, label_col] + text_fields)

    if valid_path:
        valid_df = load_dataframe(valid_path, raw_format=raw_format, id_column=id_col)
        valid_df = ensure_columns(valid_df, [id_col, label_col] + text_fields)
    else:
        train_df, valid_df = make_train_valid_split(
            train_df,
            label_column=label_col,
            valid_size=data_cfg.get("valid_size", 0.2),
            random_state=config["project"].get("seed", 42),
            stratify=data_cfg.get("stratify", True),
        )

    if test_path:
        test_df = load_dataframe(test_path, raw_format=raw_format, id_column=id_col)
        test_df = ensure_columns(test_df, [id_col] + text_fields)
    else:
        test_df = pd.DataFrame(columns=[id_col] + text_fields)

    string_cols = text_fields + [id_col, label_col, "sender", "from", "fromname", "url", "htmltag", "attach"]
    train_df = coerce_string_columns(train_df, string_cols)
    valid_df = coerce_string_columns(valid_df, string_cols)
    if not test_df.empty:
        test_df = coerce_string_columns(test_df, string_cols)

    return train_df, valid_df, test_df


def _eval_epoch(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            struct = batch["structured_features"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attn, structured_features=struct)
            pred = torch.argmax(logits, dim=-1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    return evaluate_multiclass(np.array(y_true), np.array(y_pred))


def run_fusion_training(config: Dict):
    data_cfg = config["data"]
    fields_cfg = config["fields"]
    model_cfg = config["model"]
    train_cfg = config["train"]

    output_root = Path(config["project"]["output_dir"])
    model_dir = output_root / "models"
    metrics_dir = output_root / "metrics" / "fusion"
    sub_dir = output_root / "submissions"
    for d in [model_dir, metrics_dir, sub_dir]:
        d.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = _load_splits(config)
    label_col = data_cfg["label_column"]

    preprocess_cfg = {
        "lowercase": True,
        "remove_urls": False,
        "max_text_length": 5000,
    }
    train_text = build_concat_text(train_df, fields_cfg["text_fields"], **preprocess_cfg)
    valid_text = build_concat_text(valid_df, fields_cfg["text_fields"], **preprocess_cfg)

    struct_proc = StructuredFeatureProcessor(with_scaler=True)
    x_train_struct = struct_proc.fit_transform(train_df)
    x_valid_struct = struct_proc.transform(valid_df)

    tokenizer_name = model_cfg.get("pretrained_model_name", "bert-base-chinese")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = FusionDataset(
        texts=train_text,
        structured_features=x_train_struct,
        tokenizer=tokenizer,
        max_length=model_cfg.get("max_length", 256),
        labels=train_df[label_col].astype(int).values,
    )
    valid_ds = FusionDataset(
        texts=valid_text,
        structured_features=x_valid_struct,
        tokenizer=tokenizer,
        max_length=model_cfg.get("max_length", 256),
        labels=valid_df[label_col].astype(int).values,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    valid_loader = DataLoader(valid_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextStructuredFusionModel(
        pretrained_model_name=tokenizer_name,
        num_labels=model_cfg.get("num_labels", 5),
        structured_dim=x_train_struct.shape[1],
        structured_hidden_dim=model_cfg.get("structured_hidden_dim", 64),
        dropout=model_cfg.get("fusion_dropout", 0.2),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    best_f1 = -1.0
    best_metrics = None

    epochs = int(train_cfg.get("epochs", 3))
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            struct = batch["structured_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attn, structured_features=struct)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        metrics = _eval_epoch(model, valid_loader, device)
        LOGGER.info(
            "Epoch %d/%d loss=%.4f val_acc=%.4f val_macro_f1=%.4f",
            epoch,
            epochs,
            avg_loss,
            metrics["accuracy"],
            metrics["macro_f1"],
        )

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics
            ckpt_path = model_dir / "fusion_model.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "tokenizer_name": tokenizer_name,
                    "num_labels": model_cfg.get("num_labels", 5),
                    "structured_hidden_dim": model_cfg.get("structured_hidden_dim", 64),
                    "dropout": model_cfg.get("fusion_dropout", 0.2),
                    "structured_columns": struct_proc.columns_,
                    "scaler_mean": struct_proc.scaler.mean_.tolist(),
                    "scaler_scale": struct_proc.scaler.scale_.tolist(),
                    "scaler_var": struct_proc.scaler.var_.tolist(),
                    "config": config,
                },
                ckpt_path,
            )

    if best_metrics is not None:
        dump_eval_results(best_metrics, metrics_dir)

    if not test_df.empty:
        model.eval()
        x_test_text = build_concat_text(test_df, fields_cfg["text_fields"], **preprocess_cfg)
        x_test_struct = struct_proc.transform(test_df)
        test_ds = FusionDataset(
            texts=x_test_text,
            structured_features=x_test_struct,
            tokenizer=tokenizer,
            max_length=model_cfg.get("max_length", 256),
            labels=None,
        )
        test_loader = DataLoader(test_ds, batch_size=train_cfg.get("batch_size", 16), shuffle=False)

        preds = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    structured_features=batch["structured_features"].to(device),
                )
                pred = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
                preds.extend(pred)

        sub_path = sub_dir / "submission_fusion.csv"
        pd.DataFrame({data_cfg["id_column"]: test_df[data_cfg["id_column"]], "pred_label": preds}).to_csv(
            sub_path, index=False
        )

    return best_metrics
