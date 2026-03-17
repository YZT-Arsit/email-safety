#!/usr/bin/env python
"""Run a trained text-only Transformer checkpoint on input mails with local backbone loading.
Inputs: checkpoint and input file. Outputs: prediction CSV with scores and summaries."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.io import ensure_columns, load_dataframe
from email_safety.data.text_dataset import TextClassificationDataset
from email_safety.models.text_transformer import TextOnlyTransformerClassifier
from email_safety.preprocessing.text_clean import build_concat_text
from email_safety.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with trained text-only Transformer")
    parser.add_argument("--config", type=str, default="configs/text_transformer.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/transformer_round1/best_model.pt")
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--raw-format", type=str, default="csv")
    parser.add_argument("--output-csv", type=str, default="outputs/transformer_round1/predictions.csv")
    return parser.parse_args()


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    backbone_path = checkpoint.get("pretrained_model_name") or checkpoint.get("model_dir")
    if not backbone_path:
        raise ValueError("checkpoint missing pretrained_model_name/model_dir")
    label_to_id = checkpoint["label_to_id"]
    id_to_label = {int(k): v for k, v in checkpoint["id_to_label"].items()} if not isinstance(next(iter(checkpoint["id_to_label"].keys())), int) else checkpoint["id_to_label"]

    data_cfg = cfg["data"]
    fields_cfg = cfg["fields"]
    preprocess_cfg = cfg["preprocess"]
    df = load_dataframe(args.input_path, raw_format=args.raw_format, id_column=data_cfg["id_column"])
    df = ensure_columns(df, [data_cfg["id_column"]] + fields_cfg["text_fields"])
    texts = build_concat_text(df, fields_cfg["text_fields"], **preprocess_cfg)

    try:
        tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load tokenizer. If the checkpoint references a remote model, "
            "please ensure network access or replace `pretrained_model_name` with a local path."
        ) from exc
    ds = TextClassificationDataset(texts, tokenizer, checkpoint["max_length"], labels=None)
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 8)), shuffle=False)

    device = _select_device()
    try:
        model = TextOnlyTransformerClassifier(
            pretrained_model_name=backbone_path,
            num_labels=len(label_to_id),
        ).to(device)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load pretrained backbone for prediction. "
            "Please ensure the model path is locally available."
        ) from exc
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds = []
    scores = []
    top2_labels = []
    top2_scores = []
    uncertainties = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            probs = torch.softmax(logits, dim=-1)
            top_scores, top_idx = torch.max(probs, dim=-1)
            order = torch.argsort(probs, dim=-1, descending=True)
            second_idx = order[:, 1] if probs.shape[1] > 1 else order[:, 0]
            second_scores = probs[torch.arange(probs.shape[0]), second_idx]
            preds.extend([id_to_label[int(i)] for i in top_idx.cpu().numpy().tolist()])
            scores.extend(top_scores.cpu().numpy().tolist())
            top2_labels.extend([id_to_label[int(i)] for i in second_idx.cpu().numpy().tolist()])
            top2_scores.extend(second_scores.cpu().numpy().tolist())
            uncertainties.extend((1.0 - top_scores).cpu().numpy().tolist())

    out = pd.DataFrame(
        {
            data_cfg["id_column"]: df[data_cfg["id_column"]],
            "pred_label": preds,
            "pred_score": scores,
            "top2_label": top2_labels,
            "top2_score": top2_scores,
            "uncertainty": uncertainties,
            "subject_summary": df.get("subject", pd.Series([""] * len(df))).fillna("").astype(str).str.slice(0, 120),
            "content_summary": df.get("content", pd.Series([""] * len(df))).fillna("").astype(str).str.slice(0, 200),
        }
    )
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
