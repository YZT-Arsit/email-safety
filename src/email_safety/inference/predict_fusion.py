from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from email_safety.data.torch_dataset import FusionDataset
from email_safety.features.structured_features import StructuredFeatureProcessor
from email_safety.models.fusion_model import TextStructuredFusionModel
from email_safety.preprocessing.text_clean import build_concat_text


def predict_with_fusion_checkpoint(
    checkpoint_path: str,
    df: pd.DataFrame,
    text_fields,
    preprocess_cfg,
    model_cfg,
    id_column: str,
    output_csv: str,
):
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    tokenizer = AutoTokenizer.from_pretrained(ckpt["tokenizer_name"])
    struct_proc = StructuredFeatureProcessor(with_scaler=True)
    struct_proc.columns_ = ckpt["structured_columns"]
    struct_proc.scaler.mean_ = np.array(ckpt["scaler_mean"])
    struct_proc.scaler.scale_ = np.array(ckpt["scaler_scale"])
    struct_proc.scaler.var_ = np.array(ckpt["scaler_var"])
    struct_proc.scaler.n_features_in_ = len(ckpt["structured_columns"])

    texts = build_concat_text(df, text_fields=text_fields, **preprocess_cfg)
    x_struct = struct_proc.transform(df)

    dataset = FusionDataset(
        texts=texts,
        structured_features=x_struct,
        tokenizer=tokenizer,
        max_length=model_cfg.get("max_length", 256),
        labels=None,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model = TextStructuredFusionModel(
        pretrained_model_name=ckpt["tokenizer_name"],
        num_labels=ckpt["num_labels"],
        structured_dim=len(ckpt["structured_columns"]),
        structured_hidden_dim=ckpt["structured_hidden_dim"],
        dropout=ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    preds = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                structured_features=batch["structured_features"],
            )
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(pred.tolist())

    out = pd.DataFrame({id_column: df[id_column], "pred_label": preds})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
