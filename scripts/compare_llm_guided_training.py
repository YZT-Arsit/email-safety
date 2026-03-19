#!/usr/bin/env python
"""Compare pure BERT and LLM-guided BERT on the same test split with metrics and precision-recall curves.
Inputs: two checkpoints and a labeled test file. Outputs: comparison summary CSV/JSON and per-class PR curve plots."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve
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


def parse_args():
    parser = argparse.ArgumentParser(description="Compare plain BERT vs LLM-guided BERT")
    parser.add_argument("--config", type=str, default="configs/llm_guided_transformer.yaml")
    parser.add_argument("--plain-checkpoint", type=str, required=True)
    parser.add_argument("--guided-checkpoint", type=str, required=True)
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="outputs/llm_guided_comparison")
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _summarize_reasoning(text: str, max_chars: int) -> str:
    text = normalize_text(text, lowercase=False, remove_urls=False)
    return text[:max_chars]


def _build_risk_hints(df: pd.DataFrame, reasoning_column: str, summary_column: str, max_chars: int) -> list[str]:
    hints = []
    for _, row in df.iterrows():
        hint = _safe_text(row.get(summary_column, "")) or _summarize_reasoning(_safe_text(row.get(reasoning_column, "")), max_chars)
        hints.append(hint[:max_chars])
    return hints


def _load_eval_df(path: str, data_cfg: dict) -> pd.DataFrame:
    df = load_dataframe(path, raw_format=data_cfg.get("raw_format", "csv"), id_column=data_cfg["id_column"]).copy()
    df = ensure_columns(
        df,
        [
            data_cfg["id_column"],
            data_cfg["label_column"],
            "subject",
            "content",
            "doccontent",
            data_cfg["reasoning_column"],
            data_cfg["reasoning_summary_column"],
        ],
    )
    for col in [
        data_cfg["id_column"],
        data_cfg["label_column"],
        "subject",
        "content",
        "doccontent",
        data_cfg["reasoning_column"],
        data_cfg["reasoning_summary_column"],
    ]:
        df[col] = df[col].map(_safe_text)
    return df[df[data_cfg["id_column"]].ne("") & df[data_cfg["label_column"]].ne("")].drop_duplicates(subset=[data_cfg["id_column"]], keep="last").reset_index(drop=True)


def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _run_eval(checkpoint_path: str, df: pd.DataFrame, cfg: dict, output_dir: Path, name: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone_path = checkpoint.get("pretrained_model_name") or checkpoint.get("model_dir")
    label_to_id = checkpoint["label_to_id"]
    id_to_label = checkpoint["id_to_label"]
    if not isinstance(next(iter(id_to_label.keys())), int):
        id_to_label = {int(k): v for k, v in id_to_label.items()}

    texts = build_concat_text(df, cfg["fields"]["text_fields"], **cfg["preprocess"])
    use_risk_hint = bool(checkpoint.get("use_risk_hint", False))
    hints = _build_risk_hints(
        df,
        cfg["data"]["reasoning_column"],
        cfg["data"]["reasoning_summary_column"],
        int(cfg["train"].get("hint_max_chars", 160)),
    ) if use_risk_hint else None
    labels = df[cfg["data"]["label_column"]].map(label_to_id).values

    tokenizer = AutoTokenizer.from_pretrained(backbone_path, local_files_only=True)
    ds = LLMGuidedTextDataset(texts, tokenizer, checkpoint["max_length"], labels=labels, risk_hints=hints)
    loader = DataLoader(ds, batch_size=int(cfg["train"].get("batch_size", 8)), shuffle=False)

    device = _select_device()
    model = TextOnlyTransformerClassifier(backbone_path, num_labels=len(label_to_id)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true, y_pred = [], []
    probs_all = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            pred_idx = np.argmax(probs, axis=1)
            y_pred.extend([id_to_label[int(i)] for i in pred_idx.tolist()])
            y_true.extend([id_to_label[int(i)] for i in batch["labels"].cpu().numpy().tolist()])
            probs_all.append(probs)

    probs_all = np.concatenate(probs_all, axis=0)
    result = evaluate_multiclass(y_true, y_pred)
    model_dir = output_dir / name
    dump_eval_results(result, model_dir)

    y_true_idx = np.array([label_to_id[label] for label in y_true])
    pr_curves = {}
    pr_stats = []
    for label, idx in label_to_id.items():
        binary_true = (y_true_idx == idx).astype(int)
        precision, recall, _ = precision_recall_curve(binary_true, probs_all[:, idx])
        ap = average_precision_score(binary_true, probs_all[:, idx]) if binary_true.sum() > 0 else 0.0
        pr_curves[label] = {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "average_precision": float(ap),
        }
        pr_stats.append({"label": label, "average_precision": float(ap)})

    with (model_dir / "pr_stats.json").open("w", encoding="utf-8") as f:
        json.dump(pr_stats, f, ensure_ascii=False, indent=2)

    return {
        "name": name,
        "test_accuracy": result["accuracy"],
        "test_macro_precision": result["macro_precision"],
        "test_macro_recall": result["macro_recall"],
        "test_macro_f1": result["macro_f1"],
        "use_risk_hint": use_risk_hint,
        "classification_report": result["classification_report"],
        "pr_curves": pr_curves,
    }


def _plot_joint_pr_curves(output_dir: Path, plain_result: dict, guided_result: dict) -> None:
    labels = sorted(set(plain_result["pr_curves"].keys()) | set(guided_result["pr_curves"].keys()))
    for label in labels:
        plt.figure(figsize=(6, 5))
        if label in plain_result["pr_curves"]:
            curve = plain_result["pr_curves"][label]
            plt.plot(curve["recall"], curve["precision"], label=f"plain AP={curve['average_precision']:.4f}")
        if label in guided_result["pr_curves"]:
            curve = guided_result["pr_curves"][label]
            plt.plot(curve["recall"], curve["precision"], label=f"guided AP={curve['average_precision']:.4f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Comparison: {label}")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.savefig(output_dir / f"pr_curve_compare_{label}.png", dpi=150)
        plt.close()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    test_path = args.test_path or str(Path(cfg["data"]["processed_dir"]) / "test.csv")
    df = _load_eval_df(test_path, cfg["data"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plain_result = _run_eval(args.plain_checkpoint, df, cfg, output_dir, "plain_bert")
    guided_result = _run_eval(args.guided_checkpoint, df, cfg, output_dir, "llm_guided_bert")
    _plot_joint_pr_curves(output_dir, plain_result, guided_result)

    summary = pd.DataFrame([
        {k: v for k, v in plain_result.items() if k not in {"classification_report", "pr_curves"}},
        {k: v for k, v in guided_result.items() if k not in {"classification_report", "pr_curves"}},
    ])
    summary["delta_macro_f1_vs_plain"] = summary["test_macro_f1"] - float(plain_result["test_macro_f1"])
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    with (output_dir / "comparison_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "plain_bert": {k: v for k, v in plain_result.items() if k not in {"classification_report", "pr_curves"}},
                "llm_guided_bert": {k: v for k, v in guided_result.items() if k not in {"classification_report", "pr_curves"}},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


if __name__ == "__main__":
    main()
