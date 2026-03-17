#!/usr/bin/env python
"""Run a trained baseline model on all unlabeled mails and export rich prediction metadata.
Inputs: unlabeled mail log and saved baseline model. Outputs: all_unlabeled_predictions.csv."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.io import ensure_columns, load_dataframe
from email_safety.inference.predict import predict_unlabeled_with_metadata
from email_safety.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict all unlabeled emails with baseline model")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--results-summary", type=str, default="outputs/formal_baselines/results_summary.csv")
    parser.add_argument("--model-root", type=str, default="outputs/formal_baselines/models")
    parser.add_argument("--selection-metric", type=str, default="macro_f1")
    parser.add_argument("--input-path", type=str, default="spam_email_data.log")
    parser.add_argument("--raw-format", type=str, default="log")
    parser.add_argument("--rules-config", type=str, default="configs/weak_label_rules.yaml")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/predictions/all_unlabeled_predictions.csv",
    )
    return parser.parse_args()


def _resolve_model_path(args) -> str:
    if args.model_path:
        return args.model_path

    summary_path = Path(args.results_summary)
    if not summary_path.exists():
        raise ValueError("model-path is empty and results_summary.csv was not found")

    summary_df = pd.read_csv(summary_path)
    if args.selection_metric not in summary_df.columns:
        raise ValueError(f"selection metric not found in results summary: {args.selection_metric}")

    best_row = summary_df.sort_values(args.selection_metric, ascending=False).iloc[0]
    experiment_name = str(best_row["experiment"])
    model_path = Path(args.model_root) / experiment_name / f"{experiment_name}.joblib"
    if not model_path.exists():
        raise ValueError(f"Best model file not found: {model_path}")
    return str(model_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    id_column = cfg["data"]["id_column"]
    model_path = _resolve_model_path(args)

    df = load_dataframe(args.input_path, raw_format=args.raw_format, id_column=id_column)
    df = ensure_columns(
        df,
        [
            id_column,
            "subject",
            "content",
            "doccontent",
            "from",
            "fromname",
            "url",
            "attach",
            "htmltag",
            "ip",
            "rcpt",
        ],
    )

    predict_unlabeled_with_metadata(
        model_path=model_path,
        df=df,
        text_fields=cfg["fields"]["text_fields"],
        preprocess_cfg=cfg["preprocess"],
        use_structured_features=cfg["model"].get("use_structured_features", True),
        id_column=id_column,
        output_csv=args.output_csv,
        rules_config_path=args.rules_config,
    )


if __name__ == "__main__":
    main()
