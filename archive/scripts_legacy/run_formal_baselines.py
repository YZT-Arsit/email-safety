#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.trainers.baseline_trainer import run_baseline_training
from email_safety.utils.config import load_config
from email_safety.utils.seed import set_seed


EXPERIMENT_SPECS = [
    {
        "name": "text_only_lr",
        "model_type": "logistic_regression",
        "use_text_features": True,
        "use_structured_features": False,
        "text_fields": ["subject", "content", "doccontent"],
    },
    {
        "name": "structured_only_lgbm",
        "model_type": "lightgbm",
        "use_text_features": False,
        "use_structured_features": True,
        "text_fields": ["subject", "content", "doccontent"],
    },
    {
        "name": "text_plus_structured_lr",
        "model_type": "logistic_regression",
        "use_text_features": True,
        "use_structured_features": True,
        "text_fields": ["subject", "content", "doccontent"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal baseline experiment suite")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--train-path", type=str, default="data/processed/train.csv")
    parser.add_argument("--valid-path", type=str, default="data/processed/valid.csv")
    parser.add_argument("--test-path", type=str, default="data/processed/test.csv")
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--output-dir", type=str, default="outputs/formal_baselines")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    set_seed(base_cfg.get("project", {}).get("seed", 42))

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for spec in EXPERIMENT_SPECS:
        cfg = copy.deepcopy(base_cfg)
        cfg["experiment_name"] = spec["name"]
        cfg["project"]["output_dir"] = str(output_root)
        cfg["data"]["train_path"] = args.train_path
        cfg["data"]["valid_path"] = args.valid_path
        cfg["data"]["test_path"] = args.test_path
        cfg["data"]["raw_format"] = "csv"
        cfg["data"]["label_column"] = args.label_column
        cfg["fields"]["text_fields"] = spec["text_fields"]
        cfg["model"]["model_type"] = spec["model_type"]
        cfg["model"]["use_text_features"] = spec["use_text_features"]
        cfg["model"]["use_structured_features"] = spec["use_structured_features"]
        cfg["train"]["save_model_name"] = f"{spec['name']}.joblib"
        cfg["train"]["save_submission_name"] = f"{spec['name']}_test_predictions.csv"

        result = run_baseline_training(cfg)

        summary = {
            "experiment": spec["name"],
            "accuracy": result["accuracy"],
            "macro_precision": result["macro_precision"],
            "macro_recall": result["macro_recall"],
            "macro_f1": result["macro_f1"],
            "metrics_json": str(output_root / "metrics" / spec["name"] / "metrics_summary.json"),
            "classification_report_csv": str(output_root / "metrics" / spec["name"] / "classification_report.csv"),
            "confusion_matrix_csv": str(output_root / "metrics" / spec["name"] / "confusion_matrix.csv"),
        }
        records.append(summary)

    results_df = pd.DataFrame(records)
    results_df.to_csv(output_root / "results_summary.csv", index=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
