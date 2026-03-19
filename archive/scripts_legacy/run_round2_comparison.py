#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description="Run round-2 baseline comparison on Gold v1/v2 and optional trusted silver")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--gold-v1-csv", type=str, default="data/annotation/clean_labeled_dataset.csv")
    parser.add_argument("--gold-v2-csv", type=str, default="data/annotation/gold/gold_v2.csv")
    parser.add_argument("--trusted-silver-csv", type=str, default="data/annotation/silver/trusted_silver.csv")
    parser.add_argument("--trusted-silver-eval-json", type=str, default="data/annotation/silver/trusted_silver_eval.json")
    parser.add_argument("--work-dir", type=str, default="data/processed/round2_comparison")
    parser.add_argument("--output-dir", type=str, default="outputs/round2_comparison")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _prepare_gold(path: str, id_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df[id_column] = df[id_column].map(_safe_text)
    df[label_column] = df[label_column].map(_safe_text)
    df = df[df[id_column].ne("") & df[label_column].ne("")].copy()
    return df.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)


def _prepare_trusted_silver(path: str, id_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df[id_column] = df[id_column].map(_safe_text)
    df["pred_label"] = df["pred_label"].map(_safe_text)
    df[label_column] = df["pred_label"]
    df = df[df[id_column].ne("") & df[label_column].ne("")].copy()
    return df.drop_duplicates(subset=[id_column], keep="last").reset_index(drop=True)


def _split_dataset(df: pd.DataFrame, label_column: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def _write_split(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    valid_df.to_csv(out_dir / "valid.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def _trusted_silver_allowed(eval_json_path: str) -> bool:
    path = Path(eval_json_path)
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload.get("class_results", []):
        if item.get("recommendation") == "include":
            return True
    return False


def main() -> None:
    args = parse_args()
    base_cfg = load_config(args.config)
    set_seed(args.seed)

    gold_v1_df = _prepare_gold(args.gold_v1_csv, args.id_column, args.label_column)
    gold_v2_df = _prepare_gold(args.gold_v2_csv, args.id_column, args.label_column)

    dataset_variants = {
        "gold_v1_only": {
            "df": gold_v1_df,
            "uses_trusted_silver": False,
            "status": "ready",
        },
        "gold_v2_only": {
            "df": gold_v2_df,
            "uses_trusted_silver": False,
            "status": "ready",
        },
    }

    trusted_silver_path = Path(args.trusted_silver_csv)
    if trusted_silver_path.exists() and trusted_silver_path.stat().st_size > 0 and _trusted_silver_allowed(args.trusted_silver_eval_json):
        trusted_df = _prepare_trusted_silver(args.trusted_silver_csv, args.id_column, args.label_column)
        merged_df = pd.concat([gold_v2_df, trusted_df], axis=0, ignore_index=True)
        merged_df = merged_df.sort_values(by=[args.id_column]).drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)
        dataset_variants["gold_v2_plus_trusted_silver"] = {
            "df": merged_df,
            "uses_trusted_silver": True,
            "status": "ready",
        }
    else:
        dataset_variants["gold_v2_plus_trusted_silver"] = {
            "df": pd.DataFrame(columns=gold_v2_df.columns),
            "uses_trusted_silver": True,
            "status": "skipped_no_trusted_silver",
        }

    work_root = Path(args.work_dir)
    output_root = Path(args.output_dir)
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for variant_name, variant_info in dataset_variants.items():
        df = variant_info["df"]
        status = variant_info["status"]
        uses_trusted_silver = variant_info["uses_trusted_silver"]

        if status != "ready" or df.empty:
            for spec in EXPERIMENT_SPECS:
                records.append(
                    {
                        "dataset_variant": variant_name,
                        "experiment": spec["name"],
                        "status": status,
                        "uses_trusted_silver": uses_trusted_silver,
                        "dataset_rows": int(len(df)),
                    }
                )
            continue

        train_df, valid_df, test_df = _split_dataset(df, args.label_column, args.seed)
        split_dir = work_root / variant_name
        _write_split(train_df, valid_df, test_df, split_dir)

        for spec in EXPERIMENT_SPECS:
            cfg = copy.deepcopy(base_cfg)
            cfg["experiment_name"] = f"{variant_name}__{spec['name']}"
            cfg["project"]["output_dir"] = str(output_root)
            cfg["data"]["train_path"] = str(split_dir / "train.csv")
            cfg["data"]["valid_path"] = str(split_dir / "valid.csv")
            cfg["data"]["test_path"] = str(split_dir / "test.csv")
            cfg["data"]["raw_format"] = "csv"
            cfg["data"]["label_column"] = args.label_column
            cfg["fields"]["text_fields"] = spec["text_fields"]
            cfg["model"]["model_type"] = spec["model_type"]
            cfg["model"]["use_text_features"] = spec["use_text_features"]
            cfg["model"]["use_structured_features"] = spec["use_structured_features"]
            cfg["train"]["save_model_name"] = f"{variant_name}__{spec['name']}.joblib"
            cfg["train"]["save_submission_name"] = f"{variant_name}__{spec['name']}_test_predictions.csv"

            result = run_baseline_training(cfg)
            records.append(
                {
                    "dataset_variant": variant_name,
                    "experiment": spec["name"],
                    "status": "done",
                    "uses_trusted_silver": uses_trusted_silver,
                    "dataset_rows": int(len(df)),
                    "train_rows": int(len(train_df)),
                    "valid_rows": int(len(valid_df)),
                    "test_rows": int(len(test_df)),
                    "accuracy": result["accuracy"],
                    "macro_precision": result["macro_precision"],
                    "macro_recall": result["macro_recall"],
                    "macro_f1": result["macro_f1"],
                }
            )

    summary_df = pd.DataFrame(records)
    gold_v1_metrics = summary_df[summary_df["dataset_variant"] == "gold_v1_only"].set_index("experiment")

    def _delta(row, metric: str):
        if row.get("status") != "done" or row.get("experiment") not in gold_v1_metrics.index:
            return ""
        base_val = gold_v1_metrics.loc[row["experiment"]].get(metric)
        cur_val = row.get(metric)
        if pd.isna(base_val) or pd.isna(cur_val):
            return ""
        return float(cur_val - base_val)

    for metric in ["accuracy", "macro_precision", "macro_recall", "macro_f1"]:
        summary_df[f"delta_vs_gold_v1_{metric}"] = summary_df.apply(lambda row: _delta(row, metric), axis=1)

    summary_path = output_root / "round2_results_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
