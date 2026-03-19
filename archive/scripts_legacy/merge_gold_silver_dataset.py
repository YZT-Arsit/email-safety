#!/usr/bin/env python
from __future__ import annotations

import argparse
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


DEFAULT_ALLOWED_LABELS = {
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge gold and optional silver datasets for later training")
    parser.add_argument(
        "--gold-csv",
        type=str,
        default="data/annotation/clean_labeled_dataset.csv",
    )
    parser.add_argument(
        "--silver-csv",
        type=str,
        default="data/annotation/silver/silver_pseudo_labeled.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/merged_training_dataset.csv",
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default="data/processed/merged_training_dataset_stats.json",
    )
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument(
        "--include-silver",
        action="store_true",
        help="If set, merge silver data in addition to gold data",
    )
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _load_dataset(path: str, label_column: str, id_column: str, source_name: str) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        return pd.DataFrame(columns=[id_column, label_column, "data_source"])

    df = pd.read_csv(csv_path).copy()
    if id_column not in df.columns:
        raise ValueError(f"{source_name} missing required column: {id_column}")
    if label_column not in df.columns:
        raise ValueError(f"{source_name} missing required column: {label_column}")

    df[id_column] = df[id_column].map(_safe_text)
    df[label_column] = df[label_column].map(_safe_text)
    df = df[df[id_column].ne("") & df[label_column].isin(DEFAULT_ALLOWED_LABELS)].copy()
    df["data_source"] = source_name
    return df


def _distribution(df: pd.DataFrame, label_column: str) -> dict:
    if df.empty:
        return {}
    return {str(k): int(v) for k, v in df[label_column].value_counts().to_dict().items()}


def main() -> None:
    args = parse_args()

    gold_df = _load_dataset(args.gold_csv, args.label_column, args.id_column, "gold")
    gold_df = gold_df.drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)

    frames = [gold_df]
    silver_df = pd.DataFrame(columns=gold_df.columns)
    if args.include_silver:
        silver_df = _load_dataset(args.silver_csv, args.label_column, args.id_column, "silver")
        silver_df = silver_df[~silver_df[args.id_column].isin(gold_df[args.id_column])].copy()
        silver_df = silver_df.drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)
        frames.append(silver_df)

    merged_df = pd.concat(frames, axis=0, ignore_index=True)
    merged_df = merged_df.drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    stats = {
        "include_silver": bool(args.include_silver),
        "gold_rows": int(len(gold_df)),
        "silver_rows_used": int(len(silver_df)),
        "merged_rows": int(len(merged_df)),
        "gold_label_distribution": _distribution(gold_df, args.label_column),
        "silver_label_distribution": _distribution(silver_df, args.label_column),
        "merged_label_distribution": _distribution(merged_df, args.label_column),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
