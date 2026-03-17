#!/usr/bin/env python
"""Merge Gold v2 and consensus trusted silver into a weighted semi-supervised training set.
Inputs: gold and silver CSVs. Outputs: merged training CSV and dataset statistics JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build semi-supervised training dataset from Gold v2 and consensus trusted silver")
    parser.add_argument("--gold-csv", type=str, default="data/annotation/gold/gold_v2.csv")
    parser.add_argument("--silver-csv", type=str, default="data/annotation/silver/consensus_trusted_silver.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/semi_supervised/semi_supervised_train.csv")
    parser.add_argument("--stats-json", type=str, default="data/annotation/semi_supervised/semi_supervised_stats.json")
    parser.add_argument("--gold-weight", type=float, default=1.0)
    parser.add_argument("--silver-weight", type=float, default=0.5)
    parser.add_argument("--gold-label-column", type=str, default="manual_label")
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _concat_gold_text(df: pd.DataFrame) -> pd.Series:
    for col in ["subject", "content", "doccontent"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_safe_text)
    return (df["subject"] + " [SEP] " + df["content"] + " [SEP] " + df["doccontent"]).str.replace(r"\s+", " ", regex=True).str.strip()


def _concat_silver_text(df: pd.DataFrame) -> pd.Series:
    for col in ["subject_summary", "content_summary"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_safe_text)
    return (df["subject_summary"] + " [SEP] " + df["content_summary"]).str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    args = parse_args()
    gold_df = pd.read_csv(args.gold_csv).copy()
    silver_df = pd.read_csv(args.silver_csv).copy() if Path(args.silver_csv).exists() else pd.DataFrame()

    if "id" not in gold_df.columns:
        raise ValueError("gold csv missing id column")
    if args.gold_label_column not in gold_df.columns:
        raise ValueError(f"gold csv missing label column: {args.gold_label_column}")

    gold_df["id"] = gold_df["id"].map(_safe_text)
    gold_df[args.gold_label_column] = gold_df[args.gold_label_column].map(_safe_text)
    gold_df = gold_df[gold_df["id"].ne("") & gold_df[args.gold_label_column].ne("")].drop_duplicates(subset=["id"], keep="last")
    gold_out = pd.DataFrame({
        "id": gold_df["id"],
        "text": _concat_gold_text(gold_df),
        "label": gold_df[args.gold_label_column],
        "data_source": "gold",
        "sample_weight": float(args.gold_weight),
    })

    silver_out = pd.DataFrame(columns=gold_out.columns)
    if not silver_df.empty:
        for col in ["id", "label"]:
            if col not in silver_df.columns:
                raise ValueError(f"silver csv missing required column: {col}")
        silver_df["id"] = silver_df["id"].map(_safe_text)
        silver_df["label"] = silver_df["label"].map(_safe_text)
        silver_df = silver_df[silver_df["id"].ne("") & silver_df["label"].ne("")]
        silver_df = silver_df[~silver_df["id"].isin(set(gold_out["id"]))].drop_duplicates(subset=["id"], keep="last")
        silver_out = pd.DataFrame({
            "id": silver_df["id"],
            "text": _concat_silver_text(silver_df),
            "label": silver_df["label"],
            "data_source": "silver",
            "sample_weight": float(args.silver_weight),
        })

    merged = pd.concat([gold_out, silver_out], ignore_index=True)
    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    stats = {
        "gold_rows": int(len(gold_out)),
        "silver_rows": int(len(silver_out)),
        "merged_rows": int(len(merged)),
        "gold_label_distribution": {str(k): int(v) for k, v in gold_out["label"].value_counts(dropna=False).to_dict().items()},
        "silver_label_distribution": {str(k): int(v) for k, v in silver_out["label"].value_counts(dropna=False).to_dict().items()} if not silver_out.empty else {},
        "merged_label_distribution": {str(k): int(v) for k, v in merged["label"].value_counts(dropna=False).to_dict().items()},
        "silver_weight": float(args.silver_weight),
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
