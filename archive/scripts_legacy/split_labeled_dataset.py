#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ALLOWED_LABELS = {
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quality check and stratified split for labeled dataset")
    parser.add_argument("--input-csv", type=str, default="data/annotation/clean_labeled_dataset.csv")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--label-column", type=str, default="manual_label")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _label_dist(df: pd.DataFrame, label_column: str) -> dict:
    counts = df[label_column].value_counts().sort_index()
    return {k: int(v) for k, v in counts.items()}


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.valid_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-8:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    df = pd.read_csv(args.input_csv).copy()
    if args.label_column not in df.columns:
        raise ValueError(f"Missing label column: {args.label_column}")
    if args.id_column not in df.columns:
        raise ValueError(f"Missing id column: {args.id_column}")

    df[args.label_column] = df[args.label_column].map(_safe_text)
    df[args.id_column] = df[args.id_column].map(_safe_text)

    quality = {
        "total_rows": int(len(df)),
        "empty_label_count": int(df[args.label_column].eq("").sum()),
        "invalid_label_count": int((~df[args.label_column].isin(ALLOWED_LABELS) & df[args.label_column].ne("")).sum()),
        "duplicate_id_count": int(df[args.id_column].duplicated().sum()),
        "label_distribution": _label_dist(df, args.label_column),
    }

    valid_mask = (
        df[args.label_column].ne("")
        & df[args.label_column].isin(ALLOWED_LABELS)
        & df[args.id_column].ne("")
        & ~df[args.id_column].duplicated(keep="first")
    )
    clean_df = df[valid_mask].reset_index(drop=True)

    train_df, temp_df = train_test_split(
        clean_df,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=clean_df[args.label_column],
    )

    valid_ratio_in_temp = args.valid_ratio / (args.valid_ratio + args.test_ratio)
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=(1.0 - valid_ratio_in_temp),
        random_state=args.seed,
        stratify=temp_df[args.label_column],
    )

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    valid_df.to_csv(output_dir / "valid.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    label_names = sorted(clean_df[args.label_column].unique().tolist())
    label_mapping = {label: idx for idx, label in enumerate(label_names)}
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as f:
        json.dump(label_mapping, f, ensure_ascii=False, indent=2)

    split_stats = {
        "quality_check": quality,
        "clean_rows_used": int(len(clean_df)),
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "train_label_distribution": _label_dist(train_df, args.label_column),
        "valid_label_distribution": _label_dist(valid_df, args.label_column),
        "test_label_distribution": _label_dist(test_df, args.label_column),
    }
    with (output_dir / "split_stats.json").open("w", encoding="utf-8") as f:
        json.dump(split_stats, f, ensure_ascii=False, indent=2)

    with (output_dir / "quality_check.json").open("w", encoding="utf-8") as f:
        json.dump(quality, f, ensure_ascii=False, indent=2)

    print(json.dumps(split_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
