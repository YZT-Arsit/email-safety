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


ALLOWED_LABELS = {
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gold v2 from Gold v1 and round2 manual annotations")
    parser.add_argument("--gold-v1-csv", type=str, default="data/annotation/clean_labeled_dataset.csv")
    parser.add_argument("--round2-csv", type=str, default="data/annotation/relabel_round2/relabel_round2_pool.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/gold/gold_v2.csv")
    parser.add_argument("--stats-json", type=str, default="data/annotation/gold/gold_v2_stats.json")
    parser.add_argument("--issues-csv", type=str, default="data/annotation/gold/gold_v2_issues.csv")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--label-column", type=str, default="manual_label")
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _prepare(df: pd.DataFrame, id_column: str, label_column: str, source_name: str, source_order: int) -> pd.DataFrame:
    out = df.copy()
    if id_column not in out.columns:
        raise ValueError(f"{source_name} missing id column: {id_column}")
    if label_column not in out.columns:
        raise ValueError(f"{source_name} missing label column: {label_column}")
    out[id_column] = out[id_column].map(_safe_text)
    out[label_column] = out[label_column].map(_safe_text)
    out["data_source"] = source_name
    out["source_order"] = source_order
    return out


def _dist(df: pd.DataFrame, label_column: str) -> dict:
    return {str(k): int(v) for k, v in df[label_column].value_counts(dropna=False).to_dict().items()}


def main() -> None:
    args = parse_args()
    gold_v1_df = _prepare(pd.read_csv(args.gold_v1_csv), args.id_column, args.label_column, "gold_v1", 0)
    round2_df = _prepare(pd.read_csv(args.round2_csv), args.id_column, args.label_column, "round2_gold", 1)

    combined_raw = pd.concat([gold_v1_df, round2_df], axis=0, ignore_index=True)
    combined_raw["id_missing"] = combined_raw[args.id_column].eq("")
    combined_raw["label_missing"] = combined_raw[args.label_column].eq("")
    combined_raw["label_invalid"] = ~combined_raw[args.label_column].isin(ALLOWED_LABELS) & ~combined_raw["label_missing"]
    combined_raw["duplicate_id_seen"] = combined_raw[args.id_column].duplicated(keep=False) & ~combined_raw["id_missing"]
    combined_raw["issue_reason"] = combined_raw.apply(
        lambda row: "|".join(
            reason
            for flag, reason in [
                (row["id_missing"], "id_missing"),
                (row["label_missing"], "label_missing"),
                (row["label_invalid"], "label_invalid"),
            ]
            if flag
        ),
        axis=1,
    )

    issues_df = combined_raw[combined_raw["issue_reason"].ne("")].copy()
    valid_df = combined_raw[combined_raw["issue_reason"].eq("")].copy()
    valid_df = valid_df.sort_values(["source_order"]).drop_duplicates(subset=[args.id_column], keep="last").reset_index(drop=True)

    gold_v1_valid = gold_v1_df[
        gold_v1_df[args.id_column].ne("") & gold_v1_df[args.label_column].isin(ALLOWED_LABELS)
    ].copy()
    gold_v1_valid = gold_v1_valid.drop_duplicates(subset=[args.id_column], keep="last").reset_index(drop=True)

    new_ids = set(valid_df[args.id_column]) - set(gold_v1_valid[args.id_column])
    overlap_ids = set(valid_df[args.id_column]) & set(gold_v1_valid[args.id_column])

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    issues_path = Path(args.issues_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.parent.mkdir(parents=True, exist_ok=True)

    valid_df.to_csv(output_path, index=False)
    issues_df.to_csv(issues_path, index=False)

    stats = {
        "gold_v1_rows": int(len(gold_v1_valid)),
        "round2_input_rows": int(len(round2_df)),
        "gold_v2_rows": int(len(valid_df)),
        "issues_rows": int(len(issues_df)),
        "new_ids_added": int(len(new_ids)),
        "overlap_ids_updated_or_kept": int(len(overlap_ids)),
        "gold_v1_distribution": _dist(gold_v1_valid, args.label_column),
        "gold_v2_distribution": _dist(valid_df, args.label_column),
        "increment_vs_gold_v1": {
            label: int(_dist(valid_df, args.label_column).get(label, 0) - _dist(gold_v1_valid, args.label_column).get(label, 0))
            for label in sorted(ALLOWED_LABELS)
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
