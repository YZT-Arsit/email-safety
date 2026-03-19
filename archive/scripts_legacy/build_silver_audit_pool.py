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


DEFAULT_LABELS = (
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual audit pool for silver candidates")
    parser.add_argument("--input-csv", type=str, default="data/annotation/silver/silver_candidates.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/silver/silver_audit_pool.csv")
    parser.add_argument(
        "--stats-json",
        type=str,
        default="data/annotation/silver/silver_audit_pool_stats.json",
    )
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--total-size", type=int, default=150)
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--strategy", choices=["fixed_per_class", "score_bucket", "combined"], default="combined")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _score_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[0.0, 0.95, 0.97, 0.99, 1.000001],
        labels=["b0_95", "b95_97", "b97_99", "b99_100"],
        include_lowest=True,
    ).astype(str)


def _prepare(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = df.copy()
    required = [
        id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "weak_label",
        "rule_hits",
        "uncertainty",
        "subject_summary",
        "content_summary",
        "url_count",
        "attach_count",
        "suspicious_tld_count",
        "html_tag_count",
        "rcpt_count",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = ""

    for col in [id_column, "pred_label", "top2_label", "weak_label", "rule_hits", "subject_summary", "content_summary"]:
        out[col] = out[col].map(_safe_text)
    for col in ["pred_score", "top2_score", "uncertainty", "url_count", "attach_count", "suspicious_tld_count", "html_tag_count", "rcpt_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out[out[id_column].ne("")].copy()
    out = out.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
    out["score_bucket"] = _score_bucket(out["pred_score"])
    out["has_rule_signal"] = out["weak_label"].map(lambda x: int(_safe_text(x) not in {"", "uncertain"}))
    out["low_uncertainty_rank"] = out["uncertainty"].rank(method="first", ascending=True)
    return out


def _sample_fixed_per_class(df: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    parts = []
    for label in DEFAULT_LABELS:
        group = df[df["pred_label"] == label].copy()
        if group.empty:
            continue
        group = group.sort_values(["pred_score", "uncertainty"], ascending=[False, True])
        sampled = group.head(per_class)
        if len(sampled) < min(per_class, len(group)):
            sampled = group.sample(n=min(per_class, len(group)), random_state=seed)
        parts.append(sampled)
    if not parts:
        return df.head(0).copy()
    return pd.concat(parts, axis=0, ignore_index=True)


def _sample_score_bucket(df: pd.DataFrame, total_size: int) -> pd.DataFrame:
    picked = []
    buckets = ["b99_100", "b97_99", "b95_97", "b0_95"]
    labels = list(DEFAULT_LABELS)
    quota = max(1, total_size // max(1, len(labels) * len(buckets)))
    for label in labels:
        label_df = df[df["pred_label"] == label].copy()
        for bucket in buckets:
            group = label_df[label_df["score_bucket"] == bucket].copy()
            if group.empty:
                continue
            group = group.sort_values(["pred_score", "uncertainty"], ascending=[False, True])
            picked.append(group.head(quota))
    if not picked:
        return df.head(0).copy()
    return pd.concat(picked, axis=0, ignore_index=True)


def _sample_combined(df: pd.DataFrame, total_size: int, per_class: int, seed: int) -> pd.DataFrame:
    fixed = _sample_fixed_per_class(df, per_class=per_class, seed=seed)
    bucketed = _sample_score_bucket(df, total_size=total_size)
    merged = pd.concat([fixed, bucketed], axis=0, ignore_index=True)
    merged = merged.sort_values(
        ["has_rule_signal", "pred_score", "uncertainty"],
        ascending=[False, False, True],
    )
    merged = merged.drop_duplicates(subset=["id"], keep="first")
    if len(merged) < total_size:
        remainder = df[~df["id"].isin(merged["id"])].copy()
        remainder = remainder.sort_values(
            ["has_rule_signal", "pred_score", "uncertainty"],
            ascending=[False, False, True],
        )
        need = total_size - len(merged)
        merged = pd.concat([merged, remainder.head(need)], axis=0, ignore_index=True)
    merged = merged.head(total_size).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = _prepare(df, args.id_column)

    if args.strategy == "fixed_per_class":
        audit_df = _sample_fixed_per_class(df, args.per_class, args.seed)
    elif args.strategy == "score_bucket":
        audit_df = _sample_score_bucket(df, args.total_size)
    else:
        audit_df = _sample_combined(df, args.total_size, args.per_class, args.seed)

    audit_df = audit_df.drop_duplicates(subset=[args.id_column], keep="first").head(args.total_size).reset_index(drop=True)
    audit_df["manual_check"] = ""
    audit_df["notes"] = ""

    output_columns = [
        args.id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "weak_label",
        "rule_hits",
        "uncertainty",
        "subject_summary",
        "content_summary",
        "url_count",
        "attach_count",
        "suspicious_tld_count",
        "html_tag_count",
        "rcpt_count",
        "manual_check",
        "notes",
    ]
    audit_df = audit_df[output_columns]

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(output_path, index=False)

    stats = {
        "input_rows": int(len(df)),
        "audit_pool_rows": int(len(audit_df)),
        "strategy": args.strategy,
        "total_size": int(args.total_size),
        "per_class": int(args.per_class),
        "pred_label_distribution": {
            str(k): int(v) for k, v in audit_df["pred_label"].value_counts(dropna=False).to_dict().items()
        },
        "score_bucket_distribution": {
            str(k): int(v)
            for k, v in _score_bucket(audit_df["pred_score"]).value_counts(dropna=False).to_dict().items()
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
