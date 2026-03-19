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


PASS_VALUES = {"1", "true", "pass", "approved", "approve", "yes", "y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build trusted silver from audited silver candidates")
    parser.add_argument("--silver-csv", type=str, default="data/annotation/silver/silver_candidates.csv")
    parser.add_argument("--audit-csv", type=str, default="data/annotation/silver/silver_audit_pool.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/silver/trusted_silver.csv")
    parser.add_argument(
        "--stats-json",
        type=str,
        default="data/annotation/silver/trusted_silver_stats.json",
    )
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--pred-score-threshold", type=float, default=0.97)
    parser.add_argument("--score-gap-threshold", type=float, default=0.25)
    parser.add_argument("--min-audit-samples-per-class", type=int, default=10)
    parser.add_argument("--min-pass-rate", type=float, default=0.85)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _parse_json(text: str) -> dict:
    raw = _safe_text(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _manual_pass(v) -> bool:
    return _safe_text(v).lower() in PASS_VALUES


def _prepare(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = df.copy()
    required = [
        id_column,
        "pred_label",
        "pred_score",
        "top2_score",
        "weak_label",
        "rule_hits",
        "subject_summary",
        "content_summary",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = ""
    for col in [id_column, "pred_label", "weak_label", "rule_hits", "subject_summary", "content_summary"]:
        out[col] = out[col].map(_safe_text)
    for col in ["pred_score", "top2_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out["score_gap"] = out["pred_score"] - out["top2_score"]
    out = out[out[id_column].ne("")].copy()
    out = out.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
    return out


def _has_strong_conflict(row: pd.Series) -> bool:
    pred_label = _safe_text(row.get("pred_label", ""))
    weak_label = _safe_text(row.get("weak_label", ""))
    if weak_label not in {"", "uncertain"} and weak_label != pred_label:
        return True
    rule_hits = _parse_json(row.get("rule_hits", ""))
    rule_weak = _safe_text(rule_hits.get("weak_label", ""))
    return rule_weak not in {"", "uncertain"} and rule_weak != pred_label


def main() -> None:
    args = parse_args()
    silver_df = _prepare(pd.read_csv(args.silver_csv), args.id_column)
    audit_df = _prepare(pd.read_csv(args.audit_csv), args.id_column)

    if "manual_check" not in audit_df.columns:
        audit_df["manual_check"] = ""
    audit_df["manual_check_pass"] = audit_df["manual_check"].map(_manual_pass)
    reviewed_df = audit_df[audit_df["manual_check"].map(lambda x: _safe_text(x) != "")].copy()

    class_stats = []
    approved_labels = []
    for label, group in reviewed_df.groupby("pred_label"):
        reviewed = int(len(group))
        passed = int(group["manual_check_pass"].sum())
        pass_rate = float(passed / reviewed) if reviewed > 0 else 0.0
        class_stats.append(
            {
                "label": label,
                "reviewed": reviewed,
                "passed": passed,
                "pass_rate": pass_rate,
            }
        )
        if reviewed >= args.min_audit_samples_per_class and pass_rate >= args.min_pass_rate:
            approved_labels.append(label)

    silver_df["strong_conflict"] = silver_df.apply(_has_strong_conflict, axis=1)
    trusted_df = silver_df[
        silver_df["pred_label"].isin(approved_labels)
        & (silver_df["pred_score"] >= args.pred_score_threshold)
        & (silver_df["score_gap"] >= args.score_gap_threshold)
        & (~silver_df["strong_conflict"])
    ].copy()

    output_columns = [
        args.id_column,
        "pred_label",
        "pred_score",
        "top2_score",
        "score_gap",
        "weak_label",
        "rule_hits",
        "subject_summary",
        "content_summary",
    ]
    trusted_df = trusted_df[output_columns]

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_df.to_csv(output_path, index=False)

    stats = {
        "silver_rows": int(len(silver_df)),
        "audit_rows": int(len(audit_df)),
        "reviewed_rows": int(len(reviewed_df)),
        "approved_labels": approved_labels,
        "thresholds": {
            "pred_score_threshold": args.pred_score_threshold,
            "score_gap_threshold": args.score_gap_threshold,
            "min_audit_samples_per_class": args.min_audit_samples_per_class,
            "min_pass_rate": args.min_pass_rate,
        },
        "class_audit_stats": class_stats,
        "trusted_silver_rows": int(len(trusted_df)),
        "trusted_label_distribution": {
            str(k): int(v) for k, v in trusted_df["pred_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
