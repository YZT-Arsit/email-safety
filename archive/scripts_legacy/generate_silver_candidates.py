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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high-confidence silver candidates from unlabeled predictions")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="outputs/predictions/all_unlabeled_predictions.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/annotation/silver/silver_candidates.csv",
    )
    parser.add_argument(
        "--stats-json",
        type=str,
        default="data/annotation/silver/silver_candidates_stats.json",
    )
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--pred-score-threshold", type=float, default=0.95)
    parser.add_argument("--score-gap-threshold", type=float, default=0.20)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _single_line_text(v) -> str:
    text = _safe_text(v)
    return " ".join(text.replace("\x00", " ").split())


def _parse_json(text: str) -> dict:
    raw = _safe_text(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _has_strong_conflict(row: pd.Series) -> bool:
    pred_label = _safe_text(row.get("pred_label", ""))
    weak_label = _safe_text(row.get("weak_label", ""))
    if weak_label and weak_label != "uncertain" and weak_label != pred_label:
        return True

    rule_hits = _parse_json(row.get("rule_hits", ""))
    rule_weak = _safe_text(rule_hits.get("weak_label", ""))
    if rule_weak and rule_weak != "uncertain" and rule_weak != pred_label:
        return True
    return False


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv).copy()

    required_cols = [
        args.id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "uncertainty",
        "weak_label",
        "rule_hits",
        "subject_summary",
        "content_summary",
        "url_count",
        "attach_count",
        "suspicious_tld_count",
        "html_tag_count",
        "rcpt_count",
    ]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df[args.id_column] = df[args.id_column].map(_safe_text)
    for col in ["pred_label", "top2_label", "weak_label", "rule_hits"]:
        df[col] = df[col].map(_safe_text)
    for col in ["subject_summary", "content_summary"]:
        df[col] = df[col].map(_single_line_text)
    for col in ["pred_score", "top2_score", "uncertainty", "url_count", "attach_count", "suspicious_tld_count", "html_tag_count", "rcpt_count"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df = df[df[args.id_column].ne("")].copy()
    df = df.drop_duplicates(subset=[args.id_column], keep="first").reset_index(drop=True)
    df["score_gap"] = df["pred_score"] - df["top2_score"]
    df["strong_conflict"] = df.apply(_has_strong_conflict, axis=1)

    candidates = df[
        (df["pred_score"] >= args.pred_score_threshold)
        & (df["score_gap"] >= args.score_gap_threshold)
        & (~df["strong_conflict"])
    ].copy()

    candidates["manual_check"] = ""
    candidates["notes"] = ""

    output_columns = [
        args.id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "uncertainty",
        "score_gap",
        "weak_label",
        "rule_hits",
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
    candidates = candidates[output_columns]

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)

    stats = {
        "input_rows": int(len(df)),
        "silver_candidate_rows": int(len(candidates)),
        "thresholds": {
            "pred_score_threshold": args.pred_score_threshold,
            "score_gap_threshold": args.score_gap_threshold,
        },
        "strong_conflict_rows": int(df["strong_conflict"].sum()),
        "pred_label_distribution": {
            str(k): int(v) for k, v in candidates["pred_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
