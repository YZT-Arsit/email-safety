#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))


DEFAULT_HIGH_RISK_LABELS = ("phishing", "malicious_link_or_attachment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select high-priority relabel candidates from unlabeled predictions")
    parser.add_argument(
        "--input-csv",
        type=str,
        default="outputs/predictions/all_unlabeled_predictions.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="outputs/relabel_pool/relabel_candidates.csv",
    )
    parser.add_argument(
        "--summary-json",
        type=str,
        default="outputs/relabel_pool/relabel_summary.json",
    )
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--boundary-margin", type=float, default=0.10)
    parser.add_argument(
        "--high-risk-labels",
        type=str,
        default=",".join(DEFAULT_HIGH_RISK_LABELS),
        help="Comma-separated predicted labels to prioritize",
    )
    parser.add_argument("--max-low-confidence", type=int, default=500)
    parser.add_argument("--max-high-risk", type=int, default=500)
    parser.add_argument("--max-conflict", type=int, default=500)
    parser.add_argument("--max-boundary", type=int, default=500)
    parser.add_argument("--max-total", type=int, default=1500)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _parse_rule_hits(text: str) -> Dict:
    raw = _safe_text(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _normalize_prediction_frame(df: pd.DataFrame, id_column: str) -> pd.DataFrame:
    out = df.copy()
    required_cols = [
        id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "uncertainty",
        "weak_label",
        "weak_label_scores",
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
        if col not in out.columns:
            out[col] = ""

    text_cols = [
        id_column,
        "pred_label",
        "top2_label",
        "weak_label",
        "weak_label_scores",
        "rule_hits",
        "subject_summary",
        "content_summary",
    ]
    for col in text_cols:
        out[col] = out[col].map(_safe_text)

    numeric_cols = [
        "pred_score",
        "top2_score",
        "uncertainty",
        "url_count",
        "attach_count",
        "suspicious_tld_count",
        "html_tag_count",
        "rcpt_count",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    out = out[out[id_column].ne("")].copy()
    out = out.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
    out["score_gap"] = out["pred_score"] - out["top2_score"]
    return out


def _select_low_confidence(df: pd.DataFrame, threshold: float, max_rows: int) -> pd.DataFrame:
    picked = df[df["pred_score"] < threshold].copy()
    picked = picked.sort_values(["pred_score", "score_gap"], ascending=[True, True]).head(max_rows)
    picked["selection_reason"] = "low_confidence"
    return picked


def _select_high_risk(df: pd.DataFrame, high_risk_labels: List[str], max_rows: int) -> pd.DataFrame:
    picked = df[df["pred_label"].isin(high_risk_labels)].copy()
    picked = picked.sort_values(["pred_score", "uncertainty"], ascending=[False, False]).head(max_rows)
    picked["selection_reason"] = "high_risk"
    return picked


def _select_rule_conflict(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    rule_info = df["rule_hits"].map(_parse_rule_hits)
    weak_from_rule = rule_info.map(lambda x: _safe_text(x.get("weak_label", "")))
    risk_flags = rule_info.map(lambda x: x.get("risk_flags", {}) if isinstance(x, dict) else {})
    has_strong_flag = risk_flags.map(lambda x: int(any(int(v) > 0 for v in x.values())))
    conflict = (weak_from_rule.ne("")) & (weak_from_rule.ne("uncertain")) & (weak_from_rule.ne(df["pred_label"]))
    strong_conflict = has_strong_flag.eq(1) & conflict
    picked = df[conflict | strong_conflict].copy()
    picked["selection_reason"] = "rule_model_conflict"
    picked["rule_weak_label"] = weak_from_rule[conflict | strong_conflict].values
    picked = picked.sort_values(["pred_score", "score_gap"], ascending=[True, True]).head(max_rows)
    return picked


def _select_boundary(df: pd.DataFrame, margin: float, max_rows: int) -> pd.DataFrame:
    picked = df[df["score_gap"] < margin].copy()
    picked = picked.sort_values(["score_gap", "pred_score"], ascending=[True, True]).head(max_rows)
    picked["selection_reason"] = "boundary_ambiguous"
    return picked


def _merge_candidates(candidates: List[pd.DataFrame], id_column: str, max_total: int) -> pd.DataFrame:
    if not candidates:
        return pd.DataFrame()
    merged = pd.concat(candidates, axis=0, ignore_index=True)
    merged["selection_reason"] = merged["selection_reason"].map(_safe_text)
    merged["selection_reason"] = merged.groupby(id_column)["selection_reason"].transform(
        lambda s: "|".join(sorted({x for x in s if x}))
    )
    merged = merged.sort_values(
        ["uncertainty", "score_gap", "pred_score"],
        ascending=[False, True, True],
    )
    merged = merged.drop_duplicates(subset=[id_column], keep="first").head(max_total).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    df = _normalize_prediction_frame(df, args.id_column)

    high_risk_labels = [_safe_text(x) for x in args.high_risk_labels.split(",") if _safe_text(x)]

    low_conf = _select_low_confidence(df, args.low_confidence_threshold, args.max_low_confidence)
    high_risk = _select_high_risk(df, high_risk_labels, args.max_high_risk)
    conflict = _select_rule_conflict(df, args.max_conflict)
    boundary = _select_boundary(df, args.boundary_margin, args.max_boundary)

    merged = _merge_candidates(
        [low_conf, high_risk, conflict, boundary],
        id_column=args.id_column,
        max_total=args.max_total,
    )

    if "rule_weak_label" not in merged.columns:
        merged["rule_weak_label"] = ""
    merged["manual_label"] = ""
    merged["notes"] = ""

    output_columns = [
        args.id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "weak_label",
        "rule_hits",
        "uncertainty",
        "score_gap",
        "selection_reason",
        "subject_summary",
        "content_summary",
        "url_count",
        "attach_count",
        "suspicious_tld_count",
        "html_tag_count",
        "rcpt_count",
        "manual_label",
        "notes",
    ]
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[output_columns]

    output_path = Path(args.output_csv)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    summary = {
        "input_rows": int(len(df)),
        "selected_rows": int(len(merged)),
        "thresholds": {
            "low_confidence_threshold": args.low_confidence_threshold,
            "boundary_margin": args.boundary_margin,
            "high_risk_labels": high_risk_labels,
        },
        "strategy_counts_before_dedup": {
            "low_confidence": int(len(low_conf)),
            "high_risk": int(len(high_risk)),
            "rule_model_conflict": int(len(conflict)),
            "boundary_ambiguous": int(len(boundary)),
        },
        "selected_reason_distribution": {
            str(k): int(v)
            for k, v in merged["selection_reason"].value_counts(dropna=False).to_dict().items()
        },
        "selected_pred_label_distribution": {
            str(k): int(v)
            for k, v in merged["pred_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
