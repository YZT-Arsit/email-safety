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


LONG_TAIL_LABELS = {"impersonation", "malicious_link_or_attachment"}
BOUNDARY_PAIRS = {
    frozenset({"phishing", "impersonation"}),
    frozenset({"advertisement", "black_industry_or_policy_violation"}),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round-2 targeted relabel pool")
    parser.add_argument("--predictions-csv", type=str, default="outputs/predictions/all_unlabeled_predictions.csv")
    parser.add_argument("--silver-csv", type=str, default="data/annotation/silver/silver_candidates.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/relabel_round2/relabel_round2_pool.csv")
    parser.add_argument(
        "--stats-json",
        type=str,
        default="data/annotation/relabel_round2/relabel_round2_stats.json",
    )
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--total-size", type=int, default=500)
    parser.add_argument("--conflict-quota", type=int, default=180)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.60)
    parser.add_argument("--low-confidence-quota", type=int, default=120)
    parser.add_argument("--boundary-margin", type=float, default=0.12)
    parser.add_argument("--boundary-quota", type=int, default=120)
    parser.add_argument("--long-tail-quota", type=int, default=180)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _parse_rule_hits(text: str) -> dict:
    raw = _safe_text(text)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def _prepare(df: pd.DataFrame, id_column: str, source_name: str) -> pd.DataFrame:
    out = df.copy()
    required = [
        id_column,
        "pred_label",
        "pred_score",
        "top2_label",
        "top2_score",
        "uncertainty",
        "weak_label",
        "rule_hits",
        "subject_summary",
        "content_summary",
    ]
    for col in required:
        if col not in out.columns:
            out[col] = ""
    for col in [id_column, "pred_label", "top2_label", "weak_label", "rule_hits", "subject_summary", "content_summary"]:
        out[col] = out[col].map(_safe_text)
    for col in ["pred_score", "top2_score", "uncertainty"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    out = out[out[id_column].ne("")].copy()
    out = out.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
    out["score_gap"] = out["pred_score"] - out["top2_score"]
    out["source_pool"] = source_name
    return out


def _is_conflict(row: pd.Series) -> bool:
    pred_label = _safe_text(row.get("pred_label", ""))
    weak_label = _safe_text(row.get("weak_label", ""))
    if weak_label not in {"", "uncertain"} and weak_label != pred_label:
        return True
    rule_hits = _parse_rule_hits(row.get("rule_hits", ""))
    rule_weak = _safe_text(rule_hits.get("weak_label", ""))
    return rule_weak not in {"", "uncertain"} and rule_weak != pred_label


def _pick_conflict(df: pd.DataFrame, quota: int) -> pd.DataFrame:
    picked = df[df.apply(_is_conflict, axis=1)].copy()
    picked = picked.sort_values(["pred_score", "score_gap"], ascending=[True, True]).head(quota)
    picked["selection_reason"] = "strong_conflict"
    return picked


def _pick_low_confidence(df: pd.DataFrame, threshold: float, quota: int) -> pd.DataFrame:
    picked = df[df["pred_score"] < threshold].copy()
    picked["long_tail_priority"] = picked["pred_label"].isin(LONG_TAIL_LABELS).astype(int)
    picked = picked.sort_values(["long_tail_priority", "pred_score", "score_gap"], ascending=[False, True, True]).head(quota)
    picked["selection_reason"] = "low_confidence"
    return picked


def _pick_boundary(df: pd.DataFrame, margin: float, quota: int) -> pd.DataFrame:
    pair_mask = df.apply(
        lambda row: frozenset({_safe_text(row["pred_label"]), _safe_text(row["top2_label"])}) in BOUNDARY_PAIRS,
        axis=1,
    )
    picked = df[pair_mask & (df["score_gap"] < margin)].copy()
    picked = picked.sort_values(["score_gap", "pred_score"], ascending=[True, True]).head(quota)
    picked["selection_reason"] = "boundary_focus"
    return picked


def _pick_long_tail(pred_df: pd.DataFrame, silver_df: pd.DataFrame, quota: int) -> pd.DataFrame:
    merged = pd.concat([pred_df, silver_df], axis=0, ignore_index=True)
    picked = merged[
        merged["pred_label"].isin(LONG_TAIL_LABELS) | merged["top2_label"].isin(LONG_TAIL_LABELS)
    ].copy()
    picked["source_priority"] = picked["source_pool"].map(lambda x: 0 if x == "predictions" else 1)
    picked = picked.sort_values(
        ["source_priority", "pred_label", "score_gap", "pred_score"],
        ascending=[True, True, True, False],
    ).head(quota)
    picked["selection_reason"] = "long_tail_priority"
    return picked


def _merge_candidates(parts: list[pd.DataFrame], id_column: str, total_size: int) -> pd.DataFrame:
    merged = pd.concat(parts, axis=0, ignore_index=True)
    merged["selection_reason"] = merged.groupby(id_column)["selection_reason"].transform(
        lambda s: "|".join(sorted({x for x in s if _safe_text(x)}))
    )
    merged = merged.sort_values(
        ["pred_label", "score_gap", "uncertainty", "pred_score"],
        ascending=[True, True, False, True],
    )
    merged = merged.drop_duplicates(subset=[id_column], keep="first").head(total_size).reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    pred_df = _prepare(pd.read_csv(args.predictions_csv), args.id_column, "predictions")
    silver_df = _prepare(pd.read_csv(args.silver_csv), args.id_column, "silver_candidates")

    conflict_df = _pick_conflict(pred_df, args.conflict_quota)
    low_conf_df = _pick_low_confidence(pred_df, args.low_confidence_threshold, args.low_confidence_quota)
    boundary_df = _pick_boundary(pred_df, args.boundary_margin, args.boundary_quota)
    long_tail_df = _pick_long_tail(pred_df, silver_df, args.long_tail_quota)

    merged = _merge_candidates(
        [conflict_df, low_conf_df, boundary_df, long_tail_df],
        args.id_column,
        args.total_size,
    )
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
        "manual_label",
        "notes",
    ]
    for col in output_columns:
        if col not in merged.columns:
            merged[col] = ""
    merged = merged[output_columns]

    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    stats = {
        "input_prediction_rows": int(len(pred_df)),
        "input_silver_rows": int(len(silver_df)),
        "round2_pool_rows": int(len(merged)),
        "quotas": {
            "conflict_quota": args.conflict_quota,
            "low_confidence_quota": args.low_confidence_quota,
            "boundary_quota": args.boundary_quota,
            "long_tail_quota": args.long_tail_quota,
            "total_size": args.total_size,
        },
        "strategy_counts_before_dedup": {
            "strong_conflict": int(len(conflict_df)),
            "low_confidence": int(len(low_conf_df)),
            "boundary_focus": int(len(boundary_df)),
            "long_tail_priority": int(len(long_tail_df)),
        },
        "selection_reason_distribution": {
            str(k): int(v) for k, v in merged["selection_reason"].value_counts(dropna=False).to_dict().items()
        },
        "pred_label_distribution": {
            str(k): int(v) for k, v in merged["pred_label"].value_counts(dropna=False).to_dict().items()
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
