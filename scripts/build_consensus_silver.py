#!/usr/bin/env python
"""Build high-precision trusted silver labels from multi-teacher consensus and rule signals.
Inputs: teacher prediction CSVs. Outputs: consensus trusted silver CSV and statistics JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

HIGH_RISK = {"phishing", "malicious_link_or_attachment", "impersonation"}
DEFAULT_CLASS_THRESHOLDS = {
    "advertisement": 0.95,
    "black_industry_or_policy_violation": 0.95,
    "phishing": 0.98,
    "malicious_link_or_attachment": 0.98,
    "impersonation": 0.97,
}
DEFAULT_CLASS_MARGINS = {
    "advertisement": 0.20,
    "black_industry_or_policy_violation": 0.20,
    "phishing": 0.25,
    "malicious_link_or_attachment": 0.25,
    "impersonation": 0.25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build high-precision consensus trusted silver from multiple teachers")
    parser.add_argument("--text-lr-csv", type=str, required=True)
    parser.add_argument("--structured-lgbm-csv", type=str, required=True)
    parser.add_argument("--fusion-lr-csv", type=str, required=True)
    parser.add_argument("--mbert-csv", type=str, required=True)
    parser.add_argument("--dapt-mbert-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/annotation/silver/consensus_trusted_silver.csv")
    parser.add_argument("--stats-json", type=str, default="data/annotation/silver/consensus_trusted_silver_stats.json")
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--min-teachers-agree", type=int, default=4)
    parser.add_argument("--default-score-threshold", type=float, default=0.95)
    parser.add_argument("--default-margin-threshold", type=float, default=0.20)
    parser.add_argument("--class-thresholds-json", type=str, default="")
    parser.add_argument("--class-margins-json", type=str, default="")
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
    except Exception:
        return {}


def _load_teacher(path: str, teacher_name: str, id_column: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    for col in [id_column, "pred_label", "pred_score", "top2_label", "top2_score", "weak_label", "rule_hits", "subject_summary", "content_summary"]:
        if col not in df.columns:
            df[col] = ""
    df[id_column] = df[id_column].map(_safe_text)
    df["pred_label"] = df["pred_label"].map(_safe_text)
    for col in ["pred_score", "top2_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["score_gap"] = df["pred_score"] - df["top2_score"]
    df = df[df[id_column].ne("") & df["pred_label"].ne("")].drop_duplicates(subset=[id_column], keep="last")
    rename_map = {
        "pred_label": f"{teacher_name}__pred_label",
        "pred_score": f"{teacher_name}__pred_score",
        "top2_label": f"{teacher_name}__top2_label",
        "top2_score": f"{teacher_name}__top2_score",
        "score_gap": f"{teacher_name}__score_gap",
    }
    keep_cols = [id_column] + list(rename_map.keys()) + ["weak_label", "rule_hits", "subject_summary", "content_summary"]
    out = df[keep_cols].rename(columns=rename_map)
    out = out.rename(
        columns={
            "weak_label": f"{teacher_name}__weak_label",
            "rule_hits": f"{teacher_name}__rule_hits",
            "subject_summary": f"{teacher_name}__subject_summary",
            "content_summary": f"{teacher_name}__content_summary",
        }
    )
    return out


def _class_thresholds(raw: str, defaults: Dict[str, float]) -> Dict[str, float]:
    if not raw:
        return defaults
    payload = _parse_json(raw)
    out = dict(defaults)
    for k, v in payload.items():
        out[str(k)] = float(v)
    return out


def _extract_rule_label(rule_hits: str) -> str:
    rule_payload = _parse_json(rule_hits)
    return _safe_text(rule_payload.get("weak_label", ""))


def _has_strong_conflict(label: str, weak_label: str, rule_hits: str) -> bool:
    weak_label = _safe_text(weak_label)
    rule_label = _extract_rule_label(rule_hits)
    if weak_label not in {"", "uncertain"} and weak_label != label:
        return True
    if rule_label not in {"", "uncertain"} and rule_label != label:
        return True
    return False


def main() -> None:
    args = parse_args()
    teachers = {
        "text_lr": args.text_lr_csv,
        "structured_lgbm": args.structured_lgbm_csv,
        "fusion_lr": args.fusion_lr_csv,
        "mbert": args.mbert_csv,
        "dapt_mbert": args.dapt_mbert_csv,
    }
    thresholds = _class_thresholds(args.class_thresholds_json, DEFAULT_CLASS_THRESHOLDS)
    margins = _class_thresholds(args.class_margins_json, DEFAULT_CLASS_MARGINS)

    merged = None
    for teacher_name, path in teachers.items():
        teacher_df = _load_teacher(path, teacher_name, args.id_column)
        merged = teacher_df if merged is None else merged.merge(teacher_df, on=args.id_column, how="inner")

    if merged is None or merged.empty:
        raise ValueError("No overlapping teacher predictions found. Please check input files.")

    records: List[dict] = []
    filtered_reason_counts: Dict[str, int] = {}
    agreement_distribution: Dict[str, int] = {}

    for _, row in merged.iterrows():
        labels = [
            _safe_text(row.get(f"{teacher}__pred_label", ""))
            for teacher in teachers.keys()
        ]
        label_counts = pd.Series(labels).value_counts()
        if label_counts.empty:
            filtered_reason_counts["missing_teacher_votes"] = filtered_reason_counts.get("missing_teacher_votes", 0) + 1
            continue

        consensus_label = str(label_counts.index[0])
        agree_count = int(label_counts.iloc[0])
        agreement_distribution[str(agree_count)] = agreement_distribution.get(str(agree_count), 0) + 1
        if agree_count < args.min_teachers_agree:
            filtered_reason_counts["teacher_agreement_below_threshold"] = filtered_reason_counts.get("teacher_agreement_below_threshold", 0) + 1
            continue

        agreeing_teachers = [teacher for teacher in teachers.keys() if _safe_text(row.get(f"{teacher}__pred_label", "")) == consensus_label]
        teacher_scores = [float(row.get(f"{teacher}__pred_score", 0.0)) for teacher in agreeing_teachers]
        teacher_margins = [float(row.get(f"{teacher}__score_gap", 0.0)) for teacher in agreeing_teachers]
        mean_score = sum(teacher_scores) / max(1, len(teacher_scores))
        min_score = min(teacher_scores) if teacher_scores else 0.0
        min_margin = min(teacher_margins) if teacher_margins else 0.0

        min_required_score = thresholds.get(consensus_label, args.default_score_threshold)
        min_required_margin = margins.get(consensus_label, args.default_margin_threshold)
        if min_score < min_required_score:
            filtered_reason_counts["score_below_threshold"] = filtered_reason_counts.get("score_below_threshold", 0) + 1
            continue
        if min_margin < min_required_margin:
            filtered_reason_counts["margin_below_threshold"] = filtered_reason_counts.get("margin_below_threshold", 0) + 1
            continue

        weak_label = _safe_text(row.get("fusion_lr__weak_label", "")) or _safe_text(row.get("text_lr__weak_label", ""))
        rule_hits = _safe_text(row.get("fusion_lr__rule_hits", "")) or _safe_text(row.get("text_lr__rule_hits", ""))
        if _has_strong_conflict(consensus_label, weak_label, rule_hits):
            filtered_reason_counts["strong_rule_conflict"] = filtered_reason_counts.get("strong_rule_conflict", 0) + 1
            continue

        subject_summary = _safe_text(row.get("fusion_lr__subject_summary", "")) or _safe_text(row.get("text_lr__subject_summary", ""))
        content_summary = _safe_text(row.get("fusion_lr__content_summary", "")) or _safe_text(row.get("text_lr__content_summary", ""))
        records.append(
            {
                "id": _safe_text(row[args.id_column]),
                "label": consensus_label,
                "pred_label": consensus_label,
                "teacher_agree_count": agree_count,
                "agreeing_teachers": "|".join(agreeing_teachers),
                "mean_pred_score": mean_score,
                "min_pred_score": min_score,
                "min_score_gap": min_margin,
                "weak_label": weak_label,
                "rule_hits": rule_hits,
                "subject_summary": subject_summary,
                "content_summary": content_summary,
                "is_high_risk": int(consensus_label in HIGH_RISK),
            }
        )

    trusted_df = pd.DataFrame(records)
    output_path = Path(args.output_csv)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    trusted_df.to_csv(output_path, index=False)

    stats = {
        "input_rows": int(len(merged)),
        "trusted_silver_rows": int(len(trusted_df)),
        "min_teachers_agree": args.min_teachers_agree,
        "class_thresholds": thresholds,
        "class_margins": margins,
        "teacher_agreement_distribution": agreement_distribution,
        "trusted_label_distribution": {
            str(k): int(v) for k, v in trusted_df["label"].value_counts(dropna=False).to_dict().items()
        } if not trusted_df.empty else {},
        "filtered_reason_counts": filtered_reason_counts,
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
