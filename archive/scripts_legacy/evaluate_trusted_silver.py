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
    parser = argparse.ArgumentParser(description="Evaluate trusted silver quality from audit results")
    parser.add_argument("--audit-csv", type=str, default="data/annotation/silver/silver_audit_pool.csv")
    parser.add_argument("--trusted-silver-csv", type=str, default="data/annotation/silver/trusted_silver.csv")
    parser.add_argument("--output-json", type=str, default="data/annotation/silver/trusted_silver_eval.json")
    parser.add_argument("--output-md", type=str, default="data/annotation/silver/trusted_silver_eval.md")
    parser.add_argument("--label-column", type=str, default="pred_label")
    parser.add_argument("--decision-column", type=str, default="manual_check")
    parser.add_argument("--include-threshold", type=float, default=0.85)
    parser.add_argument("--review-threshold", type=float, default=0.70)
    parser.add_argument("--min-samples-per-class", type=int, default=10)
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _manual_pass(v) -> bool:
    return _safe_text(v).lower() in PASS_VALUES


def _recommend(pass_rate: float, reviewed: int, include_threshold: float, review_threshold: float, min_samples: int) -> str:
    if reviewed < min_samples:
        return "review_more"
    if pass_rate >= include_threshold:
        return "include"
    if pass_rate >= review_threshold:
        return "review_more"
    return "exclude"


def main() -> None:
    args = parse_args()
    audit_df = pd.read_csv(args.audit_csv).copy()
    if args.label_column not in audit_df.columns:
        raise ValueError(f"audit csv missing label column: {args.label_column}")
    if args.decision_column not in audit_df.columns:
        raise ValueError(f"audit csv missing decision column: {args.decision_column}")

    audit_df[args.label_column] = audit_df[args.label_column].map(_safe_text)
    audit_df[args.decision_column] = audit_df[args.decision_column].map(_safe_text)
    reviewed_df = audit_df[audit_df[args.decision_column].ne("")].copy()
    reviewed_df["manual_check_pass"] = reviewed_df[args.decision_column].map(_manual_pass)

    class_results = []
    for label, group in reviewed_df.groupby(args.label_column):
        reviewed = int(len(group))
        passed = int(group["manual_check_pass"].sum())
        pass_rate = float(passed / reviewed) if reviewed > 0 else 0.0
        class_results.append(
            {
                "label": label,
                "reviewed": reviewed,
                "passed": passed,
                "pass_rate": pass_rate,
                "recommendation": _recommend(
                    pass_rate,
                    reviewed,
                    args.include_threshold,
                    args.review_threshold,
                    args.min_samples_per_class,
                ),
            }
        )

    trusted_rows = 0
    trusted_dist = {}
    trusted_path = Path(args.trusted_silver_csv)
    if trusted_path.exists():
        trusted_df = pd.read_csv(trusted_path)
        trusted_rows = int(len(trusted_df))
        if args.label_column in trusted_df.columns:
            trusted_dist = {str(k): int(v) for k, v in trusted_df[args.label_column].value_counts(dropna=False).to_dict().items()}

    overall_reviewed = int(len(reviewed_df))
    overall_passed = int(reviewed_df["manual_check_pass"].sum()) if overall_reviewed > 0 else 0
    overall_pass_rate = float(overall_passed / overall_reviewed) if overall_reviewed > 0 else 0.0

    payload = {
        "audit_rows": int(len(audit_df)),
        "reviewed_rows": overall_reviewed,
        "overall_passed": overall_passed,
        "overall_pass_rate": overall_pass_rate,
        "thresholds": {
            "include_threshold": args.include_threshold,
            "review_threshold": args.review_threshold,
            "min_samples_per_class": args.min_samples_per_class,
        },
        "class_results": class_results,
        "trusted_silver_rows": trusted_rows,
        "trusted_silver_distribution": trusted_dist,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Trusted Silver Evaluation",
        "",
        f"- audit_rows: {payload['audit_rows']}",
        f"- reviewed_rows: {payload['reviewed_rows']}",
        f"- overall_passed: {payload['overall_passed']}",
        f"- overall_pass_rate: {payload['overall_pass_rate']:.4f}",
        f"- trusted_silver_rows: {payload['trusted_silver_rows']}",
        "",
        "## Per-class Recommendation",
        "",
        "| label | reviewed | passed | pass_rate | recommendation |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for item in class_results:
        lines.append(
            f"| {item['label']} | {item['reviewed']} | {item['passed']} | {item['pass_rate']:.4f} | {item['recommendation']} |"
        )
    if not class_results:
        lines.append("| n/a | 0 | 0 | 0.0000 | review_more |")

    with output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
