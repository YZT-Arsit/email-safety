#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd


DEFAULT_ALLOWED_LABELS = (
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
)

DEFAULT_LABEL_ALIASES: Dict[str, str] = {
    "advertisemen": "advertisement",
    "ad": "advertisement",
    "promo": "advertisement",
    "phish": "phishing",
    "impersonate": "impersonation",
    "malicious_attachment": "malicious_link_or_attachment",
    "malicious_link": "malicious_link_or_attachment",
    "black_industry_or_policy_violatio": "black_industry_or_policy_violation",
    "blackindustry_or_policy_violation": "black_industry_or_policy_violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean labeled dataset from annotated csv")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="data/annotation/clean_labeled_dataset.csv")
    parser.add_argument("--stats-csv", type=str, default="data/annotation/label_distribution.csv")
    parser.add_argument("--issues-csv", type=str, default="data/annotation/label_issues.csv")
    parser.add_argument("--summary-json", type=str, default="data/annotation/label_build_summary.json")
    parser.add_argument(
        "--allowed-labels",
        type=str,
        default=",".join(DEFAULT_ALLOWED_LABELS),
        help="Comma-separated allowed labels",
    )
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero if invalid rows exist")
    return parser.parse_args()


def _safe_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def _normalize_label(v: str, alias_map: Dict[str, str]) -> str:
    text = _safe_text(v).lower().replace("\ufeff", "")
    text = " ".join(text.split())
    text = text.replace("-", "_").replace("/", "_")
    return alias_map.get(text, text)


def main() -> None:
    args = parse_args()
    allowed = {_safe_text(x).lower() for x in args.allowed_labels.split(",") if _safe_text(x)}
    alias_map = dict(DEFAULT_LABEL_ALIASES)

    df = pd.read_csv(args.input_csv)
    if "manual_label" not in df.columns:
        raise ValueError("Input csv missing required column: manual_label")
    if "id" not in df.columns:
        raise ValueError("Input csv missing required column: id")

    df["id"] = df["id"].map(_safe_text)
    df["manual_label_raw"] = df["manual_label"]
    df["manual_label"] = df["manual_label"].map(lambda x: _normalize_label(x, alias_map))

    df["id_missing"] = df["id"].eq("")
    df["id_duplicated"] = df["id"].duplicated(keep=False) & ~df["id_missing"]
    df["manual_label_missing"] = df["manual_label"].eq("")
    df["manual_label_invalid"] = ~df["manual_label"].isin(allowed) & ~df["manual_label_missing"]

    issue_reason_cols = [
        ("id_missing", "id_missing"),
        ("id_duplicated", "id_duplicated"),
        ("manual_label_missing", "manual_label_missing"),
        ("manual_label_invalid", "manual_label_invalid"),
    ]
    df["issue_reason"] = df.apply(
        lambda row: "|".join(name for col, name in issue_reason_cols if bool(row[col])),
        axis=1,
    )

    issues = df[df["issue_reason"].ne("")].copy()
    clean = df[df["issue_reason"].eq("")].copy()

    out_path = Path(args.output_csv)
    stats_path = Path(args.stats_csv)
    issues_path = Path(args.issues_csv)
    summary_path = Path(args.summary_json)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    issues_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    clean.to_csv(out_path, index=False)

    dist = clean["manual_label"].value_counts(dropna=False).rename_axis("label").reset_index(name="count")
    dist["ratio"] = dist["count"] / max(1, len(clean))
    dist.to_csv(stats_path, index=False)

    issues.to_csv(issues_path, index=False)

    summary = {
        "total_rows": int(len(df)),
        "clean_rows": int(len(clean)),
        "issue_rows": int(len(issues)),
        "id_missing": int(df["id_missing"].sum()),
        "id_duplicated": int(df["id_duplicated"].sum()),
        "manual_label_missing": int(df["manual_label_missing"].sum()),
        "manual_label_invalid": int(df["manual_label_invalid"].sum()),
        "allowed_labels": sorted(allowed),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Total rows: {len(df)}")
    print(f"Clean rows: {len(clean)}")
    print(f"Issue rows: {len(issues)}")
    print(f"Saved clean dataset to {out_path}")
    print(f"Saved label distribution to {stats_path}")
    print(f"Saved label issues to {issues_path}")
    print(f"Saved build summary to {summary_path}")

    if len(issues) > 0:
        print("Issue breakdown:")
        print(f"- id missing: {int(df['id_missing'].sum())}")
        print(f"- id duplicated: {int(df['id_duplicated'].sum())}")
        print(f"- manual_label missing: {int(df['manual_label_missing'].sum())}")
        print(f"- manual_label invalid: {int(df['manual_label_invalid'].sum())}")
        print(f"Allowed labels: {sorted(allowed)}")
        if args.strict:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
