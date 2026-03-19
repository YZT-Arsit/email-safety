#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("/Users/Hoshino/Documents/emailsafety")
CSV_PATH = PROJECT_ROOT / "data/annotation/gold/gold_v2_llm_guided.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs/llm_labeling/gold_v2_llm_summary.json"
OUT_JSON = PROJECT_ROOT / "outputs/llm_guided/llm_guided_data_quality.json"

LABELS = [
    "advertisement",
    "phishing",
    "impersonation",
    "malicious_link_or_attachment",
    "black_industry_or_policy_violation",
]


def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_probs(x):
    if pd.isna(x):
        return {}
    s = str(x).strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    for col in [
        "manual_label",
        "reasoning",
        "reasoning_summary",
        "class_probs",
        "confidence",
        "ambiguous",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["manual_label"] = df["manual_label"].map(safe_text)
    df["reasoning"] = df["reasoning"].map(safe_text)
    df["reasoning_summary"] = df["reasoning_summary"].map(safe_text)
    df["class_probs"] = df["class_probs"].map(parse_probs)

    total_rows = len(df)
    non_empty_reasoning = int(df["reasoning"].ne("").sum())
    non_empty_reasoning_summary = int(df["reasoning_summary"].ne("").sum())
    non_empty_probs = int(df["class_probs"].map(bool).sum())

    df["llm_label"] = df["class_probs"].map(
        lambda d: max(d.items(), key=lambda kv: kv[1])[0] if d else ""
    )
    df["llm_max_prob"] = df["class_probs"].map(
        lambda d: max(d.values()) if d else 0.0
    )

    valid_prob_rows = 0
    invalid_prob_rows = 0
    strong_conf_rows = 0
    agree_rows = 0
    disagree_rows = 0

    for _, row in df.iterrows():
        probs = row["class_probs"]
        if not probs:
            continue

        total_prob = sum(float(v) for v in probs.values())
        has_all_labels = all(label in probs for label in LABELS)
        if abs(total_prob - 1.0) < 0.05 and has_all_labels:
            valid_prob_rows += 1
        else:
            invalid_prob_rows += 1

        if row["llm_max_prob"] >= 0.85:
            strong_conf_rows += 1

        manual = row["manual_label"]
        llm = row["llm_label"]
        if manual and llm:
            if manual == llm:
                agree_rows += 1
            else:
                disagree_rows += 1

    result = {
        "total_rows": total_rows,
        "manual_label_distribution": df["manual_label"].value_counts(dropna=False).to_dict(),
        "llm_signal_coverage": {
            "non_empty_reasoning_rows": non_empty_reasoning,
            "non_empty_reasoning_ratio": round(non_empty_reasoning / max(total_rows, 1), 4),
            "non_empty_reasoning_summary_rows": non_empty_reasoning_summary,
            "non_empty_reasoning_summary_ratio": round(non_empty_reasoning_summary / max(total_rows, 1), 4),
            "non_empty_class_probs_rows": non_empty_probs,
            "non_empty_class_probs_ratio": round(non_empty_probs / max(total_rows, 1), 4),
        },
        "llm_prob_quality": {
            "valid_prob_rows": valid_prob_rows,
            "invalid_prob_rows": invalid_prob_rows,
            "high_confidence_rows_ge_0_85": strong_conf_rows,
        },
        "manual_vs_llm": {
            "agree_rows": agree_rows,
            "disagree_rows": disagree_rows,
            "agreement_ratio_over_rows_with_llm": round(
                agree_rows / max(agree_rows + disagree_rows, 1), 4
            ),
        },
        "llm_pred_distribution": df["llm_label"].value_counts(dropna=False).to_dict(),
    }

    if SUMMARY_PATH.exists():
        try:
            result["llm_generation_summary"] = json.loads(
                SUMMARY_PATH.read_text(encoding="utf-8")
            )
        except Exception:
            result["llm_generation_summary"] = {
                "warning": "failed_to_parse_summary_json"
            }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved to: {OUT_JSON}")


if __name__ == "__main__":
    main()
