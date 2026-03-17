#!/usr/bin/env python
"""Aggregate final data, model, and semi-supervised results into GitHub- and interview-friendly summaries.
Inputs: key stats/results files. Outputs: final summary CSV and markdown notes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize final closed-loop results for interview materials")
    parser.add_argument("--gold-v1-csv", type=str, default="data/annotation/clean_labeled_dataset.csv")
    parser.add_argument("--gold-v2-stats-json", type=str, default="data/annotation/gold/gold_v2_stats.json")
    parser.add_argument("--mlm-stats-json", type=str, default="data/mlm_corpus/mail_corpus_stats.json")
    parser.add_argument("--download-summary-json", type=str, default="outputs/modelscope_download/download_summary.json")
    parser.add_argument("--baseline-summary-csv", type=str, default="outputs/formal_baselines/results_summary.csv")
    parser.add_argument("--dapt-summary-csv", type=str, default="outputs/multilingual_dapt_comparison/results_summary.csv")
    parser.add_argument("--consensus-silver-stats-json", type=str, default="data/annotation/silver/consensus_trusted_silver_stats.json")
    parser.add_argument("--semi-summary-csv", type=str, default="outputs/semi_supervised_comparison/results_summary.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/final_summary")
    return parser.parse_args()


def _load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _best_row(df: pd.DataFrame, metric_col: str):
    if df.empty or metric_col not in df.columns:
        return None
    return df.sort_values(metric_col, ascending=False).iloc[0].to_dict()


def _format_metric(row, metric_key: str) -> str:
    if not row:
        return "0.0000"
    try:
        return f"{float(row.get(metric_key, 0.0)):.4f}"
    except Exception:
        return "0.0000"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gold_v1_df = _load_csv(args.gold_v1_csv)
    gold_v2_stats = _load_json(args.gold_v2_stats_json)
    mlm_stats = _load_json(args.mlm_stats_json)
    download_summary = _load_json(args.download_summary_json)
    baseline_df = _load_csv(args.baseline_summary_csv)
    dapt_df = _load_csv(args.dapt_summary_csv)
    consensus_stats = _load_json(args.consensus_silver_stats_json)
    semi_df = _load_csv(args.semi_summary_csv)

    rows = []
    if not baseline_df.empty:
        tmp = baseline_df.copy()
        tmp.insert(0, "stage", "baseline")
        rows.extend(tmp.to_dict(orient="records"))
    if not dapt_df.empty:
        tmp = dapt_df.copy()
        tmp.insert(0, "stage", "dapt_downstream")
        rows.extend(tmp.to_dict(orient="records"))
    if not semi_df.empty:
        tmp = semi_df.copy()
        tmp.insert(0, "stage", "semi_supervised")
        rows.extend(tmp.to_dict(orient="records"))
    if not gold_v1_df.empty:
        label_col = "manual_label" if "manual_label" in gold_v1_df.columns else "label"
        rows.append({
            "stage": "gold_v1",
            "size": int(len(gold_v1_df)),
            "notes": f"labels={gold_v1_df[label_col].value_counts().to_dict()}",
        })
    if gold_v2_stats:
        rows.append({
            "stage": "gold_v2",
            "size": int(gold_v2_stats.get("gold_v2_rows", 0)),
            "notes": f"increment={gold_v2_stats.get('new_ids_added', 0)} dist={gold_v2_stats.get('gold_v2_distribution', {})}",
        })
    if mlm_stats:
        rows.append({
            "stage": "mlm_corpus",
            "size": int(mlm_stats.get("retained_rows", 0)),
            "notes": f"avg_len={mlm_stats.get('avg_length', 0)} keep_rate={mlm_stats.get('retention_rate', 0)}",
        })
    if consensus_stats:
        rows.append({
            "stage": "consensus_trusted_silver",
            "size": int(consensus_stats.get("trusted_silver_rows", 0)),
            "notes": f"dist={consensus_stats.get('trusted_label_distribution', {})}",
        })

    best_baseline = _best_row(baseline_df, "macro_f1")
    best_dapt = _best_row(dapt_df, "test_macro_f1")
    best_semi = _best_row(semi_df, "test_macro_f1")

    results_path = output_dir / "final_closed_loop_results.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    summary_lines = [
        "# Final Closed Loop Summary",
        "",
        "## Data Scale",
        f"- Gold v1: {len(gold_v1_df) if not gold_v1_df.empty else 0}",
        f"- Gold v2: {gold_v2_stats.get('gold_v2_rows', 0)}",
        f"- MLM corpus retained rows: {mlm_stats.get('retained_rows', 0)}",
        f"- Consensus trusted silver: {consensus_stats.get('trusted_silver_rows', 0)}",
        "",
        "## Model Download / DAPT",
        f"- ModelScope model id: {download_summary.get('model_id', '')}",
        f"- Local model dir: {download_summary.get('resolved_model_dir', '')}",
        f"- DAPT corpus file: data/mlm_corpus/mail_corpus.txt",
        "",
        "## Best Results",
        f"- Best baseline: {best_baseline.get('experiment', 'n/a') if best_baseline else 'n/a'} / macro_f1={_format_metric(best_baseline, 'macro_f1')}",
        f"- Best multilingual BERT: {best_dapt.get('experiment', 'n/a') if best_dapt else 'n/a'} / test_macro_f1={_format_metric(best_dapt, 'test_macro_f1')}",
        f"- Best semi-supervised: {best_semi.get('experiment', 'n/a') if best_semi else 'n/a'} / test_macro_f1={_format_metric(best_semi, 'test_macro_f1')}",
        "",
        "## Closed Loop",
        "- 2.4w unlabeled mails contributed in two ways: domain-adaptive MLM pretraining and high-precision consensus trusted silver.",
        "- Gold stayed the only source for valid/test; silver was used only in train and with lower sample weight.",
        "",
        "## Limitations",
        "- Long-tail classes remain harder than the head class black_industry_or_policy_violation.",
        "- Border cases between phishing / impersonation and advertisement / black_industry_or_policy_violation remain noisy.",
        "- Trusted silver prioritizes precision, so recall and coverage are intentionally conservative.",
    ]
    (output_dir / "final_closed_loop_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    interview_lines = [
        "# Interview Bullets",
        "",
        "- Built an end-to-end mail risk classification pipeline from 24k unlabeled logs to production-style supervised + semi-supervised training assets.",
        "- Replaced online model dependency with local ModelScope checkpoint download and local-path-only training/inference.",
        "- Used domain-adaptive masked language modeling on the full unlabeled corpus before downstream 5-way classification.",
        "- Designed a high-precision consensus silver strategy across linear models, tree models, transformer teachers, and rule signals.",
        "- Kept evaluation clean by using Gold-only valid/test and training-only trusted silver with lower sample weights.",
        "- Produced reproducible artifacts for data quality, DAPT, downstream comparison, trusted silver selection, semi-supervised comparison, and final reporting.",
    ]
    (output_dir / "final_interview_bullets.md").write_text("\n".join(interview_lines) + "\n", encoding="utf-8")

    print(f"Saved summary files to {output_dir}")


if __name__ == "__main__":
    main()
