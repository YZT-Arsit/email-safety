#!/usr/bin/env python
"""Build a cleaned MLM corpus from unlabeled mail logs.
Inputs: spam mail log file. Outputs: mail_corpus.txt and corpus statistics JSON."""
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

from email_safety.data.io import ensure_columns, load_dataframe
from email_safety.preprocessing.text_clean import build_concat_text


def parse_args():
    parser = argparse.ArgumentParser(description="Build MLM corpus from unlabeled mail logs")
    parser.add_argument("--input-path", type=str, default="spam_email_data.log")
    parser.add_argument("--raw-format", type=str, default="log")
    parser.add_argument("--output-txt", "--output-path", dest="output_txt", type=str, default="data/mlm_corpus/mail_corpus.txt")
    parser.add_argument("--stats-json", type=str, default="data/mlm_corpus/mail_corpus_stats.json")
    parser.add_argument("--min-length", type=int, default=20)
    return parser.parse_args()


def _looks_garbled(text: str) -> bool:
    if not text:
        return True
    bad_chars = sum(1 for ch in text if ord(ch) == 65533 or ch in {"�", "\x00"})
    ratio = bad_chars / max(1, len(text))
    return ratio > 0.05


def main():
    args = parse_args()
    df = load_dataframe(args.input_path, raw_format=args.raw_format, id_column="id")
    df = ensure_columns(df, ["subject", "content", "doccontent"])
    raw_rows = len(df)

    texts = build_concat_text(
        df,
        text_fields=["subject", "content", "doccontent"],
        lowercase=False,
        remove_urls=False,
        max_text_length=5000,
    )
    texts = texts.map(lambda x: " ".join(str(x).split()))
    texts = texts[texts.str.len() >= args.min_length]
    texts = texts[~texts.map(_looks_garbled)]

    output_path = Path(args.output_txt)
    stats_path = Path(args.stats_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(texts.tolist()), encoding="utf-8")

    lengths = texts.str.len()
    stats = {
        "raw_rows": int(raw_rows),
        "retained_rows": int(len(texts)),
        "retention_rate": float(len(texts) / max(1, raw_rows)),
        "avg_length": float(lengths.mean()) if len(lengths) > 0 else 0.0,
        "length_summary": {
            "min": int(lengths.min()) if len(lengths) > 0 else 0,
            "p25": float(lengths.quantile(0.25)) if len(lengths) > 0 else 0.0,
            "p50": float(lengths.quantile(0.50)) if len(lengths) > 0 else 0.0,
            "p75": float(lengths.quantile(0.75)) if len(lengths) > 0 else 0.0,
            "max": int(lengths.max()) if len(lengths) > 0 else 0,
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
