#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.io import load_dataframe
from email_safety.inference.predict import predict_with_saved_baseline
from email_safety.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Predict test labels with saved baseline model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="outputs/submissions/submission.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    df = load_dataframe(
        args.input_path,
        raw_format=cfg["data"].get("raw_format", "auto"),
        id_column=cfg["data"]["id_column"],
    )
    predict_with_saved_baseline(
        model_path=args.model_path,
        df=df,
        text_fields=cfg["fields"]["text_fields"],
        preprocess_cfg=cfg["preprocess"],
        use_structured_features=cfg["model"].get("use_structured_features", True),
        id_column=cfg["data"]["id_column"],
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
