#!/usr/bin/env python
"""Lightweight repository layout smoke check for the public GitHub version."""
from __future__ import annotations

from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "RUNBOOK.md",
    "PROJECT_STRUCTURE.md",
    ".gitignore",
    "requirements.txt",
    "docs/label_schema.md",
    "docs/project_summary.md",
    "docs/interview_notes.md",
    "scripts/download_model_from_modelscope.py",
    "scripts/build_mlm_corpus.py",
    "scripts/train_dapt_mlm.py",
    "scripts/run_multilingual_dapt_comparison.py",
    "scripts/predict_all_unlabeled.py",
    "scripts/predict_text_transformer.py",
    "scripts/build_consensus_silver.py",
    "scripts/build_semi_supervised_dataset.py",
    "scripts/run_semi_supervised_comparison.py",
    "scripts/summarize_final_closed_loop.py",
    "outputs/final_summary/final_closed_loop_results.csv",
    "outputs/final_summary/final_closed_loop_summary.md",
    "outputs/final_summary/final_interview_bullets.md",
]


def main() -> None:
    missing = [path for path in REQUIRED_FILES if not Path(path).exists()]
    if missing:
        print("Missing files:")
        for path in missing:
            print(f"- {path}")
        raise SystemExit(1)
    print("Repository smoke check passed.")


if __name__ == "__main__":
    main()
