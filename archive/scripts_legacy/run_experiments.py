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

from email_safety.experiments.runner import run_experiments
from email_safety.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment suite")
    parser.add_argument("--config", type=str, required=True, help="Experiment config")
    parser.add_argument("--baseline-config", type=str, default="configs/baseline.yaml", help="Baseline template config")
    return parser.parse_args()


def main():
    args = parse_args()
    exp_cfg = load_config(args.config)
    baseline_cfg = load_config(args.baseline_config)
    run_experiments(exp_cfg, baseline_cfg)


if __name__ == "__main__":
    main()
