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

from email_safety.trainers.fusion_trainer import run_fusion_training
from email_safety.utils.config import load_config
from email_safety.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train fusion model")
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    run_fusion_training(cfg)


if __name__ == "__main__":
    main()
