from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

from email_safety.trainers.baseline_trainer import run_baseline_training


def _pick_summary(result: Dict) -> Dict:
    return {
        "accuracy": result.get("accuracy", 0.0),
        "macro_precision": result.get("macro_precision", 0.0),
        "macro_recall": result.get("macro_recall", 0.0),
        "macro_f1": result.get("macro_f1", 0.0),
    }


def run_experiments(config: Dict, baseline_template: Dict):
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    exp_cfg = config.get("experiments", {})

    if exp_cfg.get("baseline_compare", {}).get("enabled", False):
        for model_name in exp_cfg["baseline_compare"].get("models", []):
            cfg = copy.deepcopy(baseline_template)
            cfg["model"]["model_type"] = model_name
            res = run_baseline_training(cfg)
            records.append({"experiment": f"baseline_{model_name}", **_pick_summary(res)})

    if exp_cfg.get("modality_ablation", {}).get("enabled", False):
        for setting in exp_cfg["modality_ablation"].get("settings", []):
            cfg = copy.deepcopy(baseline_template)
            use_text = setting.get("use_text", True)
            use_struct = setting.get("use_structured", True)

            cfg["model"]["use_structured_features"] = bool(use_struct)
            if not use_text:
                cfg["fields"]["text_fields"] = ["_empty_text_col"]
            res = run_baseline_training(cfg)
            records.append({"experiment": f"modality_{setting['name']}", **_pick_summary(res)})

    if exp_cfg.get("text_field_ablation", {}).get("enabled", False):
        for field_set in exp_cfg["text_field_ablation"].get("field_sets", []):
            cfg = copy.deepcopy(baseline_template)
            cfg["fields"]["text_fields"] = field_set
            res = run_baseline_training(cfg)
            records.append({"experiment": f"text_fields_{'+'.join(field_set)}", **_pick_summary(res)})

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    with open(output_dir / "experiment_records.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return str(summary_path)
