from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def export_badcases(df: pd.DataFrame, y_true_col: str, y_pred_col: str, output_path: str):
    bad = df[df[y_true_col] != df[y_pred_col]].copy()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bad.to_csv(output_path, index=False)
    return bad


def export_feature_importance(importance_df: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    return output_path


def llm_assisted_badcase_interface(records: Iterable[dict]):
    """占位接口：后续可接入 LLM badcase 聚类、错误模式归因、改进建议生成。"""
    return {
        "status": "placeholder",
        "message": "LLM-assisted badcase analysis interface is reserved.",
        "num_records": len(list(records)),
    }
