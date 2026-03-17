from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None


def evaluate_multiclass(y_true, y_pred) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def dump_eval_results(result: Dict, output_dir: str | Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
    }
    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report_df = pd.DataFrame(result["classification_report"]).transpose()
    report_df.to_csv(output_dir / "classification_report.csv", index=True)

    cm_df = pd.DataFrame(result["confusion_matrix"])
    cm_df.to_csv(output_dir / "confusion_matrix.csv", index=False)

    plt.figure(figsize=(7, 6))
    if sns is not None:
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    else:
        plt.imshow(cm_df.values, cmap="Blues")
        for i in range(cm_df.shape[0]):
            for j in range(cm_df.shape[1]):
                plt.text(j, i, str(cm_df.iloc[i, j]), ha="center", va="center")
        plt.xticks(range(cm_df.shape[1]), cm_df.columns)
        plt.yticks(range(cm_df.shape[0]), cm_df.index)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
