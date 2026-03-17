from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

from email_safety.data.io import coerce_string_columns, ensure_columns, load_dataframe
from email_safety.data.split import make_train_valid_split
from email_safety.evaluation.metrics import dump_eval_results, evaluate_multiclass
from email_safety.explain.analysis import export_badcases, export_feature_importance
from email_safety.features.structured_features import StructuredFeatureProcessor
from email_safety.models.baseline import BaselineClassifier
from email_safety.preprocessing.text_clean import build_concat_text
from email_safety.utils.logger import get_logger

LOGGER = get_logger(__name__)


def _load_splits(config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    fields_cfg = config["fields"]
    id_col = data_cfg["id_column"]
    label_col = data_cfg["label_column"]
    raw_format = data_cfg.get("raw_format", "auto")

    train_df = load_dataframe(data_cfg["train_path"], raw_format=raw_format, id_column=id_col)
    valid_path = data_cfg.get("valid_path")
    test_path = data_cfg.get("test_path")

    text_fields = fields_cfg["text_fields"]
    required_train_cols = [id_col, label_col] + text_fields
    train_df = ensure_columns(train_df, required_train_cols)

    if label_col not in train_df.columns:
        raise ValueError(f"Train data missing label column: {label_col}")

    if valid_path:
        valid_df = load_dataframe(valid_path, raw_format=raw_format, id_column=id_col)
        valid_df = ensure_columns(valid_df, [id_col, label_col] + text_fields)
    else:
        train_df, valid_df = make_train_valid_split(
            train_df,
            label_column=label_col,
            valid_size=data_cfg.get("valid_size", 0.2),
            random_state=config["project"].get("seed", 42),
            stratify=data_cfg.get("stratify", True),
        )

    if test_path:
        test_df = load_dataframe(test_path, raw_format=raw_format, id_column=id_col)
        test_df = ensure_columns(test_df, [id_col] + text_fields)
    else:
        test_df = pd.DataFrame(columns=[id_col] + text_fields)

    all_string_cols = list(
        set(text_fields + config["fields"].get("structured_fields", []) + [id_col, label_col])
    )
    train_df = coerce_string_columns(train_df, all_string_cols)
    valid_df = coerce_string_columns(valid_df, all_string_cols)
    if not test_df.empty:
        test_df = coerce_string_columns(test_df, all_string_cols)

    return train_df, valid_df, test_df


def run_baseline_training(config: Dict) -> Dict:
    data_cfg = config["data"]
    fields_cfg = config["fields"]
    preprocess_cfg = config["preprocess"]
    train_cfg = config["train"]
    experiment_name = config.get("experiment_name", "baseline")

    output_root = Path(config["project"]["output_dir"])
    metrics_dir = output_root / "metrics" / experiment_name
    model_dir = output_root / "models" / experiment_name
    submission_dir = output_root / "submissions" / experiment_name
    report_dir = output_root / "reports" / experiment_name

    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    submission_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df, valid_df, test_df = _load_splits(config)
    label_col = data_cfg["label_column"]
    text_fields = fields_cfg["text_fields"]
    use_text = config["model"].get("use_text_features", True)

    train_text, valid_text = None, None
    if use_text:
        train_text = build_concat_text(train_df, text_fields=text_fields, **preprocess_cfg)
        valid_text = build_concat_text(valid_df, text_fields=text_fields, **preprocess_cfg)

    use_struct = config["model"].get("use_structured_features", True)
    struct_proc = None
    train_struct, valid_struct = None, None

    if use_struct:
        LOGGER.info("Building structured features...")
        struct_proc = StructuredFeatureProcessor(with_scaler=True)
        train_struct = struct_proc.fit_transform(train_df)
        valid_struct = struct_proc.transform(valid_df)

    clf = BaselineClassifier(config)
    LOGGER.info("Training baseline model: %s", config["model"].get("model_type", "logistic_regression"))
    clf.fit(train_text, train_df[label_col].values, structured_train=train_struct)

    pred_valid = clf.predict(valid_text, structured=valid_struct)
    result = evaluate_multiclass(valid_df[label_col].values, pred_valid)
    dump_eval_results(result, metrics_dir)

    valid_pred_df = valid_df.copy()
    valid_pred_df["pred_label"] = pred_valid
    export_badcases(valid_pred_df, y_true_col=label_col, y_pred_col="pred_label", output_path=str(report_dir / "badcases.csv"))

    if hasattr(clf.model, "coef_") and clf.vectorizer is not None:
        tfidf_names = clf.vectorizer.get_feature_names_out()
        feature_names = list(tfidf_names)
        if use_struct and struct_proc is not None:
            feature_names.extend(struct_proc.columns_)
        coef = np.abs(clf.model.coef_)
        importance = coef.mean(axis=0)
        top_idx = np.argsort(importance)[::-1][:200]
        imp_df = pd.DataFrame(
            {
                "feature": [feature_names[i] for i in top_idx],
                "importance": [float(importance[i]) for i in top_idx],
            }
        )
        export_feature_importance(imp_df, str(report_dir / "feature_importance_top200.csv"))

    model_path = model_dir / train_cfg.get("save_model_name", "baseline_model.joblib")
    joblib.dump({"classifier": clf, "structured_processor": struct_proc, "config": config}, model_path)
    LOGGER.info("Saved model to %s", model_path)

    if not test_df.empty:
        test_text = None
        if use_text:
            test_text = build_concat_text(test_df, text_fields=text_fields, **preprocess_cfg)
        test_struct = struct_proc.transform(test_df) if (use_struct and struct_proc is not None) else None
        pred_test = clf.predict(test_text, structured=test_struct)

        sub_path = submission_dir / train_cfg.get("save_submission_name", "submission.csv")
        pd.DataFrame({data_cfg["id_column"]: test_df[data_cfg["id_column"]], "pred_label": pred_test}).to_csv(
            sub_path, index=False
        )
        LOGGER.info("Saved submission to %s", sub_path)

    LOGGER.info(
        "Valid metrics: acc=%.4f macro_p=%.4f macro_r=%.4f macro_f1=%.4f",
        result["accuracy"],
        result["macro_precision"],
        result["macro_recall"],
        result["macro_f1"],
    )

    return result
