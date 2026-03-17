from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from email_safety.features.risk_flags import build_rule_risk_flags
from email_safety.features.structured_features import StructuredFeatureProcessor, build_structured_features
from email_safety.preprocessing.text_clean import build_concat_text
from email_safety.rules.weak_label import apply_weak_label_rules, load_weak_label_rules, summarize_rule_hits


def predict_with_saved_baseline(
    model_path: str,
    df: pd.DataFrame,
    text_fields,
    preprocess_cfg,
    use_structured_features: bool,
    id_column: str,
    output_csv: str,
):
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    struct_proc = bundle.get("structured_processor", None)

    texts = build_concat_text(
        df,
        text_fields=text_fields,
        lowercase=preprocess_cfg.get("lowercase", True),
        remove_urls=preprocess_cfg.get("remove_urls", False),
        max_text_length=preprocess_cfg.get("max_text_length", 5000),
    )

    structured = struct_proc.transform(df) if (use_structured_features and struct_proc is not None) else None
    pred = clf.predict(texts, structured)

    out = pd.DataFrame({id_column: df[id_column], "pred_label": pred})
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out


def predict_unlabeled_with_metadata(
    model_path: str,
    df: pd.DataFrame,
    text_fields,
    preprocess_cfg,
    use_structured_features: bool,
    id_column: str,
    output_csv: str,
    rules_config_path: str | None = None,
    subject_max_len: int = 120,
    content_max_len: int = 200,
):
    bundle = joblib.load(model_path)
    clf = bundle["classifier"]
    struct_proc = bundle.get("structured_processor", None)
    model_cfg = bundle.get("config", {}).get("model", {})
    use_text_features = model_cfg.get("use_text_features", True)

    texts = None
    if use_text_features:
        texts = build_concat_text(
            df,
            text_fields=text_fields,
            lowercase=preprocess_cfg.get("lowercase", True),
            remove_urls=preprocess_cfg.get("remove_urls", False),
            max_text_length=preprocess_cfg.get("max_text_length", 5000),
        )

    structured = struct_proc.transform(df) if (use_structured_features and struct_proc is not None) else None
    pred = clf.predict(texts, structured)
    proba = clf.predict_proba(texts, structured)

    classes = list(getattr(clf.model, "classes_", []))
    top1_scores = np.ones(len(df), dtype=float)
    top2_scores = np.zeros(len(df), dtype=float)
    top2_labels = [""] * len(df)
    uncertainty = np.zeros(len(df), dtype=float)

    if proba is not None and len(classes) > 0:
        order = np.argsort(-proba, axis=1)
        top1_idx = order[:, 0]
        top2_idx = order[:, 1] if proba.shape[1] > 1 else order[:, 0]
        top1_scores = proba[np.arange(len(df)), top1_idx]
        top2_scores = proba[np.arange(len(df)), top2_idx]
        top2_labels = [str(classes[idx]) for idx in top2_idx]
        uncertainty = 1.0 - top1_scores

    rules = load_weak_label_rules(rules_config_path)
    weak_df = apply_weak_label_rules(df, rules)
    risk_df = build_rule_risk_flags(df)
    struct_summary = build_structured_features(df)

    rule_hits = []
    for idx in df.index:
        risk_flags = risk_df.loc[idx].to_dict()
        rule_hits.append(
            summarize_rule_hits(
                weak_label=str(weak_df.loc[idx, "weak_label"]),
                weak_rule_matches=str(weak_df.loc[idx, "weak_rule_matches"]),
                risk_flags=risk_flags,
            )
        )

    out = pd.DataFrame(
        {
            id_column: df[id_column].astype(str),
            "pred_label": pred,
            "pred_score": top1_scores,
            "top2_label": top2_labels,
            "top2_score": top2_scores,
            "uncertainty": uncertainty,
            "weak_label": weak_df["weak_label"],
            "weak_label_scores": weak_df["weak_label_scores"],
            "rule_hits": rule_hits,
            "subject_summary": df.get("subject", pd.Series([""] * len(df))).fillna("").astype(str).str.slice(0, subject_max_len),
            "content_summary": df.get("content", pd.Series([""] * len(df))).fillna("").astype(str).str.slice(0, content_max_len),
            "url_count": struct_summary["url_count"],
            "attach_count": struct_summary["attach_count"],
            "suspicious_tld_count": struct_summary["suspicious_suffix_count"],
            "html_tag_count": struct_summary["html_tag_count"],
            "rcpt_count": struct_summary["rcpt_count"],
        }
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    return out
