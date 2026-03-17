from __future__ import annotations

import pandas as pd


def _get_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field in df.columns:
        return df[field].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def build_rule_risk_flags(df: pd.DataFrame) -> pd.DataFrame:
    """可选模块：规则风险标记（预留可扩展接口）。"""
    out = pd.DataFrame(index=df.index)
    subject = _get_series(df, "subject").str.lower()
    content = _get_series(df, "content").str.lower()

    out["flag_urgent_keywords"] = (
        subject.str.contains("urgent|verify|blocked|account|invoice", regex=True)
        | content.str.contains("urgent|verify|blocked|account|invoice", regex=True)
    ).astype(int)
    out["flag_has_short_url"] = (
        _get_series(df, "url").str.contains(r"bit\.ly|t\.cn|tinyurl", regex=True).astype(int)
    )
    return out
