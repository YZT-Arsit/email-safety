from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

WS_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")


def normalize_text(text: str, lowercase: bool = True, remove_urls: bool = False) -> str:
    if text is None:
        return ""
    text = str(text)
    if remove_urls:
        text = URL_RE.sub(" ", text)
    text = WS_RE.sub(" ", text).strip()
    if lowercase:
        text = text.lower()
    return text


def build_concat_text(
    df: pd.DataFrame,
    text_fields: Iterable[str],
    lowercase: bool = True,
    remove_urls: bool = False,
    max_text_length: int = 5000,
) -> pd.Series:
    fields = list(text_fields)
    if not fields:
        return pd.Series([""] * len(df), index=df.index)

    parts = []
    for field in fields:
        if field in df.columns:
            parts.append(df[field].fillna("").astype(str))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))

    merged = pd.Series([" ".join(items) for items in zip(*parts)], index=df.index)
    merged = merged.map(lambda x: normalize_text(x, lowercase=lowercase, remove_urls=remove_urls))
    if max_text_length and max_text_length > 0:
        merged = merged.str.slice(0, max_text_length)
    return merged
