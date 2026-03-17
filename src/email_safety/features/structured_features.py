from __future__ import annotations

import re
from typing import Dict, Iterable, List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
SUSPICIOUS_SUFFIX = (".top", ".xyz", ".click", ".loan", ".work", ".zip", ".icu")


def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v)


def _split_tokens(text: str) -> List[str]:
    if not text:
        return []
    return [t for t in re.split(r"[;,\s]+", text) if t]


def _extract_urls(raw: str) -> List[str]:
    raw = _safe_str(raw)
    return URL_PATTERN.findall(raw)


def _get_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field in df.columns:
        return df[field].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _get_numeric_series(df: pd.DataFrame, field: str) -> pd.Series:
    if field in df.columns:
        return pd.to_numeric(df[field], errors="coerce").fillna(0)
    return pd.Series([0.0] * len(df), index=df.index, dtype="float64")


def _domain_count(urls: List[str]) -> int:
    domains = set()
    for u in urls:
        p = urlparse(u if u.startswith("http") else f"http://{u}")
        if p.netloc:
            domains.add(p.netloc.lower())
    return len(domains)


def build_structured_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    url_series = _get_series(df, "url")
    html_series = _get_series(df, "htmltag")
    attach_series = _get_series(df, "attach")

    urls = url_series.map(_extract_urls)
    out["url_count"] = urls.map(len)
    out["domain_count"] = urls.map(_domain_count)
    out["suspicious_suffix_count"] = urls.map(
        lambda arr: sum(1 for u in arr if any(u.lower().endswith(s) for s in SUSPICIOUS_SUFFIX))
    )

    out["html_tag_count"] = html_series.map(lambda x: len(_split_tokens(x)))
    out["attach_count"] = attach_series.map(lambda x: len(_split_tokens(x)))

    for field in ["sender", "from", "fromname"]:
        s = _get_series(df, field)
        out[f"{field}_missing"] = (s.str.strip() == "").astype(int)
        out[f"{field}_len"] = s.str.len()

    sender_s = _get_series(df, "sender").str.strip().str.lower()
    from_s = _get_series(df, "from").str.strip().str.lower()
    fromname_s = _get_series(df, "fromname").str.strip().str.lower()
    out["sender_from_same"] = (sender_s == from_s).astype(int)
    out["sender_fromname_same"] = (sender_s == fromname_s).astype(int)

    xmailer_s = _get_series(df, "xmailer")
    out["xmailer_missing"] = (xmailer_s.str.strip() == "").astype(int)

    ip_s = _get_series(df, "ip")
    region_s = _get_series(df, "region")
    out["ip_missing"] = (ip_s.str.strip() == "").astype(int)
    out["region_missing"] = (region_s.str.strip() == "").astype(int)

    out["wlistcnt"] = _get_numeric_series(df, "wlistcnt")
    out["dwlistcnt"] = _get_numeric_series(df, "dwlistcnt")

    rcpt_s = _get_series(df, "rcpt")
    out["rcpt_count"] = rcpt_s.map(lambda x: len(_split_tokens(x)))

    for t in ["subject", "content", "doccontent"]:
        s = _get_series(df, t)
        out[f"{t}_len"] = s.str.len()

    return out


class StructuredFeatureProcessor:
    def __init__(self, with_scaler: bool = True):
        self.with_scaler = with_scaler
        self.scaler = StandardScaler() if with_scaler else None
        self.columns_: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        feats = build_structured_features(df)
        self.columns_ = feats.columns.tolist()
        x = feats.values.astype(float)
        if self.scaler is not None:
            x = self.scaler.fit_transform(x)
        return x

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        feats = build_structured_features(df)
        for c in self.columns_:
            if c not in feats.columns:
                feats[c] = 0.0
        feats = feats[self.columns_]
        x = feats.values.astype(float)
        if self.scaler is not None:
            x = self.scaler.transform(x)
        return x
