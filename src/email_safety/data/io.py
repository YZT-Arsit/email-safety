from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def _safe_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def read_log_with_json(path: str | Path, id_column: str = "id") -> pd.DataFrame:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" not in line:
                continue
            raw_id, raw_json = line.split("\t", 1)
            try:
                payload = json.loads(raw_json)
            except json.JSONDecodeError:
                continue
            payload[id_column] = raw_id
            rows.append(payload)
    return pd.DataFrame(rows)


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    return "log"


def load_dataframe(path: str | Path, raw_format: str = "auto", id_column: str = "id") -> pd.DataFrame:
    path = Path(path)
    fmt = _detect_format(path) if raw_format == "auto" else raw_format
    if fmt == "log":
        return read_log_with_json(path, id_column=id_column)
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported format: {fmt}")


def ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = ""
    return df


def coerce_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in columns:
        if c in df.columns:
            df[c] = df[c].map(_safe_str)
    return df
