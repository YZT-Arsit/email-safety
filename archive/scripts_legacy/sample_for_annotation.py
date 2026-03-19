#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.io import ensure_columns, load_dataframe


EXPORT_COLUMNS = [
    "id",
    "subject",
    "content",
    "doccontent",
    "from",
    "fromname",
    "url",
    "attach",
    "htmltag",
    "ip",
    "rcpt",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample seed emails for manual annotation")
    parser.add_argument("--input-path", type=str, default="spam_email_data.log")
    parser.add_argument("--output-path", type=str, default="data/annotation/seed_samples.csv")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--mode", choices=["random", "bucket"], default="random")
    parser.add_argument(
        "--bucket-by",
        choices=["subject_len_bin", "has_url", "has_attach", "sender_domain", "region"],
        default="subject_len_bin",
    )
    parser.add_argument("--content-max-len", type=int, default=800)
    parser.add_argument("--doccontent-max-len", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def _sender_domain(sender: str) -> str:
    sender = _safe_str(sender).strip().lower()
    if "@" not in sender:
        return "unknown"
    return sender.split("@", 1)[-1]


def _build_bucket(df: pd.DataFrame, bucket_by: str) -> pd.Series:
    if bucket_by == "subject_len_bin":
        lengths = df["subject"].fillna("").astype(str).str.len()
        bins = pd.cut(lengths, bins=[-1, 10, 30, 80, 200, 10**9], labels=["s0", "s1", "s2", "s3", "s4"])
        return bins.astype(str)
    if bucket_by == "has_url":
        return df["url"].fillna("").astype(str).str.strip().eq("").map({True: "no_url", False: "has_url"})
    if bucket_by == "has_attach":
        return df["attach"].fillna("").astype(str).str.strip().eq("").map({True: "no_attach", False: "has_attach"})
    if bucket_by == "sender_domain":
        return df["sender"].map(_sender_domain)
    if bucket_by == "region":
        return df["region"].fillna("").astype(str).str.strip().replace("", "unknown")
    return pd.Series(["all"] * len(df), index=df.index)


def _stratified_sample(df: pd.DataFrame, n: int, bucket_col: str, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    rng = np.random.default_rng(seed)

    group_sizes = df.groupby(bucket_col).size().sort_values(ascending=False)
    total = int(group_sizes.sum())

    allocated = {}
    for bucket, size in group_sizes.items():
        k = int(round(n * (size / total)))
        if size > 0:
            k = max(1, min(k, int(size)))
        allocated[bucket] = k

    current = sum(allocated.values())
    buckets = list(group_sizes.index)
    while current > n:
        b = buckets[current % len(buckets)]
        if allocated[b] > 1:
            allocated[b] -= 1
            current -= 1
        else:
            buckets = [x for x in buckets if allocated[x] > 1]
            if not buckets:
                break
    while current < n:
        b = buckets[current % len(buckets)]
        if allocated[b] < int(group_sizes.loc[b]):
            allocated[b] += 1
            current += 1
        else:
            buckets = [x for x in buckets if allocated[x] < int(group_sizes.loc[x])]
            if not buckets:
                break

    sampled = []
    for b, k in allocated.items():
        group = df[df[bucket_col] == b]
        if k > 0:
            idx = rng.choice(group.index.to_numpy(), size=min(k, len(group)), replace=False)
            sampled.append(group.loc[idx])

    out = pd.concat(sampled, axis=0) if sampled else df.sample(n=n, random_state=seed)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.input_path, raw_format="log", id_column="id")
    df = ensure_columns(df, EXPORT_COLUMNS + ["sender", "region"])

    for c in EXPORT_COLUMNS:
        df[c] = df[c].map(_safe_str)

    if args.mode == "random":
        sampled = df.sample(n=min(args.sample_size, len(df)), random_state=args.seed).reset_index(drop=True)
    else:
        df["__bucket"] = _build_bucket(df, args.bucket_by)
        sampled = _stratified_sample(df, n=args.sample_size, bucket_col="__bucket", seed=args.seed)

    sampled["content"] = sampled["content"].str.slice(0, args.content_max_len)
    sampled["doccontent"] = sampled["doccontent"].str.slice(0, args.doccontent_max_len)

    out = sampled[EXPORT_COLUMNS].copy()
    out["manual_label"] = ""
    out["weak_label"] = ""
    out["notes"] = ""

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Saved {len(out)} rows to {output_path}")
    if args.mode == "bucket" and "__bucket" in sampled.columns:
        print("Bucket distribution:")
        print(sampled["__bucket"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
