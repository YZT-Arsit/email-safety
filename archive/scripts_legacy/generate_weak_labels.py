#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

DEFAULT_RULES = {
    "labels": {
        "advertisement": {
            "min_score": 2,
            "patterns": [
                {"field": "subject", "regex": r"(发票|开票|推广|营销|优惠|广告)", "weight": 2},
                {"field": "content", "regex": r"(联系方式|微信|免费|商务合作)", "weight": 1},
                {"field": "url", "regex": r"(http|www\.)", "weight": 1},
            ],
        },
        "phishing": {
            "min_score": 2,
            "patterns": [
                {"field": "subject", "regex": r"(账号|验证|异常登录|blocked|quarantine)", "weight": 2},
                {"field": "content", "regex": r"(点击|立即验证|重新登录|账号异常)", "weight": 2},
            ],
        },
        "impersonation": {
            "min_score": 2,
            "patterns": [
                {"field": "fromname", "regex": r"(财务|经理|ceo|hr|采购)", "weight": 2},
                {"field": "subject", "regex": r"(紧急|转账|汇款|付款)", "weight": 2},
            ],
        },
        "malicious_link_or_attachment": {
            "min_score": 2,
            "patterns": [
                {"field": "url", "regex": r"(bit\.ly|tinyurl|\.top|\.xyz|\.click|\.icu|\.zip)", "weight": 2},
                {"field": "attach", "regex": r"(exe|js|vbs|scr|docm|xlsm|zip|rar)", "weight": 2},
            ],
        },
        "black_industry_or_policy_violation": {
            "min_score": 2,
            "patterns": [
                {"field": "subject", "regex": r"(办证|博彩|贷款秒批|刷单|灰产)", "weight": 2},
                {"field": "content", "regex": r"(暗网|黑客|绕过审核|批量账号)", "weight": 2},
            ],
        },
    },
    "fallback_label": "uncertain",
    "label_priority": [
        "phishing",
        "impersonation",
        "malicious_link_or_attachment",
        "black_industry_or_policy_violation",
        "advertisement",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate weak labels by rule patterns")
    parser.add_argument("--input-csv", type=str, default="data/annotation/seed_samples.csv")
    parser.add_argument("--output-csv", type=str, default="data/annotation/seed_samples_weak_labeled.csv")
    parser.add_argument("--rules-config", type=str, default="configs/weak_label_rules.yaml")
    parser.add_argument("--keep-existing-weak-label", action="store_true")
    return parser.parse_args()


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def _load_rules(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return DEFAULT_RULES
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _compile_rules(rules: Dict[str, Any]) -> Dict[str, List[Tuple[str, re.Pattern, int]]]:
    compiled: Dict[str, List[Tuple[str, re.Pattern, int]]] = {}
    for label, cfg in rules.get("labels", {}).items():
        triples = []
        for pat in cfg.get("patterns", []):
            field = pat.get("field", "content")
            regex = re.compile(pat.get("regex", r"$^"), flags=re.IGNORECASE)
            weight = int(pat.get("weight", 1))
            triples.append((field, regex, weight))
        compiled[label] = triples
    return compiled


def _predict_label(row: pd.Series, rules: Dict[str, Any], compiled_rules: Dict[str, List[Tuple[str, re.Pattern, int]]]) -> Tuple[str, Dict[str, int]]:
    scores: Dict[str, int] = {label: 0 for label in rules.get("labels", {}).keys()}

    for label, patterns in compiled_rules.items():
        for field, regex, weight in patterns:
            text = _safe_text(row.get(field, ""))
            if regex.search(text):
                scores[label] += weight

    candidates = []
    for label, score in scores.items():
        min_score = int(rules.get("labels", {}).get(label, {}).get("min_score", 1))
        if score >= min_score:
            candidates.append((label, score))

    if not candidates:
        return rules.get("fallback_label", "uncertain"), scores

    priority = rules.get("label_priority", list(scores.keys()))
    candidates.sort(key=lambda x: (x[1], -priority.index(x[0]) if x[0] in priority else -10**6), reverse=True)
    return candidates[0][0], scores


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    rules = _load_rules(args.rules_config)
    compiled = _compile_rules(rules)

    if "manual_label" not in df.columns:
        df["manual_label"] = ""
    if "weak_label" not in df.columns:
        df["weak_label"] = ""

    weak_labels = []
    weak_scores = []

    for _, row in df.iterrows():
        if args.keep_existing_weak_label and str(row.get("weak_label", "")).strip():
            weak_labels.append(str(row.get("weak_label", "")).strip())
            weak_scores.append("{}")
            continue

        pred, scores = _predict_label(row, rules, compiled)
        weak_labels.append(pred)
        weak_scores.append(json.dumps(scores, ensure_ascii=False))

    df["weak_label"] = weak_labels
    df["weak_label_scores"] = weak_scores

    # 明确不覆盖人工标签
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved weak-labeled file to {output_path}")
    print("Weak label distribution:")
    print(df["weak_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
