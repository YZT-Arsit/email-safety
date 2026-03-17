from __future__ import annotations

import json
import re
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


def _safe_text(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v)


def load_weak_label_rules(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        return DEFAULT_RULES
    rule_path = Path(path)
    if not rule_path.exists():
        return DEFAULT_RULES
    with rule_path.open("r", encoding="utf-8") as f:
        if rule_path.suffix.lower() == ".json":
            return json.load(f)
        return yaml.safe_load(f)


def _compile_rules(rules: Dict[str, Any]) -> Dict[str, List[Tuple[str, re.Pattern, int]]]:
    compiled: Dict[str, List[Tuple[str, re.Pattern, int]]] = {}
    for label, cfg in rules.get("labels", {}).items():
        items: List[Tuple[str, re.Pattern, int]] = []
        for pat in cfg.get("patterns", []):
            items.append(
                (
                    pat.get("field", "content"),
                    re.compile(pat.get("regex", r"$^"), flags=re.IGNORECASE),
                    int(pat.get("weight", 1)),
                )
            )
        compiled[label] = items
    return compiled


def _predict_one(
    row: pd.Series,
    rules: Dict[str, Any],
    compiled_rules: Dict[str, List[Tuple[str, re.Pattern, int]]],
) -> Tuple[str, Dict[str, int], Dict[str, List[str]]]:
    scores: Dict[str, int] = {label: 0 for label in rules.get("labels", {})}
    matched_patterns: Dict[str, List[str]] = {label: [] for label in rules.get("labels", {})}

    for label, patterns in compiled_rules.items():
        for field, regex, weight in patterns:
            text = _safe_text(row.get(field, ""))
            if regex.search(text):
                scores[label] += weight
                matched_patterns[label].append(field)

    candidates = []
    for label, score in scores.items():
        min_score = int(rules.get("labels", {}).get(label, {}).get("min_score", 1))
        if score >= min_score:
            candidates.append((label, score))

    if not candidates:
        return rules.get("fallback_label", "uncertain"), scores, matched_patterns

    priority = rules.get("label_priority", list(scores.keys()))
    candidates.sort(
        key=lambda item: (item[1], -priority.index(item[0]) if item[0] in priority else -10**6),
        reverse=True,
    )
    return candidates[0][0], scores, matched_patterns


def apply_weak_label_rules(df: pd.DataFrame, rules: Dict[str, Any]) -> pd.DataFrame:
    compiled = _compile_rules(rules)
    weak_labels = []
    weak_scores = []
    matched = []

    for _, row in df.iterrows():
        pred, scores, matched_patterns = _predict_one(row, rules, compiled)
        weak_labels.append(pred)
        weak_scores.append(json.dumps(scores, ensure_ascii=False))
        matched.append(
            json.dumps(
                {label: fields for label, fields in matched_patterns.items() if fields},
                ensure_ascii=False,
            )
        )

    out = pd.DataFrame(index=df.index)
    out["weak_label"] = weak_labels
    out["weak_label_scores"] = weak_scores
    out["weak_rule_matches"] = matched
    return out


def summarize_rule_hits(weak_label: str, weak_rule_matches: str, risk_flags: Dict[str, int]) -> str:
    summary = {
        "weak_label": weak_label,
        "weak_rule_matches": weak_rule_matches,
        "risk_flags": {k: int(v) for k, v in risk_flags.items() if int(v) > 0},
    }
    return json.dumps(summary, ensure_ascii=False)
