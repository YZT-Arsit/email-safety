#!/usr/bin/env python
"""Label unlabeled email logs with an OpenAI-compatible LLM and split high-confidence silver vs hard cases.
Inputs: mail log/csv/jsonl, API endpoint, model name, and concurrency settings. Outputs: silver_dataset.jsonl and hard_cases.jsonl."""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from email_safety.data.io import ensure_columns, load_dataframe
from email_safety.preprocessing.text_clean import normalize_text

TAXONOMY = {
    "advertisement": "广告/营销类垃圾邮件，核心目的是推广、引流、商务拓客或群发广告。",
    "phishing": "钓鱼类邮件，核心目的是诱导点击链接、登录账号、验证身份、提交密码或认证信息。",
    "impersonation": "冒充类邮件，核心是伪装成领导、财务、HR、客户、供应商或系统通知来误导收件人执行操作。",
    "malicious_link_or_attachment": "恶意链接或附件类邮件，核心是投递危险链接、压缩包、脚本、宏文档或诱导下载执行。",
    "black_industry_or_policy_violation": "黑灰产/合规风险类邮件，如博彩、办证、非法发票、刷单、违规交易、灰黑产资源推广等。",
}
LABELS = list(TAXONOMY.keys())

PROMPT_TEMPLATE = """
你是一名邮件安全智能标注员。请根据邮件内容进行五分类，并给出简洁的风险推导逻辑。

# 分类体系
- advertisement: {advertisement}
- phishing: {phishing}
- impersonation: {impersonation}
- malicious_link_or_attachment: {malicious_link_or_attachment}
- black_industry_or_policy_violation: {black_industry_or_policy_violation}

# 判定要求
1. 只能从以上 5 个标签中选择 1 个最终标签。
2. 必须输出完整类别概率分布 class_probs，包含全部 5 个标签，且总和约等于 1。
3. 输出必须包含：label, confidence, class_probs, reasoning, ambiguous。
4. confidence 取值范围为 0 到 1。
5. 如果样本信息不足、多个类别边界模糊、存在明显冲突，请将 ambiguous 设为 true。
6. reasoning 请写 2 到 5 句，总结关键信号，不要输出额外解释。
7. 严格只返回 JSON。

# 返回格式
{{
  "label": "advertisement|phishing|impersonation|malicious_link_or_attachment|black_industry_or_policy_violation",
  "confidence": 0.0,
  "ambiguous": false,
  "class_probs": {{
    "advertisement": 0.0,
    "phishing": 0.0,
    "impersonation": 0.0,
    "malicious_link_or_attachment": 0.0,
    "black_industry_or_policy_violation": 0.0
  }},
  "reasoning": "..."
}}

# 待分类邮件
id: {id}
subject: {subject}
content: {content}
doccontent: {doccontent}
sender: {sender}
from: {from_field}
fromname: {fromname}
url: {url}
attach: {attach}
htmltag: {htmltag}
ip: {ip}
rcpt: {rcpt}
""".strip()

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
WRITE_LOCK = threading.Lock()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use an OpenAI-compatible LLM to build silver and hard-case datasets.")
    parser.add_argument("--input-path", type=str, default="spam_email_data.log")
    parser.add_argument("--raw-format", type=str, default="log", choices=["auto", "log", "csv", "jsonl"])
    parser.add_argument("--id-column", type=str, default="id")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--api-url", type=str, default="https://api.deepseek.com/v1/chat/completions")
    parser.add_argument("--model", type=str, default="deepseek-v3")
    parser.add_argument("--api-key-env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--silver-threshold", type=float, default=0.85)
    parser.add_argument("--hard-threshold", type=float, default=0.60)
    parser.add_argument("--silver-output", type=str, default="silver_dataset.jsonl")
    parser.add_argument("--hard-output", type=str, default="hard_cases.jsonl")
    parser.add_argument("--summary-output", type=str, default="llm_labeling_summary.json")
    return parser.parse_args()


def _safe_text(value: Any, max_len: int = 2000) -> str:
    if value is None:
        return ""
    text = str(value)
    text = normalize_text(text, lowercase=False, remove_urls=False)
    return text[:max_len]


def _build_prompt(row: Dict[str, Any], id_column: str) -> str:
    return PROMPT_TEMPLATE.format(
        advertisement=TAXONOMY["advertisement"],
        phishing=TAXONOMY["phishing"],
        impersonation=TAXONOMY["impersonation"],
        malicious_link_or_attachment=TAXONOMY["malicious_link_or_attachment"],
        black_industry_or_policy_violation=TAXONOMY["black_industry_or_policy_violation"],
        id=_safe_text(row.get(id_column, ""), 200),
        subject=_safe_text(row.get("subject", ""), 500),
        content=_safe_text(row.get("content", ""), 2500),
        doccontent=_safe_text(row.get("doccontent", ""), 2500),
        sender=_safe_text(row.get("sender", ""), 300),
        from_field=_safe_text(row.get("from", ""), 300),
        fromname=_safe_text(row.get("fromname", ""), 300),
        url=_safe_text(row.get("url", ""), 800),
        attach=_safe_text(row.get("attach", ""), 400),
        htmltag=_safe_text(row.get("htmltag", ""), 500),
        ip=_safe_text(row.get("ip", ""), 200),
        rcpt=_safe_text(row.get("rcpt", ""), 500),
    )


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = text.replace("```json", "```").replace("```JSON", "```")
    if "```" in fenced:
        for part in fenced.split("```"):
            part = part.strip()
            if not part:
                continue
            try:
                return json.loads(part)
            except json.JSONDecodeError:
                continue

    match = JSON_RE.search(text)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


def _normalize_probs(raw_probs: Any) -> Dict[str, float]:
    if not isinstance(raw_probs, dict):
        raise ValueError("class_probs must be a dict")
    probs = {}
    total = 0.0
    for label in LABELS:
        value = raw_probs.get(label, 0.0)
        try:
            value = max(0.0, float(value))
        except Exception as exc:
            raise ValueError(f"Invalid probability for {label}: {value}") from exc
        probs[label] = value
        total += value
    if total <= 0.0:
        raise ValueError("class_probs sum must be positive")
    return {label: value / total for label, value in probs.items()}


def _parse_label_result(raw_text: str) -> Dict[str, Any]:
    payload = _extract_json(raw_text)
    label = str(payload.get("label", "")).strip()
    confidence = payload.get("confidence", 0.0)
    ambiguous = payload.get("ambiguous", False)
    reasoning = str(payload.get("reasoning", "")).strip()
    class_probs = _normalize_probs(payload.get("class_probs", {}))

    if label not in LABELS:
        raise ValueError(f"Invalid label: {label}")

    try:
        confidence = float(confidence)
    except Exception as exc:
        raise ValueError(f"Invalid confidence: {confidence}") from exc
    confidence = max(0.0, min(1.0, confidence))

    if isinstance(ambiguous, str):
        ambiguous = ambiguous.strip().lower() in {"true", "1", "yes", "y"}
    else:
        ambiguous = bool(ambiguous)

    return {
        "label": label,
        "confidence": confidence,
        "ambiguous": ambiguous,
        "reasoning": reasoning,
        "class_probs": class_probs,
    }


def _call_chat_completion(api_url: str, api_key: str, model: str, prompt: str, timeout: int, temperature: float) -> str:
    payload = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": "你是严谨的邮件安全标注助手。必须严格按 JSON 输出。"},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    try:
        return result["choices"][0]["message"]["content"]
    except Exception as exc:
        raise ValueError(f"Unexpected API response: {result}") from exc


def _classify_one(row: Dict[str, Any], args: argparse.Namespace, api_key: str) -> Dict[str, Any]:
    prompt = _build_prompt(row, args.id_column)
    last_error: Optional[str] = None
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_output = _call_chat_completion(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                timeout=args.request_timeout,
                temperature=args.temperature,
            )
            parsed = _parse_label_result(raw_output)
            return {
                args.id_column: row.get(args.id_column, ""),
                "subject": _safe_text(row.get("subject", ""), 500),
                "label": parsed["label"],
                "confidence": parsed["confidence"],
                "ambiguous": parsed["ambiguous"],
                "class_probs": parsed["class_probs"],
                "reasoning": parsed["reasoning"],
                "raw_response": raw_output,
            }
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(min(2 ** attempt, 8))
    return {
        args.id_column: row.get(args.id_column, ""),
        "subject": _safe_text(row.get("subject", ""), 500),
        "label": "",
        "confidence": 0.0,
        "ambiguous": True,
        "class_probs": {label: 0.0 for label in LABELS},
        "reasoning": f"API_ERROR: {last_error or 'unknown error'}",
        "raw_response": "",
    }


def _write_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(f"Environment variable {args.api_key_env} is empty")

    df = load_dataframe(args.input_path, raw_format=args.raw_format, id_column=args.id_column)
    df = ensure_columns(
        df,
        [args.id_column, "subject", "content", "doccontent", "sender", "from", "fromname", "url", "attach", "htmltag", "ip", "rcpt"],
    )
    if args.offset > 0:
        df = df.iloc[args.offset :]
    if args.limit > 0:
        df = df.iloc[: args.limit]
    rows = df.to_dict(orient="records")

    silver_path = Path(args.silver_output)
    hard_path = Path(args.hard_output)
    summary_path = Path(args.summary_output)
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if silver_path.exists():
        silver_path.unlink()
    if hard_path.exists():
        hard_path.unlink()

    summary = {
        "input_rows": len(rows),
        "silver_rows": 0,
        "hard_rows": 0,
        "skipped_mid_confidence_rows": 0,
        "label_distribution": {},
    }

    with futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {executor.submit(_classify_one, row, args, api_key): row for row in rows}
        for future in tqdm(futures.as_completed(future_map), total=len(future_map), desc="LLM labeling"):
            result = future.result()
            label = result.get("label", "")
            confidence = float(result.get("confidence", 0.0) or 0.0)
            ambiguous = bool(result.get("ambiguous", False))
            if label:
                summary["label_distribution"][label] = summary["label_distribution"].get(label, 0) + 1

            if confidence > args.silver_threshold and not ambiguous:
                _write_jsonl(silver_path, result)
                summary["silver_rows"] += 1
            elif confidence < args.hard_threshold or ambiguous or not label:
                _write_jsonl(hard_path, result)
                summary["hard_rows"] += 1
            else:
                summary["skipped_mid_confidence_rows"] += 1

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
