from __future__ import annotations

from typing import Iterable, List, Dict


def llm_badcase_prompt(records: Iterable[dict]) -> str:
    """为 LLM badcase 分析准备提示词（占位接口）。"""
    rows: List[Dict] = list(records)
    return (
        "你是邮件安全分析专家，请根据以下误分类样本总结 1) 错误模式 2) 可能改进特征 3) 数据标注风险。"
        f"\n样本数量: {len(rows)}"
    )


def weak_supervision_prompt_template() -> str:
    return "请根据邮件主题、正文和结构化元数据打弱标签（0-4），并给出置信度(0-1)。"
