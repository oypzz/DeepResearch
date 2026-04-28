"""Utility helpers for normalizing agent generated text."""

from __future__ import annotations

import logging
import re

from typing import Any

logger = logging.getLogger(__name__)


def strip_tool_calls(text: str) -> str:
    """移除文本中的工具调用标记。"""

    if not text:
        return text

    pattern = re.compile(r"\[TOOL_CALL:[^\]]+\]")
    return pattern.sub("", text)


def deduplicate_and_format_sources(
    search_result: dict[str, Any] | None,
    max_tokens_per_source: int = 2000,
    fetch_full_page: bool = True,
) -> str:
    """
    去重并格式化搜索结果。

    容错策略：
    - 如果处理失败，返回降级内容（直接拼接原始摘要）
    """
    try:
        results = search_result.get("results", []) if search_result else []

        if not results:
            return "暂无搜索结果"

        formatted_parts = []
        for idx, result in enumerate(results[:5], start=1):
            title = result.get("title", "未知来源")
            url = result.get("url", "")
            content = result.get("content", "")

            # 优先使用 raw_content（完整内容），否则使用 content（摘要）
            if fetch_full_page and result.get("raw_content"):
                content = result["raw_content"]
                # 按token限制截断
                content = _truncate_by_tokens(content, max_tokens_per_source)

            if content:
                formatted_parts.append(f"[{idx}] {title}\n{content}\n来源: {url}")

        if not formatted_parts:
            return "暂无可用内容"

        return "\n\n".join(formatted_parts)

    except Exception as exc:  # pragma: no cover - 防御性容错
        logger.warning("Source formatting failed: %s, returning raw content", exc)
        # 降级处理：直接返回原始摘要
        results = search_result.get("results", []) if search_result else []
        if results:
            return "\n\n".join(
                r.get("content", "")[:500] for r in results[:3] if r.get("content")
            )
        return "信息来源处理失败"


def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    """
    按token数安全截断文本。

    估算：1 token ≈ 4 字符
    """
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


