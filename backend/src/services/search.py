"""Search dispatch helpers leveraging HelloAgents SearchTool."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Tuple

from hello_agents.tools import SearchTool

from config import Configuration
from utils import (
    deduplicate_and_format_sources,
    format_sources,
    get_config_value,
)

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_SOURCE = 2000
SEARCH_TIMEOUT = 30  # 搜索超时秒数
MAX_RETRIES = 3  # 最大重试次数
INITIAL_BACKOFF = 1.0  # 初始退避时间（秒）
BACKOFF_MULTIPLIER = 2.0  # 退避倍数

_GLOBAL_SEARCH_TOOL = SearchTool(backend="hybrid")


def _execute_search_with_retry(
    query: str,
    search_api: str,
    config: Configuration,
    loop_count: int,
) -> Any:
    """
    执行搜索，带重试机制（指数退避）

    最多重试MAX_RETRIES次，每次失败后等待时间翻倍：
    1s -> 2s -> 4s
    """
    last_exception: Exception | None = None

    for attempt in range(MAX_RETRIES):
        try:
            raw_response = _GLOBAL_SEARCH_TOOL.run(
                {
                    "input": query,
                    "backend": search_api,
                    "mode": "structured",
                    "fetch_full_page": config.fetch_full_page,
                    "max_results": 5,
                    "max_tokens_per_source": MAX_TOKENS_PER_SOURCE,
                    "loop_count": loop_count,
                }
            )
            return raw_response
        except Exception as exc:  # pragma: no cover - 防御性容错
            last_exception = exc
            if attempt < MAX_RETRIES - 1:
                wait_time = INITIAL_BACKOFF * (BACKOFF_MULTIPLIER**attempt)
                logger.warning(
                    "Search attempt %d/%d failed for %s: %s. Retrying in %.1fs...",
                    attempt + 1, MAX_RETRIES, search_api, exc, wait_time
                )
                time.sleep(wait_time)
            else:
                logger.error(
                    "Search failed after %d attempts for %s: %s",
                    MAX_RETRIES, search_api, exc
                )

    # 如果所有重试都失败，抛出最后一个异常
    if last_exception:
        raise last_exception
    raise RuntimeError(f"Search failed after {MAX_RETRIES} attempts")


def dispatch_search(
    query: str,
    config: Configuration,
    loop_count: int,
) -> Tuple[dict[str, Any] | None, list[str], Optional[str], str]:
    """
    Execute configured search backend and normalise response payload.

    带有重试机制和降级处理：
    - 网络失败：最多重试3次（指数退避）
    - API限流：等待后重试
    - 超时/异常：返回降级payload
    """

    search_api = get_config_value(config.search_api)

    try:
        raw_response = _execute_search_with_retry(
            query, search_api, config, loop_count
        )
    except Exception as exc:  # pragma: no cover - 防御性容错
        logger.exception("Search backend %s failed after retries: %s", search_api, exc)
        # 降级处理：返回空结果，不阻塞整个流程
        return {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": [f"搜索服务暂时不可用: {str(exc)[:100]}"],
        }, [f"搜索失败: {str(exc)[:100]}"], None, search_api

    # 降级处理字符串响应（如API返回错误消息）
    if isinstance(raw_response, str):
        notices = [raw_response]
        logger.warning("Search backend %s returned text notice: %s", search_api, raw_response)
        payload: dict[str, Any] = {
            "results": [],
            "backend": search_api,
            "answer": None,
            "notices": notices,
        }
    else:
        payload = raw_response
        notices = list(payload.get("notices") or [])

    backend_label = str(payload.get("backend") or search_api)
    answer_text = payload.get("answer")
    results = payload.get("results", [])

    # 结果为空时记录warning但不阻塞
    if not results:
        logger.warning("Search returned empty results for query: %s", query[:50])

    if notices:
        for notice in notices:
            logger.info("Search notice (%s): %s", backend_label, notice)

    logger.info(
        "Search backend=%s resolved_backend=%s answer=%s results=%s",
        search_api,
        backend_label,
        bool(answer_text),
        len(results),
    )

    return payload, notices, answer_text, backend_label


def prepare_research_context(
    search_result: dict[str, Any] | None,
    answer_text: Optional[str],
    config: Configuration,
) -> tuple[str, str]:
    """Build structured context and source summary for downstream agents."""

    sources_summary = format_sources(search_result)
    context = deduplicate_and_format_sources(
        search_result or {"results": []},
        max_tokens_per_source=MAX_TOKENS_PER_SOURCE,
        fetch_full_page=config.fetch_full_page,
    )

    if answer_text:
        context = f"AI直接答案：\n{answer_text}\n\n{context}"

    return sources_summary, context
