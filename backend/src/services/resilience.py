"""统一容错工具库 - 全系统共享"""

from __future__ import annotations

import functools
import logging
import re
import signal
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


# ============ 1. 重试装饰器 ============


def retry_with_backoff(
    max_retries: int = 3,
    initial_backoff: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    指数退避重试装饰器

    用法:
    @retry_with_backoff(max_retries=3, initial_backoff=1.0)
    def search_with_retry(query):
        ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # pragma: no cover - 防御性容错
                    last_exception = exc
                    if attempt < max_retries - 1:
                        wait_time = initial_backoff * (backoff_multiplier**attempt)
                        logger.warning(
                            "%s failed (attempt %d/%d), retrying in %.1fs: %s",
                            func.__name__,
                            attempt + 1,
                            max_retries,
                            wait_time,
                            exc,
                        )
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__,
                            max_retries,
                            exc,
                        )
            if last_exception:
                raise last_exception
            raise RuntimeError(f"{func.__name__} failed after {max_retries} attempts")

        return wrapper

    return decorator


# ============ 2. 超时装饰器 ============


def timeout(seconds: float) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    超时控制装饰器（适用于同步函数）

    用法:
    @timeout(seconds=30)
    def long_running_task():
        ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            def handler(signum: int, frame: Any) -> None:
                raise TimeoutError(f"{func.__name__} exceeded {seconds}s timeout")

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(int(seconds))
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

        return wrapper

    return decorator


# ============ 3. 降级装饰器 ============


def fallback(
    default: Any = None, exceptions: tuple = (Exception,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    降级处理装饰器 - 失败时返回默认值

    用法:
    @fallback(default="降级内容")
    def risky_operation():
        ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except exceptions as exc:  # pragma: no cover - 防御性容错
                logger.warning("%s failed, using fallback: %s", func.__name__, exc)
                return default

        return wrapper

    return decorator


# ============ 4. 熔断器 ============


@dataclass
class CircuitBreakerState:
    """熔断器内部状态"""

    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False


class CircuitBreaker:
    """
    熔断器 - 连续失败超过阈值后"跳闸"，避免雪崩

    用法:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    result = breaker.call(risky_function, fallback_value)
    """

    def __init__(
        self, failure_threshold: int = 5, recovery_timeout: float = 60.0
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._state = CircuitBreakerState()

    def call(self, func: Callable[..., T], fallback: T = None) -> T:
        """执行函数，失败时触发熔断"""
        # 检查是否需要恢复（半开状态）
        if self._state.is_open:
            if (
                time.time() - self._state.last_failure_time
                > self.recovery_timeout
            ):
                logger.info("Circuit breaker entering half-open state")
                self._state.is_open = False
                self._state.failure_count = 0
            else:
                logger.warning(
                    "Circuit breaker is OPEN, skipping %s", func.__name__
                )
                return fallback

        try:
            result = func()
            # 成功，重置计数器
            self._state.failure_count = 0
            return result
        except Exception as exc:  # pragma: no cover - 防御性容错
            self._state.failure_count += 1
            self._state.last_failure_time = time.time()
            if self._state.failure_count >= self.failure_threshold:
                logger.error(
                    "Circuit breaker OPENED after %d failures",
                    self._state.failure_count,
                )
                self._state.is_open = True
            raise exc

    def reset(self) -> None:
        """手动重置熔断器"""
        self._state = CircuitBreakerState()


# ============ 5. 验证工具函数 ============


def validate_query(query: str, topic: str) -> tuple[bool, str]:
    """
    验证Query有效性，无效时返回修正值

    Returns: (is_valid, query_or_fallback)
    """
    if not query or not query.strip():
        return False, f"{topic} 最新动态"

    query = query.strip()

    # 纯标点检测
    if not re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", query):
        return False, f"{topic} 最新动态"

    # 最小长度检测
    has_cjk = bool(re.search(r"[\u4e00-\u9fff]", query))
    min_len = 3 if has_cjk else 5
    if len(query) < min_len:
        return False, f"{topic} 最新动态"

    # 乱码检测（连续30+拉丁字母）
    if re.match(r"^[a-zA-Z]{30,}$", query):
        return False, f"{topic} 最新动态"

    return True, query


def validate_content(
    content: str | None, min_length: int = 10
) -> tuple[bool, str]:
    """
    验证内容有效性，无效时返回降级内容
    """
    if not content or not content.strip():
        return False, "暂无相关信息，请参考上述搜索结果。"

    content = content.strip()

    # 纯标点/空白检测
    if not re.search(r"[a-zA-Z0-9\u4e00-\u9fff]", content):
        return False, "暂无相关信息，请参考上述搜索结果。"

    if len(content) < min_length:
        return False, "暂无相关信息，请参考上述搜索结果。"

    return True, content


def safe_truncate(text: str, max_chars: int = 6000) -> str:
    """
    安全截断，避免截断到半个字

    Args:
        text: 原始文本
        max_chars: 最大字符数

    Returns:
        截断后的文本
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."
