"""Task summarization utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple
import logging
import shutil
import time
import tiktoken

RAG_EMBEDDING_TIMEOUT = 60  # RAG embedding超时时间（秒）
from hello_agents import ToolAwareSimpleAgent
from hello_agents.tools import RAGTool
from hello_agents.context import ContextBuilder, ContextConfig
from hello_agents.memory.rag import search_vectors_expanded, index_chunks, load_and_chunk_texts
from hello_agents.memory.storage.qdrant_store import QdrantConnectionManager

if TYPE_CHECKING:
    from services.evaluator import CompressionEvaluator

from models import SummaryState, TodoItem
from config import Configuration
from utils import strip_thinking_tokens
from services.notes import build_note_guidance
from services.text_processing import strip_tool_calls
from services.resilience import validate_content, safe_truncate

logger = logging.getLogger(__name__)


class SummarizationService:
    """Handles synchronous and streaming task summarization."""

    def __init__(
        self,
        summarizer_factory: Callable[[], ToolAwareSimpleAgent],
        config: Configuration,
    ) -> None:
        self._agent_factory = summarizer_factory
        self._config = config
        self.rag_tool = RAGTool(
            knowledge_base_path="./workspace/temp_dynamic_rag",
            rag_namespace="temp_summary"
        )

        # 追踪已创建的命名空间，用于清理
        self._task_namespaces: list[str] = []

        # 压缩效果评估器（可选）
        self._evaluator: Optional["CompressionEvaluator"] = None

        # 直接实现压缩逻辑，不依赖ContextBuilder的Gather-Select流程
        # 因为我们需要直接压缩原始文本，而不是从RAG检索
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def set_evaluator(self, evaluator: "CompressionEvaluator") -> None:
        """设置压缩效果评估器"""
        self._evaluator = evaluator

    def _optimize_context(
        self, task: "TodoItem", raw_context: str, state: Optional[SummaryState] = None
    ) -> str:
        """
        对原始搜索结果进行语义压缩，目标是保留约60%的内容。

        压缩策略（RAG召回）：
        1. 将原始文本添加到RAG向量库
        2. 用task.intent作为query，通过RAG检索召回相关片段
        3. 调整limit参数控制召回量，目标是保留约60%的原始内容
        """
        logger.info(
            "[RAG Compression] 正在为任务 '%s' 进行RAG召回压缩，目标保留60%...", task.title
        )

        # 验证原始内容有效性
        is_valid, validated_content = validate_content(raw_context)
        if not is_valid:
            logger.warning(
                "Raw context validation failed for task %d: '%s...'",
                task.id, raw_context[:50] if raw_context else "empty"
            )
            return validated_content

        try:
            # 计算原始token数
            original_tokens = self._count_tokens(raw_context)
            target_tokens = int(original_tokens * 0.6)  # 目标：保留60%，减少语义割裂

            logger.info(
                "[RAG Compression] 原始token数=%d, 目标token数=%d (60%%)",
                original_tokens, target_tokens
            )

            # 1. 为当前任务创建独立的命名空间，避免数据污染
            namespace = f"task_{task.id}"
            self.rag_tool.rag_namespace = namespace
            self._task_namespaces.append(namespace)

            # 2. 直接使用RAG pipeline函数添加文本（绕过rag_tool.execute的接口问题）
            pipeline = self.rag_tool._get_pipeline(namespace)

            # 将原始文本写入临时文件供pipeline处理
            import tempfile
            import os as _os
            tmp_path = tempfile.mktemp(suffix=".md", dir="./workspace/temp_dynamic_rag")
            _os.makedirs(_os.path.dirname(tmp_path), exist_ok=True)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                f.write(raw_context)

            try:
                chunks_added = pipeline["add_documents"](
                    file_paths=[tmp_path],
                    chunk_size=800,
                    chunk_overlap=100
                )
                logger.info("[RAG Compression] 添加文档到RAG，分块数=%d", chunks_added)
            finally:
                # 清理临时文件
                try:
                    if _os.path.exists(tmp_path):
                        _os.remove(tmp_path)
                except Exception:
                    pass

            # 3. 直接调用RAG检索，用task.intent召回相关片段
            # 计算需要召回多少个chunk（假设每个chunk约800 tokens）
            estimated_chunk_size = 800
            estimated_chunks = max(1, int(original_tokens / estimated_chunk_size))
            # 目标召回约70%的chunks（提高到60%保留率）
            target_limit = max(8, int(estimated_chunks * 0.7))

            logger.info(
                "[RAG Compression] 估计原始chunks数=%d, 目标召回limit=%d",
                estimated_chunks, target_limit
            )

            # 调用RAG搜索（直接调用pipeline函数获取实际内容）
            results = pipeline["search_advanced"](
                query=f"请提取满足以下意图的核心信息：{task.intent}",
                top_k=target_limit,
                enable_mqe=True,
                enable_hyde=True
            )

            # 4. 处理检索结果 - 提取实际内容
            optimized_context = ""
            if results:
                contents = []
                for result in results:
                    meta = result.get("metadata", {})
                    content = meta.get("content", "").strip()
                    if content:
                        contents.append(content)
                optimized_context = "\n\n".join(contents)

            if not optimized_context:
                logger.warning("RAG检索返回空，使用原始内容截断")
                optimized_context = safe_truncate(raw_context, max_chars=int(len(raw_context) * 0.6))

            final_tokens = self._count_tokens(optimized_context)
            actual_ratio = final_tokens / original_tokens if original_tokens > 0 else 0
            logger.info(
                "[RAG Compression] 压缩完成: %d -> %d tokens (保留%.1f%%)",
                original_tokens, final_tokens, actual_ratio * 100
            )

            # 验证压缩结果有效性
            is_valid, _ = validate_content(optimized_context)
            if not is_valid:
                logger.warning(
                    "Compression result validation failed for task %d, using original truncated",
                    task.id
                )
                optimized_context = safe_truncate(raw_context, max_chars=int(len(raw_context) * 0.6))

            # 记录压缩评估指标
            if self._evaluator:
                self._evaluator.evaluate(
                    task_id=task.id,
                    original_text=raw_context,
                    compressed_text=optimized_context,
                    task_topic=state.research_topic if state else None,
                )

            return optimized_context

        except Exception as exc:  # pragma: no cover - 防御性容错
            logger.warning(
                "RAG compression failed for task %d: %s, falling back to truncation",
                task.id, exc
            )
            # 计算截断结果用于评估
            truncated_context = safe_truncate(raw_context, max_chars=int(len(raw_context) * 0.4))
            if self._evaluator:
                self._evaluator.evaluate(
                    task_id=task.id,
                    original_text=raw_context,
                    compressed_text=truncated_context,
                    task_topic=state.research_topic if state else None,
                )
            return truncated_context

    def _count_tokens(self, text: str) -> int:
        """计算文本token数"""
        try:
            return len(self._encoding.encode(text))
        except Exception:
            return len(text) // 4

    def cleanup(self) -> None:
        """
        清理所有RAG临时资源。

        包括：
        - 删除Qdrant中的任务命名空间
        - 删除本地临时目录
        """
        logger.info("Starting RAG cleanup for %d namespaces", len(self._task_namespaces))

        # 清理Qdrant命名空间
        for namespace in self._task_namespaces:
            try:
                self.rag_tool._clear_knowledge_base(confirm=True, namespace=namespace)
                logger.debug("Cleaned up namespace: %s", namespace)
            except Exception as exc:  # pragma: no cover - 防御性容错
                logger.warning("Failed to cleanup namespace %s: %s", namespace, exc)

        self._task_namespaces.clear()

        # 清理本地临时目录
        temp_dir = Path("./workspace/temp_dynamic_rag")
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug("Cleaned up temp directory: %s", temp_dir)
            except Exception as exc:  # pragma: no cover - 防御性容错
                logger.warning("Failed to cleanup temp directory %s: %s", temp_dir, exc)

        logger.info("RAG cleanup completed")

    def count_tokens(self, text: str) -> int:
        """计算文本token数（使用tiktoken）"""
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:  # pragma: no cover - 防御性容错
            # 降级方案：粗略估算（1 token ≈ 4 字符）
            return len(text) // 4


    def summarize_task(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Generate a task-specific summary using the summarizer agent."""

        # 上下文优化（容错：RAG失败会降级到截断）
        optimized_context = self._optimize_context(task, context, state)
        prompt = self._build_prompt(state, task, optimized_context)

        agent = self._agent_factory()
        try:
            response = agent.run(prompt)
        except Exception as exc:  # pragma: no cover - 防御性容错
            logger.exception("Summarizer agent failed for task %d: %s", task.id, exc)
            agent.clear_history()
            return f"任务总结生成失败: {str(exc)[:100]}"
        finally:
            agent.clear_history()

        summary_text = response.strip()
        if self._config.strip_thinking_tokens:
            summary_text = strip_thinking_tokens(summary_text)

        summary_text = strip_tool_calls(summary_text).strip()

        return summary_text or "暂无可用信息"

    def stream_task_summary(
        self, state: SummaryState, task: TodoItem, context: str
    ) -> Tuple[Iterator[str], Callable[[], str]]:
        """Stream the summary text for a task while collecting full output."""
        original_token = self.count_tokens(context)
        origin_time = time.time()
        print(f"时间{origin_time}-原始token数为:{original_token}")
        optimized_context = self._optimize_context(task, context, state)
        current_time = time.time()
        optimize_token = self.count_tokens(optimized_context)
        print(f"时间{current_time}-现在token数为:{optimize_token}")
        prompt = self._build_prompt(state, task, optimized_context)
        remove_thinking = self._config.strip_thinking_tokens
        raw_buffer = ""
        visible_output = ""
        emit_index = 0
        agent = self._agent_factory()

        def flush_visible() -> Iterator[str]:
            nonlocal emit_index, raw_buffer
            while True:
                start = raw_buffer.find("<think>", emit_index)
                if start == -1:
                    if emit_index < len(raw_buffer):
                        segment = raw_buffer[emit_index:]
                        emit_index = len(raw_buffer)
                        if segment:
                            yield segment
                    break

                if start > emit_index:
                    segment = raw_buffer[emit_index:start]
                    emit_index = start
                    if segment:
                        yield segment

                end = raw_buffer.find("</think>", start)
                if end == -1:
                    break
                emit_index = end + len("</think>")

        def generator() -> Iterator[str]:
            nonlocal raw_buffer, visible_output, emit_index
            try:
                for chunk in agent.stream_run(prompt):
                    raw_buffer += chunk
                    if remove_thinking:
                        for segment in flush_visible():
                            visible_output += segment
                            if segment:
                                yield segment
                    else:
                        visible_output += chunk
                        if chunk:
                            yield chunk
            finally:
                if remove_thinking:
                    for segment in flush_visible():
                        visible_output += segment
                        if segment:
                            yield segment
                agent.clear_history()



        def get_summary() -> str:
            if remove_thinking:
                cleaned = strip_thinking_tokens(visible_output)
            else:
                cleaned = visible_output

            return strip_tool_calls(cleaned).strip()

        return generator(), get_summary

    def _build_prompt(self, state: SummaryState, task: TodoItem, context: str) -> str:
        """Construct the summarization prompt shared by both modes."""

        return (
            f"任务主题：{state.research_topic}\n"
            f"任务名称：{task.title}\n"
            f"任务目标：{task.intent}\n"
            f"检索查询：{task.query}\n"
            f"任务上下文（已提取核心片段）：\n{context}\n" 
            f"{build_note_guidance(task)}\n"
            "请按照以上协作要求先同步笔记，然后返回一份面向用户的 Markdown 总结（仍遵循任务总结模板）。"
        )
