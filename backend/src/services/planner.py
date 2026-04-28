"""Service responsible for converting the research topic into actionable tasks."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, List, Optional

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState, TodoItem
from config import Configuration
from prompts import get_current_date, todo_planner_instructions
from utils import strip_thinking_tokens
from services.resilience import validate_query

logger = logging.getLogger(__name__)

TOOL_CALL_PATTERN = re.compile(
    r"\[TOOL_CALL:(?P<tool>[^:]+):(?P<body>[^\]]+)\]",
    re.IGNORECASE,
)

class PlanningService:
    """Wraps the planner agent to produce structured TODO items."""

    def __init__(self, planner_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = planner_agent
        self._config = config

    def plan_todo_list(self, state: SummaryState) -> List[TodoItem]:
        """Ask the planner agent to break the topic into actionable tasks."""

        prompt = todo_planner_instructions.format(
            current_date=get_current_date(),
            research_topic=state.research_topic,
        )

        try:
            response = self._agent.run(prompt)
        except Exception as exc:
            logger.exception("Planner LLM call failed: %s", exc)
            self._agent.clear_history()
            logger.warning("Planner failed, creating fallback task")
            return [self.create_fallback_task(state)]

        self._agent.clear_history()

        logger.info("Planner raw output (truncated): %s", response[:500])

        tasks_payload = self._extract_tasks(response)

        # 第5层Fallback：如果所有层都失败，尝试严格JSON重试
        if not tasks_payload:
            logger.warning("All 4 fallback layers exhausted, attempting strict JSON regeneration")
            tasks_payload = self._regenerate_strict_json(state)

        todo_items: List[TodoItem] = []

        for idx, item in enumerate(tasks_payload, start=1):
            title = str(item.get("title") or f"任务{idx}").strip()
            intent = str(item.get("intent") or "聚焦主题的关键问题").strip()
            raw_query = str(item.get("query") or state.research_topic).strip()

            # Query验证，无效时使用修正值
            is_valid, query = validate_query(raw_query, state.research_topic)
            if not is_valid:
                logger.warning(
                    "Query validation failed for task %d: '%s' -> '%s'",
                    idx, raw_query, query
                )

            task = TodoItem(
                id=idx,
                title=title,
                intent=intent,
                query=query,
            )
            todo_items.append(task)

        # 如果没有任何有效任务，创建fallback任务
        if not todo_items:
            logger.warning("No valid tasks generated, creating fallback task")
            todo_items = [self.create_fallback_task(state)]

        state.todo_items = todo_items

        titles = [task.title for task in todo_items]
        logger.info("Planner produced %d tasks: %s", len(todo_items), titles)
        return todo_items

    @staticmethod
    def create_fallback_task(state: SummaryState) -> TodoItem:
        """Create a minimal fallback task when planning failed."""

        return TodoItem(
            id=1,
            title="基础背景梳理",
            intent="收集主题的核心背景与最新动态",
            query=f"{state.research_topic} 最新进展" if state.research_topic else "基础背景梳理",
        )

    def _regenerate_strict_json(self, state: SummaryState) -> List[dict[str, Any]]:
        """
        第5层Fallback：使用严格的JSON格式prompt重新生成任务列表。

        最多重试2次，如果仍然失败则返回空列表。
        """
        strict_prompt = (
            f"研究主题：{state.research_topic}\n\n"
            "请严格按照以下JSON格式输出任务列表，不要包含任何其他文字：\n"
            '[{"title":"任务标题","intent":"任务意图","query":"搜索查询"}]\n\n'
            "要求：\n"
            "1. 只输出JSON数组，不要有markdown代码块标记\n"
            "2. 每个任务至少3个字符的标题\n"
            "3. 查询词至少5个字符\n"
            "4. 不要包含特殊符号或纯数字查询\n"
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self._agent.run(strict_prompt)
                self._agent.clear_history()

                # 清理可能存在的markdown标记
                text = response.strip()
                text = re.sub(r"^```json\s*", "", text)
                text = re.sub(r"^```\s*", "", text)
                text = re.sub(r"\s*```$", "", text)

                # 尝试找到JSON数组
                start = text.find("[")
                end = text.rfind("]")
                if start != -1 and end != -1 and end > start:
                    parsed = json.loads(text[start : end + 1])
                    if isinstance(parsed, list) and parsed:
                        # 过滤并验证每个任务
                        valid_tasks = []
                        for item in parsed:
                            if isinstance(item, dict):
                                title = str(item.get("title", "")).strip()
                                query = str(item.get("query", "")).strip()
                                # 验证标题和查询的有效性
                                if len(title) >= 3 and len(query) >= 3:
                                    valid_tasks.append({
                                        "title": title,
                                        "intent": str(item.get("intent", title)).strip(),
                                        "query": query,
                                    })
                        if valid_tasks:
                            logger.info(
                                "Strict JSON regeneration succeeded on attempt %d, got %d tasks",
                                attempt + 1, len(valid_tasks)
                            )
                            return valid_tasks
            except Exception as exc:  # pragma: no cover - 防御性容错
                logger.warning(
                    "Strict JSON regeneration attempt %d/%d failed: %s",
                    attempt + 1, max_retries, exc
                )

        logger.error("Strict JSON regeneration failed after %d attempts", max_retries)
        return []

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------
    def _extract_tasks(self, raw_response: str) -> List[dict[str, Any]]:
        """Parse planner output into a list of task dictionaries."""

        text = raw_response.strip()
        if self._config.strip_thinking_tokens:
            text = strip_thinking_tokens(text)

        json_payload = self._extract_json_payload(text)
        tasks: List[dict[str, Any]] = []

        if isinstance(json_payload, dict):
            candidate = json_payload.get("tasks")
            if isinstance(candidate, list):
                for item in candidate:
                    if isinstance(item, dict):
                        tasks.append(item)
        elif isinstance(json_payload, list):
            for item in json_payload:
                if isinstance(item, dict):
                    tasks.append(item)

        if not tasks:
            tool_payload = self._extract_tool_payload(text)
            if tool_payload and isinstance(tool_payload.get("tasks"), list):
                for item in tool_payload["tasks"]:
                    if isinstance(item, dict):
                        tasks.append(item)

        return tasks

    def _extract_json_payload(self, text: str) -> Optional[dict[str, Any] | list]:
        """Try to locate and parse a JSON object or array from the text."""

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                return None

        return None

    def _extract_tool_payload(self, text: str) -> Optional[dict[str, Any]]:
        """Parse the first TOOL_CALL expression in the output."""

        match = TOOL_CALL_PATTERN.search(text)
        if not match:
            return None

        body = match.group("body")

        try:
            payload = json.loads(body)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        parts = [segment.strip() for segment in body.split(",") if segment.strip()]
        payload: dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            payload[key.strip()] = value.strip().strip('"').strip("'")

        return payload or None
