"""Service that consolidates task results into the final report."""

from __future__ import annotations

import json
import logging
from datetime import datetime

from hello_agents import ToolAwareSimpleAgent

from models import SummaryState
from config import Configuration
from utils import strip_thinking_tokens
from services.text_processing import strip_tool_calls

logger = logging.getLogger(__name__)


class ReportingService:
    """Generates the final structured report."""

    def __init__(self, report_agent: ToolAwareSimpleAgent, config: Configuration) -> None:
        self._agent = report_agent
        self._config = config

    def generate_report(self, state: SummaryState) -> str:
        """Generate a structured report based on completed tasks."""

        try:
            return self._generate_with_agent(state)
        except Exception as exc:  # pragma: no cover - 防御性容错
            logger.exception("Report agent failed: %s", exc)
            self._agent.clear_history()
            return self._generate_fallback_report(state)

    def _generate_with_agent(self, state: SummaryState) -> str:
        """使用Agent生成报告（可能抛出异常）"""

        tasks_block = []
        for task in state.todo_items:
            summary_block = task.summary or "暂无可用信息"
            sources_block = task.sources_summary or "暂无来源"
            tasks_block.append(
                f"### 任务 {task.id}: {task.title}\n"
                f"- 任务目标：{task.intent}\n"
                f"- 检索查询：{task.query}\n"
                f"- 执行状态：{task.status}\n"
                f"- 任务总结：\n{summary_block}\n"
                f"- 来源概览：\n{sources_block}\n"
            )

        note_references = []
        for task in state.todo_items:
            if task.note_id:
                note_references.append(
                    f"- 任务 {task.id}《{task.title}》：note_id={task.note_id}"
                )

        notes_section = "\n".join(note_references) if note_references else "- 暂无可用任务笔记"

        read_template = json.dumps({"action": "read", "note_id": "<note_id>"}, ensure_ascii=False)
        create_conclusion_template = json.dumps(
            {
                "action": "create",
                "title": f"研究报告：{state.research_topic}",
                "note_type": "conclusion",
                "tags": ["deep_research", "report"],
                "content": "请在此沉淀最终报告要点",
            },
            ensure_ascii=False,
        )

        prompt = (
            f"研究主题：{state.research_topic}\n"
            f"任务概览：\n{''.join(tasks_block)}\n"
            f"可用任务笔记：\n{notes_section}\n"
            f"请针对每条任务笔记使用格式：[TOOL_CALL:note:{read_template}] 读取内容，整合所有信息后撰写报告。\n"
            f"如需输出汇总结论，可追加调用：[TOOL_CALL:note:{create_conclusion_template}] 保存报告要点。"
        )

        response = self._agent.run(prompt)
        self._agent.clear_history()

        report_text = response.strip()
        if self._config.strip_thinking_tokens:
            report_text = strip_thinking_tokens(report_text)

        report_text = strip_tool_calls(report_text).strip()

        if not report_text:
            logger.warning("Agent returned empty report, using fallback")
            return self._generate_fallback_report(state)

        return report_text

    def _generate_fallback_report(self, state: SummaryState) -> str:
        """
        生成降级报告模板。

        当Agent调用失败或返回空时使用此方法。
        """
        logger.info("Generating fallback report for topic: %s", state.research_topic)

        task_summaries = []
        for task in state.todo_items:
            if task.summary:
                task_summaries.append(f"## {task.title}\n{task.summary}\n")

        failed_tasks = [t for t in state.todo_items if t.status == "failed"]
        skipped_tasks = [t for t in state.todo_items if t.status == "skipped"]

        issues_section = ""
        if failed_tasks:
            issues_section += "\n### 执行异常的任务\n"
            for task in failed_tasks:
                issues_section += f"- {task.title}: {task.error_detail or '未知错误'}\n"
        if skipped_tasks:
            issues_section += "\n### 跳过的任务\n"
            for task in skipped_tasks:
                issues_section += f"- {task.title}: {task.intent}\n"

        separator = "\n---\n"
        task_content = separator.join(task_summaries) if task_summaries else "暂无有效摘要。"

        return f"""# 研究报告：{state.research_topic}

## 说明
自动报告生成服务暂时不可用，以下为任务摘要汇总。

{task_content}

{issues_section if issues_section else ''}

---
*报告生成时间：{datetime.now().isoformat()}*
*此为降级报告，内容可能不完整*
"""

