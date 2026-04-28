#!/usr/bin/env python3
"""命令行深度研究工具 - 直接执行研究，无需前端界面"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# 添加 src 目录到路径，以便导入模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# 加载.env文件
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from config import Configuration
from agent import DeepResearchAgent


def setup_logging(verbose: bool = False) -> None:
    """
    配置日志系统。

    Args:
        verbose: 是否启用DEBUG级别日志
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python -m scripts.cli_research "AI芯片行业动态"
  python -m scripts.cli_research "特斯拉自动驾驶" --search-api duckduckgo
  python -m scripts.cli_research "量子计算进展" --output ./report.md -v
        """,
    )

    parser.add_argument(
        "topic",
        type=str,
        help="研究主题/问题",
    )

    parser.add_argument(
        "--search-api",
        type=str,
        choices=["duckduckgo", "tavily", "perplexity", "searxng", "advanced"],
        default="duckduckgo",
        help="搜索API后端 (默认: duckduckgo)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="报告输出文件路径 (默认: 打印到标准输出)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="启用详细日志输出",
    )

    return parser.parse_args()


def format_event(event: dict[str, Any]) -> str | None:
    """
    格式化事件为可读字符串。

    Args:
        event: 事件字典

    Returns:
        str | None: 格式化后的事件字符串
    """
    event_type = event.get("type", "unknown")

    if event_type == "status":
        return f"[STATUS] {event.get('message', '')}"

    elif event_type == "task_status":
        task_id = event.get("task_id", 0)
        title = event.get("title", "")
        status = event.get("status", "")
        detail = event.get("detail", "")
        if detail:
            return f"[TASK:{task_id}] {title} - {status} | {detail}"
        return f"[TASK:{task_id}] {title} - {status}"

    elif event_type == "sources":
        return f"[SOURCES] {event.get('count', 0)} 条来源"

    elif event_type == "tool_call":
        tool = event.get("tool", "")
        return f"[TOOL:{tool}]"

    elif event_type == "error":
        return f"[ERROR] {event.get('detail', 'Unknown error')}"

    return None


def main() -> int:
    """
    主函数。

    Returns:
        int: 退出码 (0=成功, 1=失败, 130=用户中断)
    """
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Deep Research Agent - CLI 模式")
    logger.info("=" * 60)
    logger.info("研究主题: %s", args.topic)
    logger.info("搜索API: %s", args.search_api)
    if args.output:
        logger.info("输出文件: %s", args.output)
    logger.info("=" * 60)

    # 构建配置
    overrides: dict[str, Any] = {"search_api": args.search_api}
    config = Configuration.from_env(overrides=overrides)

    try:
        # 创建Agent并执行研究
        agent = DeepResearchAgent(config=config)

        # 收集所有事件
        events: list[dict[str, Any]] = []
        interrupted = False

        for event in agent.run_stream(args.topic):
            events.append(event)

            # 格式化并输出事件
            formatted = format_event(event)
            if formatted:
                logger.info(formatted)

            # 检查取消标志
            if event.get("type") == "done":
                break

        # 提取最终报告
        report = ""
        for event in reversed(events):
            if event.get("type") == "final_report":
                report = event.get("report", "")
                break

        if not report:
            logger.error("未能生成报告")
            return 1

        # 输出报告
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info("=" * 60)
            logger.info("报告已保存到: %s", output_path.absolute())
        else:
            print("\n" + "=" * 80)
            print("DEEP RESEARCH REPORT")
            print("=" * 80)
            print(f"\n主题: {args.topic}\n")
            print(report)
            print("\n" + "=" * 80)

        logger.info("=" * 60)
        logger.info("研究完成")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("研究被用户中断")
        return 130

    except Exception as exc:
        logger.exception("研究失败: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
