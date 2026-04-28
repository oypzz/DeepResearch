"""科学评估框架 - 上下文压缩效果评估"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from hello_agents import HelloAgentsLLM

logger = logging.getLogger(__name__)


@dataclass
class CompressionMetrics:
    """单次压缩操作的评估指标"""

    task_id: int
    original_length: int
    compressed_length: int
    compression_ratio: float
    recall: float = 0.0  # 召回率（原始信息保留比例）- 需LLM评估
    precision: float = 0.0  # 精确率（压缩结果相关比例）- 需LLM评估
    f1_score: float = 0.0  # F1综合得分
    semantic_similarity: float = 0.0  # 语义相似度（基于字符n-gram）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_topic: Optional[str] = None


@dataclass
class CompressionRecord:
    """压缩记录 - 用于诊断内容质量"""
    task_id: int
    original_text: str
    compressed_text: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    task_topic: Optional[str] = None


class CompressionEvaluator:
    """
    上下文压缩效果评估器。

    评估指标：
    1. 压缩率 (Compression Ratio): compressed_length / original_length
    2. 语义相似度 (Semantic Similarity): 基于字符n-gram的Jaccard相似度
    3. 召回率/精确率: 使用LLM评估

    使用方式：
    ```python
    evaluator = CompressionEvaluator()
    evaluator.set_llm(llm)  # 可选：设置LLM用于召回/精确率评估
    metrics = evaluator.evaluate(task_id=1, original_text="...", compressed_text="...")
    evaluator.save_report()
    ```
    """

    def __init__(
        self,
        output_dir: str = "./workspace/evaluation",
        llm: Optional["HelloAgentsLLM"] = None,
    ) -> None:
        """
        初始化评估器。

        Args:
            output_dir: 评估报告输出目录
            llm: 可选的LLM实例，用于召回率和精确率的LLM评估
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._metrics: list[CompressionMetrics] = []
        self._records: list[CompressionRecord] = []  # 新增：原始和压缩内容
        self._llm = llm

    def set_llm(self, llm: "HelloAgentsLLM") -> None:
        """设置LLM实例，用于召回率和精确率的评估"""
        self._llm = llm
        logger.debug("LLM set for recall/precision evaluation")

    def evaluate(
        self,
        task_id: int,
        original_text: str,
        compressed_text: str,
        task_topic: Optional[str] = None,
    ) -> CompressionMetrics:
        """
        评估单次压缩效果并记录指标。

        Args:
            task_id: 任务ID
            original_text: 原始文本
            compressed_text: 压缩后文本
            task_topic: 任务主题（可选）

        Returns:
            CompressionMetrics: 评估指标对象
        """
        original_len = len(original_text)
        compressed_len = len(compressed_text)

        # 计算压缩率
        compression_ratio = (
            compressed_len / original_len if original_len > 0 else 1.0
        )

        # 计算语义相似度（基于字符3-gram的Jaccard相似度）
        semantic_similarity = self._compute_similarity(original_text, compressed_text)

        # 如果有LLM，评估召回率和精确率
        recall = 0.0
        precision = 0.0
        if self._llm:
            try:
                recall, precision = self._llm_evaluate_quality(
                    original_text,
                    compressed_text,
                    task_topic=task_topic
                )
            except Exception as exc:  # pragma: no cover - 防御性容错
                logger.warning(
                    "LLM evaluation failed for task %d: %s, using defaults",
                    task_id, exc
                )

        # 计算F1得分
        f1_score = (
            2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0
        )

        metrics = CompressionMetrics(
            task_id=task_id,
            original_length=original_len,
            compressed_length=compressed_len,
            compression_ratio=compression_ratio,
            recall=recall,
            precision=precision,
            f1_score=f1_score,
            semantic_similarity=semantic_similarity,
            task_topic=task_topic,
        )

        self._metrics.append(metrics)

        # 保存压缩记录用于诊断
        record = CompressionRecord(
            task_id=task_id,
            original_text=original_text,
            compressed_text=compressed_text,
            task_topic=task_topic,
        )
        self._records.append(record)

        logger.debug(
            "Compression metrics for task %d: ratio=%.2f, similarity=%.2f, recall=%.2f, precision=%.2f",
            task_id, compression_ratio, semantic_similarity, recall, precision
        )

        return metrics

    def _llm_evaluate_quality(
        self,
        original_text: str,
        compressed_text: str,
        task_topic: Optional[str] = None
    ) -> tuple[float, float]:
        """
        使用LLM评估压缩质量。

        评估逻辑（已改进）：
        - 不再以"保留原文比例"计算召回率（因为原始文本可能包含大量冗余）
        - 改为判断"压缩结果是否有效支持任务目标"
        - 召回率：压缩结果是否覆盖了任务所需的核心信息
        - 精确率：压缩结果中有多少是直接服务于任务的

        Args:
            original_text: 原始文本（保留作为参数兼容，实际评估不依赖原文完整性）
            compressed_text: 压缩后文本
            task_topic: 研究主题（用于提供评估上下文）

        Returns:
            tuple[float, float]: (recall, precision)
        """
        # 构建评估prompt
        topic_context = f"\n## 研究主题：{task_topic}\n" if task_topic else "\n"

        eval_prompt = f"""你是一个文本压缩质量评估专家。请评估以下压缩结果对于完成研究任务的有效性。

{topic_context}
## 压缩后文本（需要评估的内容）：
{compressed_text}

## 评估任务：

请假设你是研究员，需要根据压缩后的内容完成研究。请从以下两个维度评分（0.0 ~ 1.0）：

1. **召回率 (Recall)** - 压缩结果是否涵盖了完成任务所需的核心信息？
   - 1.0 = 压缩结果提供了完成研究任务所需的全部关键信息，无重大遗漏
   - 0.7 = 压缩结果覆盖了大部分关键信息，有轻微遗漏
   - 0.5 = 压缩结果覆盖了一半关键信息，存在明显遗漏
   - 0.3 = 压缩结果仅包含少量有用信息，大量关键信息缺失
   - 0.0 = 压缩结果几乎没有有用的信息，无法支持任务

2. **精确率 (Precision)** - 压缩结果中有多大比例是直接服务于研究任务的？
   - 1.0 = 所有内容都高度相关，没有废话
   - 0.7 = 大部分内容相关，有少量无关信息
   - 0.5 = 一半内容相关，一半是噪音或边缘信息
   - 0.3 = 大部分内容是噪音，仅少量相关
   - 0.0 = 几乎所有内容都与研究任务无关

请直接输出以下JSON格式（不要有任何其他文字）：
{{"recall": 0.0~1.0的数值, "precision": 0.0~1.0的数值}}

示例输出：
{{"recall": 0.75, "precision": 0.85}}
"""
        try:
            response = self._llm.invoke([{"role": "user", "content": eval_prompt}])

            # 提取JSON
            json_match = re.search(r'\{[^{}]*"recall"\s*:\s*[\d.]+[^}]*\}', response, re.DOTALL)
            if not json_match:
                # 尝试更宽松的匹配
                json_match = re.search(r'\{[^}]+\}', response)

            if json_match:
                result = json.loads(json_match.group())
                recall = float(result.get("recall", 0.0))
                precision = float(result.get("precision", 0.0))
                # 确保在有效范围内
                recall = max(0.0, min(1.0, recall))
                precision = max(0.0, min(1.0, precision))
                logger.debug("LLM evaluation result: recall=%.2f, precision=%.2f", recall, precision)
                return recall, precision

        except Exception as exc:  # pragma: no cover - 防御性容错
            logger.warning("Failed to parse LLM evaluation response: %s", exc)

        return 0.0, 0.0

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度。

        使用字符n-gram的Jaccard相似度作为代理指标。
        注意：这是一种简单的近似方法，不考虑语义深度。

        Args:
            text1: 文本1
            text2: 文本2

        Returns:
            float: 0.0 ~ 1.0 的相似度分数
        """
        if not text1 or not text2:
            return 0.0

        def get_ngrams(text: str, n: int = 3) -> set[str]:
            """获取字符n-gram集合"""
            text = text.lower()
            return set(text[i : i + n] for i in range(len(text) - n + 1))

        grams1 = get_ngrams(text1)
        grams2 = get_ngrams(text2)

        if not grams1 or not grams2:
            return 0.0

        intersection = len(grams1 & grams2)
        union = len(grams1 | grams2)

        return intersection / union if union > 0 else 0.0

    def save_report(self, research_id: Optional[str] = None) -> Path:
        """
        保存评估报告到JSON文件。

        Args:
            research_id: 研究ID（可选，默认使用时间戳）

        Returns:
            Path: 报告文件路径
        """
        research_id = research_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self._output_dir / f"compression_eval_{research_id}.json"

        # 计算汇总统计
        total = len(self._metrics)
        avg_compression_ratio = (
            sum(m.compression_ratio for m in self._metrics) / total if total > 0 else 0
        )
        avg_semantic_similarity = (
            sum(m.semantic_similarity for m in self._metrics) / total
            if total > 0
            else 0
        )

        report_data = {
            "research_id": research_id,
            "generated_at": datetime.now().isoformat(),
            "total_compressions": total,
            "metrics": [
                {
                    "task_id": m.task_id,
                    "original_length": m.original_length,
                    "compressed_length": m.compressed_length,
                    "compression_ratio": round(m.compression_ratio, 4),
                    "recall": m.recall,
                    "precision": m.precision,
                    "f1_score": m.f1_score,
                    "semantic_similarity": round(m.semantic_similarity, 4),
                    "timestamp": m.timestamp,
                    "task_topic": m.task_topic,
                }
                for m in self._metrics
            ],
            "summary": {
                "avg_compression_ratio": round(avg_compression_ratio, 4),
                "avg_semantic_similarity": round(avg_semantic_similarity, 4),
                "min_compression_ratio": (
                    min(m.compression_ratio for m in self._metrics) if self._metrics else 0
                ),
                "max_compression_ratio": (
                    max(m.compression_ratio for m in self._metrics) if self._metrics else 0
                ),
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        # 保存详细内容（含原始和压缩文本）
        detailed_path = self._output_dir / f"compression_detail_{research_id}.json"
        detailed_data = {
            "research_id": research_id,
            "generated_at": datetime.now().isoformat(),
            "records": [
                {
                    "task_id": r.task_id,
                    "original_length": len(r.original_text),
                    "compressed_length": len(r.compressed_text),
                    "compression_ratio": len(r.compressed_text) / len(r.original_text) if r.original_text else 1.0,
                    "original_snippet": r.original_text[:2000] + "..." if len(r.original_text) > 2000 else r.original_text,
                    "compressed_snippet": r.compressed_text[:2000] + "..." if len(r.compressed_text) > 2000 else r.compressed_text,
                    "task_topic": r.task_topic,
                }
                for r in self._records
            ]
        }
        with open(detailed_path, "w", encoding="utf-8") as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)

        # 保存完整的原始搜索结果和压缩结果到单独的文件
        original_only_path = self._output_dir / f"original_search_{research_id}.txt"
        compressed_only_path = self._output_dir / f"compressed_text_{research_id}.txt"

        with open(original_only_path, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(f"{'='*80}\n")
                f.write(f"Task ID: {r.task_id}\n")
                f.write(f"Original Length: {len(r.original_text)} chars\n")
                f.write(f"{'='*80}\n\n")
                f.write(r.original_text)
                f.write("\n\n")

        with open(compressed_only_path, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(f"{'='*80}\n")
                f.write(f"Task ID: {r.task_id}\n")
                f.write(f"Compressed Length: {len(r.compressed_text)} chars\n")
                f.write(f"{'='*80}\n\n")
                f.write(r.compressed_text)
                f.write("\n\n")

        logger.info("Full original search results saved to %s", original_only_path)
        logger.info("Full compressed text saved to %s", compressed_only_path)

        logger.info(
            "Evaluation report saved to %s (avg_ratio=%.2f, avg_similarity=%.2f)",
            report_path, avg_compression_ratio, avg_semantic_similarity
        )
        logger.info("Detailed compression records saved to %s", detailed_path)

        return report_path

    def get_metrics(self) -> list[CompressionMetrics]:
        """返回所有记录的评估指标"""
        return self._metrics.copy()

    def clear(self) -> None:
        """清空所有记录的指标和记录"""
        self._metrics.clear()
        self._records.clear()
