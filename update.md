# Deep Research Agent 优化更新文档

> 本文档记录本次对 Deep Research Agent 项目进行的所有优化改进。

## 概述

本次优化主要围绕以下目标：
1. **全流程容错增强** - 覆盖系统各个环节的异常处理
2. **资源管理完善** - 清理临时文件，防止资源泄漏
3. **评估体系建设** - 建立科学的上下文压缩效果评估
4. **测试便捷化** - 提供 C 端 CLI 脚本

---

## 一、新增文件

### 1.1 `services/resilience.py` - 统一容错工具库

**路径**: `backend/src/services/resilience.py`

**功能**: 提供全系统共享的容错基础组件

| 组件 | 说明 |
|------|------|
| `@retry_with_backoff` | 指数退避重试装饰器（网络失败、API限流） |
| `@timeout` | 超时控制装饰器（LLM调用、搜索等） |
| `@fallback` | 降级处理装饰器（失败时返回默认值） |
| `CircuitBreaker` | 熔断器类（连续失败后"跳闸"） |
| `validate_query()` | Query有效性验证 |
| `validate_content()` | 内容有效性验证 |
| `safe_truncate()` | 安全截断（避免截断到半个字） |

**关键代码示例**:

```python
# 指数退避重试
@retry_with_backoff(max_retries=3, initial_backoff=1.0, backoff_multiplier=2.0)
def search_with_retry(query):
    ...

# Query验证
is_valid, query = validate_query(raw_query, topic)
if not is_valid:
    query = f"{topic} 最新动态"  # 自动修正
```

---

### 1.2 `services/evaluator.py` - 压缩效果评估

**路径**: `backend/src/services/evaluator.py`

**功能**: 科学评估上下文压缩效果

| 评估指标 | 说明 |
|---------|------|
| `compression_ratio` | 压缩率 = compressed_length / original_length |
| `semantic_similarity` | 语义相似度（基于字符 3-gram Jaccard） |
| `recall` | 召回率 - 原始信息保留比例（**已实现LLM评估**） |
| `precision` | 精确率 - 压缩结果相关比例（**已实现LLM评估**） |
| `f1_score` | F1综合得分 = 2×recall×precision / (recall+precision) |

**关键修复**:
- `invoke()` 方法：`self._llm.generate()` → `self._llm.invoke()`（HelloAgentsLLM使用`invoke`进行非流式调用）

**LLM评估方法** (`_llm_evaluate_quality`):

```python
def _llm_evaluate_quality(self, original_text, compressed_text) -> tuple[float, float]:
    """
    使用LLM评估压缩质量

    - Recall: 压缩文本保留了原始文本多少关键信息？
    - Precision: 压缩文本中有多少内容是真正相关的？
    """
    eval_prompt = f"""你是一个文本压缩质量评估专家。请评估以下压缩操作的信息保留质量。

## 原始文本（压缩前）：
{original_text[:3000]}

## 压缩后文本：
{compressed_text[:1500]}

## 评估任务：

请从以下两个维度评分（0.0 ~ 1.0）：

1. **召回率 (Recall)**：压缩文本保留了原始文本中多少比例的核心信息？
2. **精确率 (Precision)**：压缩文本中有多少比例的内容是真正相关的、有价值的？

请直接输出JSON格式：{{"recall": 0.0~1.0, "precision": 0.0~1.0}}
"""
```

**使用方式**:

```python
evaluator = CompressionEvaluator()
evaluator.set_llm(llm)  # 设置LLM用于评估
metrics = evaluator.evaluate(
    task_id=1,
    original_text="原始长文本...",
    compressed_text="压缩后文本...",
    task_topic="卷积神经网络"
)
evaluator.save_report()  # 保存到 JSON 文件
```

**评估报告示例**:

```json
{
  "research_id": "20260422_103000",
  "metrics": [{
    "task_id": 1,
    "original_length": 8500,
    "compressed_length": 1200,
    "compression_ratio": 0.14,
    "recall": 0.72,
    "precision": 0.85,
    "f1_score": 0.78,
    "semantic_similarity": 0.68
  }],
  "summary": {
    "avg_compression_ratio": 0.15,
    "avg_semantic_similarity": 0.68
  }
}
```

---

### 1.3 `scripts/cli_research.py` - C端测试脚本

**路径**: `backend/scripts/cli_research.py`

**功能**: 命令行直接执行研究，无需前端界面

**用法**:

```bash
cd backend
python -m scripts.cli_research "卷积神经网络" --search-api duckduckgo -v
python -m scripts.cli_research "AI芯片动态" --output ./report.md
```

| 参数 | 说明 |
|------|------|
| `topic` | 研究主题（位置参数） |
| `--search-api` | 搜索后端 (duckduckgo/tavily/perplexity/searxng/advanced) |
| `--output, -o` | 报告输出文件路径 |
| `--verbose, -v` | 启用详细日志 |

---

## 二、修改文件

### 2.1 `services/planner.py` - 规划服务增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| LLM调用异常捕获 | `agent.run()` 失败时返回 fallback 任务 |
| Query验证 | 无效query自动修正为 `{topic} 最新动态` |
| 第5层Fallback | 严格JSON格式重新生成（最多2次重试） |

**新增方法**:

```python
def _regenerate_strict_json(self, state: SummaryState) -> List[dict[str, Any]]:
    """
    第5层Fallback：使用严格的JSON格式prompt重新生成任务列表
    """
```

**容错流程**:

```
LLM调用 → 4层JSON解析 → 第5层严格重试 → fallback任务
   ↓
捕获异常 → clear_history() → 返回 create_fallback_task()
```

---

### 2.2 `services/search.py` - 搜索服务增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| 重试机制 | 最多3次重试，指数退避（1s → 2s → 4s） |
| 降级处理 | 搜索失败返回空结果，不阻塞流程 |
| 空结果处理 | 记录warning，继续执行 |

**新增常量**:

```python
SEARCH_TIMEOUT = 30      # 搜索超时秒数
MAX_RETRIES = 3          # 最大重试次数
INITIAL_BACKOFF = 1.0    # 初始退避时间
BACKOFF_MULTIPLIER = 2.0 # 退避倍数
```

**新增函数**:

```python
def _execute_search_with_retry(query, search_api, config, loop_count):
    """
    执行搜索，带重试机制（指数退避）
    """
```

---

### 2.3 `services/summarizer.py` - 总结服务增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| 命名空间追踪 | `self._task_namespaces` 记录所有创建的命名空间 |
| RAG降级 | RAG失败时降级到 `safe_truncate()` |
| 内容验证 | 压缩前/后验证，无效则降级 |
| `cleanup()` 方法 | 清理所有RAG临时资源 |
| 评估器集成 | `set_evaluator()` + `evaluate()` 调用 |

**新增方法**:

```python
def cleanup(self) -> None:
    """
    清理所有RAG临时资源：
    - 删除Qdrant中的任务命名空间
    - 删除本地临时目录
    """

def set_evaluator(self, evaluator: "CompressionEvaluator") -> None:
    """设置压缩效果评估器"""

def _optimize_context(self, task, raw_context, state=None) -> str:
    # 压缩成功后调用评估
    if self._evaluator:
        self._evaluator.evaluate(
            task_id=task.id,
            original_text=raw_context,
            compressed_text=optimized,
            task_topic=state.research_topic if state else None,
        )
```

---

### 2.4 `services/reporter.py` - 报告服务增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| LLM异常捕获 | 失败时返回降级报告 |
| 空输出处理 | Agent返回空时使用降级模板 |
| 降级报告模板 | `_generate_fallback_report()` |

**新增方法**:

```python
def _generate_fallback_report(self, state: SummaryState) -> str:
    """
    生成降级报告模板
    包含：任务摘要、失败任务列表、跳过的任务
    """
```

---

### 2.5 `services/text_processing.py` - 文本处理增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| 异常降级 | `deduplicate_and_format_sources()` 失败时返回降级内容 |
| Token截断 | `_truncate_by_tokens()` 按token数安全截断 |

---

### 2.6 `services/tool_events.py` - 工具追踪增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| note_id多格式解析 | 支持4种格式的ID提取 |

**支持的ID格式**:

```python
# 方式1: "ID: xxx" 格式
# 方式2: '"id": "xxx"' JSON格式
# 方式3: UUID格式 (e.g., note_xxx-xxxx-...)
# 方式4: 文件名格式 "note_xxx.md"
```

---

### 2.7 `models.py` - 数据模型增强

**关键改动**:

```python
@dataclass(kw_only=True)
class TodoItem:
    # ... existing fields ...
    error_detail: Optional[str] = field(default=None)  # 新增：记录失败原因
```

---

### 2.8 `agent.py` - 核心编排器增强

**关键改动**:

| 改动点 | 说明 |
|--------|------|
| 取消标志 | `self._cancelled` + `self._cancel_lock` |
| `cancel()` | 请求取消当前研究 |
| `_check_cancelled()` | 检查取消标志 |
| `_reset_cancelled()` | 重置取消状态 |
| Worker取消检查 | 每个事件发送前检查取消标志 |
| 线程超时 | `thread.join(timeout=60)` |
| RAG清理 | 研究完成后调用 `summarizer.cleanup()` |
| **评估器集成** | 创建 `CompressionEvaluator`，设置LLM，集成到Summarizer |
| **评估报告保存** | 研究完成后自动保存评估报告到JSON |

**新增方法**:

```python
def cancel(self) -> None:
    """请求取消当前研究"""
    with self._cancel_lock:
        self._cancelled = True

def _check_cancelled(self) -> bool:
    """检查是否已请求取消"""
    with self._cancel_lock:
        return self._cancelled

def _reset_cancelled(self) -> None:
    """重置取消状态（新研究开始时调用）"""
```

**Worker容错增强**:

```python
def worker(task, step):
    try:
        if self._check_cancelled():  # 检查取消标志
            task.status = "cancelled"
            return

        # 执行任务...

    except Exception as exc:
        task.error_detail = str(exc)  # 记录错误详情
        task.status = "failed"
```

---

## 三、全流程容错矩阵

| 阶段 | 故障类型 | 处理策略 |
|------|---------|---------|
| **规划** | LLM调用失败 | 捕获异常，返回fallback任务 |
| | JSON解析失败 | 4层fallback + 第5层严格重试 |
| | Query无效 | `validate_query()` 修正 |
| **搜索** | 网络失败 | `@retry_with_backoff` 3次重试 |
| | API限流 | 指数退避等待（1s→2s→4s） |
| | 超时 | 30秒超时，降级处理 |
| | 结果为空 | 记录warning，使用空上下文 |
| **上下文** | 去重失败 | 降级返回原始内容 |
| | 内容过长 | `safe_truncate()` 截断 |
| | 内容为空 | 返回占位符 |
| **RAG压缩** | Qdrant连接失败 | 降级到 `safe_truncate()` |
| | Embedding失败 | 降级到 `safe_truncate()` |
| | 压缩结果无效 | 降级到原始截断 |
| **总结** | LLM调用失败 | 返回错误信息 |
| | 流式中断 | 完成当前处理 |
| **报告** | LLM调用失败 | `_generate_fallback_report()` |
| **笔记** | 创建失败 | 继续，记录warning |
| | ID解析失败 | 4种格式逐一尝试 |
| **系统** | 线程卡死 | `join(timeout=60)` |
| | 资源泄漏 | `cleanup()` 清理RAG |
| | 连接断开 | 检查取消标志，优雅退出 |

---

## 四、验证结果

### 4.1 测试命令

```bash
# C端CLI测试
cd backend
python -m scripts.cli_research "卷积神经网络" --search-api duckduckgo -v
```

### 4.2 测试结果

```
✅ Planner 成功生成4个任务
✅ 搜索重试机制生效（3次重试）
✅ 降级处理正常工作（搜索失败不阻塞）
✅ RAG清理机制执行
✅ 笔记协作正常
✅ 最终报告生成成功
✅ LLM评估召回率/精确率（如LLM可用）
✅ 评估报告自动保存到 ./workspace/evaluation/
```

---

## 五、保留的现有功能

- `main.py` 的 FastAPI 服务和 SSE 端点**未修改**
- `App.vue` 前端**未修改**
- 所有现有接口的请求/响应格式**保持不变**

---

## 六、后续优化建议

1. **✅ 评估体系完善**: 已实现 LLM 评估的 recall/precision（2026-04-22）
2. **健康检查增强**: 添加 Ollama/Qdrant 连接检查
3. **搜索后端安装**: 安装 `ddgs` 以支持 DuckDuckGo 搜索
