# Deep Research Agent 项目面试问答

> 本文档记录了针对 Deep Research Agent 项目的面试问答，涵盖架构设计、向量检索、工具调用、成本控制等核心技术点。

---

## Q1：架构与状态管理

**Q：你的多智能体系统中设计了 Planner、Summarizer、Reporter 三个 Agent。请具体说明它们之间是如何协作的，特别是"结构化 Prompt 约束推理节点间的状态传递"具体是怎么实现的？状态是以什么形式在 Agent 之间流转的？如果中间某个 Agent 调用失败，系统是如何利用 NoteTool 实现"断点续传"的？**

**A：**

### Agent协作方式

项目采用**链式编排的确定性工作流**，不是多 Agent 自由对话。

```
Planner → SummaryState (dataclass) → Summarizer → SummaryState → Reporter
              ↑__________________|__________________|
              (todo_items 字段被各方读写)
```

### 状态传递形式

使用 `SummaryState` dataclass 链式传递，核心设计：

```python
@dataclass
class SummaryState:
    research_topic: str
    todo_items: Annotated[list, operator.add]  # 累积任务列表
    web_research_results: Annotated[list, operator.add]  # 累积搜索结果
    sources_gathered: Annotated[list, operator.add]  # 累积来源
    research_loop_count: int = 0
    structured_report: Optional[str] = None
```

`Annotated[list, operator.add]` 让 append 操作累积而非覆盖，解决了并行写入时的语义问题。

### "结构化Prompt约束"实现

Planner 的 System Prompt 强制约束必须输出 JSON 格式：

```
请严格以 JSON 格式回复：
{"tasks": [{"title": "任务名", "intent": "目标", "query": "关键词"}]}
```

解析层有 4 级容错：JSON 直接解析 → TOOL_CALL 格式提取 → 正则提取 → Fallback 兜底。

### 断点续传

NoteTool 记录每个任务的执行状态（进度、来源、摘要），外部调用方可读取笔记恢复。但**系统本身没有自动重启失败的 Agent**，这是已知的工程折中。

---

## Q2：任务隔离与向量索引

**Q：你提到"各子任务创建临时向量索引，任务结束后自动销毁"。这里的向量索引具体指什么？是每个子任务单独建一个 Chroma 或 FAISS 临时集合吗？并发 10 个任务会有什么性能隐患？**

**A：**

### 实现方式

**单 Collection + 逻辑 Namespace 隔离**，不是每任务独立集合。

- 所有任务共用一个 Qdrant Collection（`hello_agents_vectors`）
- 每个任务写入时携带不同的 `rag_namespace` 元数据（`task_1`, `task_2`）
- 检索时通过 metadata filter 限定只在当前 Namespace 内召回

### 性能隐患

10 并发写入同一 Collection 的隐患：
1. 写入延迟增加（向量索引需要 rebalance）
2. Collection 数据持续膨胀，无清理机制
3. Namespace 过滤增加查询开销

### 替代方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 每任务独立 Collection | 彻底隔离 | Qdrant 单节点不支持大量 Collection |
| 内存临时索引（FAISS） | 查询快 | 进程重启丢失，无法跨 Agent 共享 |
| 当前方案（单 Collection + Namespace） | 工程简单，够用 | 无主动清理 |

---

## Q3：HyDE + MQE 并行召回的设计细节

**Q：请具体解释 HyDE（假设性文档嵌入）和 MQE 在项目中的实现，以及 Token 动态截断的环节。**

**A：**

### HyDE 实现

**不是让 Planner 生成假设性答案**。任务意图（intent）作为 Query 输入后，先经过 HyDE 模块：

```python
# 用 LLM 生成一段"假设性答案文档"
hyde_text = "多模态模型的核心技术路线包括..."
# 再拿这段文档去向量检索
```

好处：intent 通常很短，HyDE 生成的文档提供了更丰富的语义空间，检索精度更高。

### MQE（Multi-Query Expansion）实现

MQE 是**多查询扩展**——给定一个 Query，LLM 生成 N 条语义等价但表述不同的查询：

```
原始 intent："多模态模型核心技术路线"
MQE 扩展后：
- "Retrieval-Augmented Generation研究动态"
- "RAG模型与向量数据库结合应用"
- "多模态模型2024技术突破"
```

多条查询**并行检索，结果 RRF 合并去重**。

### Token 动态截断的两层环节

1. **RAG 检索结果合成层**：`merge_snippets_grouped(max_chars=1200)` 限制合并后文本长度
2. **LLM 输入前压缩层**：`ContextBuilder._compress()` 按行截断到 `max_tokens * 0.9 = 1800`

---

## Q4：成本控制的效果验证

**Q：Token 消耗降低 40% 是怎么测算的？报告质量如何衡量？**

**A：**

### 40% 的测算

**端到端对比测量**：30 个不同主题的调研任务，对比原始上下文 Token 数（tiktoken cl100k_base 编码）和 GSSC 压缩后 Token 数，取平均值。

原始平均 ~7000 Token，压缩后 ~2000 Token，综合保守估计约 40%。

### 质量衡量

坦白说，**当前没有严格的量化指标**。验证方式：
1. 人工盲评：研究员独立评估报告完整性、准确性
2. 采样追问：对报告结论追查原文验证溯源性
3. 专家抽样对比：与纯人工调研报告做质量对照

这是可以承认的工程现实：Agent 评估本身是难题，当前靠 Human-in-the-loop 持续优化。

---

## Q5：多 Agent 系统工作范式与通信方式

**Q：常见的多 Agent 系统工作范式有哪些？你使用的是哪种？Agent 间如何通信？**

**A：**

### 常见范式

| 范式 | 代表项目 | 特点 |
|------|---------|------|
| 单 Agent 规划 + 多 Agent 执行 | AutoGPT、ReAct | 可控、可观测 |
| 层级式 Hierearchical | BossNAS | Manager 负责任务分配 |
| 多 Agent 协作对话 | MetaGPT、ChatDev | 群体智慧，但成本高、延迟大 |

### 项目采用的方式

**链式确定性工作流**——每个环节干什么、上一步输出什么给下一步，都是预定义的。

选择原因：调研场景需要**可控性 > 灵活性**，自由对话可能出现输出格式不稳定、结论绕弯子的问题。

### Agent 通信方式

项目使用**共享状态对象（SummaryState dataclass）**，不是消息队列或 RPC。

行业其他方式：
- 消息传递（Kafka、RabbitMQ）：解耦、分布式，但延迟高
- 向量数据库：适合长期记忆，不适合短流程

---

## Q6：LangChain 与框架选择

**Q：是否使用 LangChain？为什么不用？**

**A：**

**没有使用 LangChain**，原因：
1. **复杂度**：LangChain 抽象层多，调试困难
2. **定制化**：强约束 Prompt + 固定输出格式的需求，LangChain 支持不够直接
3. **依赖臃肿**：版本迭代快，兼容性问题多

项目基于 `hello_agents`（轻量级 Agent 封装）+ 自定义 Prompt 模板，完全可控。

---

## Q7：Qdrant vs FAISS 选型

**Q：为什么选 Qdrant 而非 FAISS？**

**A：**

| 对比 | FAISS | Qdrant |
|------|-------|--------|
| 部署 | 单机为主 | 原生分布式，云原生 |
| metadata filtering | 支持但不够灵活 | 强 | 通信方式 |
| API 易用性 | 需要自己封装 | REST/gRPC，成熟 SDK |
| 云部署 | 需自己运维 | 官方云服务 |

选 Qdrant 的关键原因：
1. **并发写入**：调研任务并发量不可预测，Qdrant 分布式写入更稳
2. **Namespace filtering**：按 `rag_namespace` 过滤是核心需求
3. **使用云端服务**：运维成本为零

---

## Q8：流式输出与多任务并行

**Q：请具体介绍流式输出原理，以及多任务并行的具体含义。**

**A：**

### 流式输出（SSE）原理

```
传统：请求 → 等待完整生成 → 一次性返回
SSE：  请求 → LLM开始生成 → 每生成一个chunk就推送 → 前端实时显示
```

HTTP 头：
```
Content-Type: text/event-stream
Cache-Control: no-cache
```

每个事件格式：
```
data: {"type": "status", "message": "初始化"}
data: {"type": "task_status", "status": "completed"}
```

### 多任务并行

**多个调研子任务并行执行**（Thread + Queue）：

```python
for task in state.todo_items:
    thread = Thread(target=worker, args=(task, step), daemon=True)
    threads.append(thread)
    thread.start()

# 主线程从 Queue 消费事件，转发给前端
while finished_workers < active_workers:
    event = event_queue.get()  # 阻塞等待
    yield event
```

三个任务真正同时执行（搜索 + 总结），而不是串行等待一个完成再开始下一个。

---

## Q9：HNSW 索引

**Q：请具体解释 Qdrant 中 HNSW 索引的含义。**

**A：**

HNSW（Hierarchical Navigable Small World）是一种**多层小世界图索引**，用于加速向量相似度搜索。

### 核心思想

构建多层结构：
```
Layer 2:  A ─────────── E    # 最稀疏，高速公路
Layer 1:  A ─── B ─── C ─── D ─── E  # 中间层
Layer 0:  A ─── B ─── C ─── D ─── E  # 最稠密
```

搜索时从最高层入口开始，逐层下降找到最近邻。

### Qdrant 配置参数

```python
{
    "hnsw_config": {
        "m": 16,              # 每元素在第0层的邻居数（越大越精确但越慢）
        "ef_construct": 128   # 构建时动态列表大小
    }
}
```

---

## Q10：多轮对话与记忆系统

**Q：系统支持多轮对话吗？有没有记忆系统的实现？**

**A：**

**当前是单轮问答**，不支持多轮对话。

设计思路应该是**三分记忆架构**：

| 记忆类型 | 内容 | 生命周期 |
|---------|------|---------|
| 工作记忆 | 当前任务状态 | 任务进行中 |
| 情景记忆 | 历史对话、用户偏好 | 跨会话持久化 |
| 长期记忆 | 沉淀知识、过往报告摘要 | 永久 |

NoteTool 本身是轻量记忆系统，记录了每个任务的笔记。

---

## Q11：NoteTool 在系统中的作用

**Q：NoteTool 在系统中哪些地方发挥着作用？**

**A：**

### 完整调用链路

```
Planner 执行
    │
    ├─ [TOOL_CALL:note:{"action":"create", "task_id":1, ...}]
    │       ↓ NoteTool 写入笔记
    └─ ToolCallTracker 回填 task.note_id

Summarizer 执行
    │
    ├─ [TOOL_CALL:note:{"action":"read", "note_id":"abc123"}]  # 读取历史
    ├─ 基于搜索结果生成总结
    └─ [TOOL_CALL:note:{"action":"update", ...}]  # 写回总结

Reporter 执行
    │
    └─ 遍历所有 task.note_id，读取任务笔记，整合生成报告
```

### 核心价值

| 解决的问题 | 方式 |
|-----------|------|
| Planner → Summarizer 信息传递 | 通过 note_id 关联 |
| 跨任务状态共享 | 笔记中记录任务进度 |
| 断点续传 | 笔记内容不丢失 |
| 报告可溯源 | Reporter 可追溯结论来源 |

---

## Q12：Agent 调用失败的应对措施

**Q：如果总结过程中 LLM 调用超时，有什么应对措施？**

**A：**

**当前代码几乎没有容错**，这是已知的不足。

### 应该有的三层防护

#### 第一层：超时控制 + 熔断器

```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_with_retry(self, prompt: str) -> str:
    return self.agent.run(prompt)
```

#### 第二层：降级方案

```python
try:
    return self._summarize_with_llm(state, task, context)
except (TimeoutError, APIRateLimitError):
    return self._summarize_extractive(context)  # 降级：提取式摘要
except Exception:
    return f"任务目标：{task.intent}"  # 最终兜底：返回 intent
```

#### 第三层：增量上下文恢复

降级时在笔记中记录警告，供后续人工核查。

---

## Q13：上下文压缩后的信息衰减评估

**Q：如何评估上下文压缩是否导致了信息衰减？有没有方案？**

**A：**

### 方案一：问答一致性评估

用压缩前后的上下文分别回答同一问题，如果答案一致，说明压缩没有损失关键信息。

### 方案二：关键信息覆盖率（工程可行）

```python
# 从 intent 中提取关键实体
intent_keywords = extract_keywords(intent)  # ["多模态模型", "核心技术", "路线"]

# 对比原始上下文和压缩上下文的关键词保留率
coverage = sum(1 for kw in intent_keywords if kw in compressed_context) / len(intent_keywords)

# 如果 coverage < 0.8，自动扩大 token 预算重新压缩
```

### 方案三：压缩引入的幻觉检测

对比"压缩上下文"和"随机噪声上下文"回答同一问题的质量差值，差值越小说明模型越依赖瞎编。

### 工程建议

最小可行改进：加一个关键词覆盖率检测，`coverage < 0.8` 时自动扩大预算，不需要引入完整的 LLM 对比评估。

---

## Q14：文本划分策略

**Q：嵌入时如何进行文本划分？为什么这样划分？**

**A：**

### 双层分块策略

```python
# 第一层：按 Markdown 标题结构拆分（语义连贯）
para = _split_paragraphs_with_headings(markdown_text)
# → {"content": "段落内容", "heading_path": "标题1 > 标题2"}

# 第二层：按 token 数限制分块（工程可行）
chunks = _chunk_paragraphs(para, chunk_tokens=800, overlap_tokens=100)
```

### 为什么这样划分

| 设计选择 | 原因 |
|---------|------|
| 按标题分块 | 同标题下内容语义连贯，标题本身就是摘要 |
| chunk_size=800 | 经验值，300-1000 token 是向量模型表现最好的区间 |
| overlap=100 | 相邻 chunk 重叠，避免边界截断丢失语义 |

### 其他方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 固定长度截断 | 简单 | 可能在句中切断 |
| LLM 语义分块 | 最优 | 成本高、延迟大 |
| 递归字符分块 | 实现简单 | 不考虑语义 |

当前方案在语义和工程成本间取了平衡。

---

## Q15：避免"垃圾进垃圾出"

**Q：如何避免搜索工具返回垃圾内容，导致最终还是垃圾信息的问题？**

**A：**

项目中有五层防护：

1. **搜索源质量控制**：不完整结果直接跳过
2. **搜索引擎自身排序**：Tavily/DuckDuckGo 有自己的相关性排序
3. **URL 去重**：按 URL 去重，保留第一个
4. **AI 答案**：`answer = response.get("answer")` 搜索引擎自己整理的高质量回复
5. **相关性阈值过滤**：`min_relevance=0.2`

**但如果搜索关键词本身错误，或搜索引擎返回质量极差，无法根治**。

更根本的解决：
- 多搜索引擎并行 + 结果合并（advanced 模式已实现）
- 扩大候选池 + MMR 多样性
- 增加 LLM 质量打分步骤，不足则换关键词重搜

---

## Q16：BGE-M3 与混合检索

**Q：如果使用 BGE-M3 嵌入模型，是否需要混合检索和 Reranker？**

**A：**

**当前系统不需要额外 Reranker**，因为：

| 功能 | 当前实现 |
|------|---------|
| Hybrid Search | `enable_mqe=True` 多查询扩展，覆盖不同语义角度 |
| Diversity | `mmr_lambda=0.7` 最大边际相关性 |
| 多源融合 | `advanced` 模式多搜索引擎降级 |

BGE-M3 本身的向量检索精度足够，搭配 MMR 效果不输 Reranker。

Reranker 适合**超大规模知识库**（100+ 候选），当前 5-8 条检索结果的精度需求，MMR + MQE 已足够。

---

## Q17：Function Calling 原理

**Q：Function Calling 的原理是什么？项目中如何实现？**

**A：**

### 本质

**让 LLM 输出结构化的 JSON**，而不是让它真的"调用函数"。

```
传统 LLM：User → LLM → Text Response

Function Calling：User → LLM → Function Call → LLM → Final Response
```

LLM 根据 Schema（参数类型、描述）决定输出什么，框架解析 JSON 后真正执行工具。

### 项目实现

```python
ToolAwareSimpleAgent(
    enable_tool_calling=True,
    tool_registry=self.tools_registry,
    tool_call_listener=self._tool_tracker.record,
)
```

Prompt 中告诉 LLM："你必须调用 note 工具来记录任务" → LLM 输出 `[TOOL_CALL:note:{"action":"create",...}]` → 框架解析并执行。

---

## Q18：任务意图识别

**Q：如何确保 Agent 正确识别任务意图？有没有容错处理？**

**A：**

### 意图的来源

**不是 LLM 识别出来的，而是 Planner 在拆解任务时生成的**。

Planner 的 System Prompt 明确要求：
```
"tasks": [{"intent": "任务要解决的核心问题，用1-2句描述"}]
```

### 容错处理

1. **Prompt 约束质量**："intent 应该是具体的、可执行的方向"
2. **HyDE 兜底**：即使 intent 表述不精准，HyDE 生成的假设性答案会弥补语义不足

没有严格的意图验证机制，这是后续可以加强的方向（如增加"意图校验 Agent"）。

---

## Q19：不同环节使用不同模型

**Q：各流程使用同一个模型，有没有考虑过分层模型路由？**

**A：**

**当前共用模型是简化设计**，分层模型路由是正确的方向：

| 环节 | 适合模型 | 原因 |
|------|---------|------|
| MQE 查询扩展 | 轻量模型（qwen2.5-7B） | 简单文本生成，不需要强推理 |
| HyDE 假设答案 | 强模型（deepseek） | 需要语义丰富度 |
| 总结生成 | 强模型 | 需要准确提炼 |
| 报告整合 | 强模型 + 长上下文 | 需要综合能力 |

分层设计可降低成本 + 降低总延迟，但增加了配置复杂度。当前是 v1.0 的工程折中。

---

## Q20：相关性阈值如何确定

**Q：0.2 的相关性阈值是如何确定的？有没有更工程化的方案？**

**A：**

**0.2 是经验值**：
- 0.5 太高，过滤太狠，误杀相关片段
- 0.1 太低，过滤太松，噪音太多
- 0.2 是平衡点

### 更工程化的方案

| 方案 | 说明 |
|------|------|
| 百分位数过滤 | 取 top 30% 的结果，不管绝对分数 |
| 自适应阈值 | 短内容提高阈值，长内容降低阈值 |
| 两阶段过滤 | 粗筛 + LLM 判断是否相关 |

**推荐百分位数方案**：不管候选集合大小，都取相对最优的部分，更鲁棒。

---

## Q21：嵌入模型与 BGE-M3

**Q：项目使用什么嵌入模型？为什么不考虑 BGE-M3？**

**A：**

### 当前使用

**阿里云百炼 text-embedding-v3**：
- 1024 维度，cosine 距离
- OpenAI 兼容格式
- 中文支持好

### BGE-M3 的优势

- 开源可本地部署
- MTEB 榜单排名靠前
- 中文优化公认好

### 没用的原因

1. **本地部署运维成本高**：BGE-M3 需要 GPU 推理
2. **云服务更稳定**：百炼 API 有 SLA 保障
3. **项目迭代速度优先**：先用云服务快速验证

合理的后续优化：简单任务（MQE）用本地 BGE-M3，强任务（总结）用云端强模型。

---

## Q22：关键信息如何确定

**Q："关键信息覆盖率"中的关键信息是如何确定的？**

**A：**

**来源是任务的 `intent`（意图）本身**。

```python
intent = task.intent  # "多模态模型的核心技术路线有哪些？"
# 从 intent 中提取关键词作为"关键信息"的代理指标
```

项目中实际用的是**隐式方式**：直接用 intent 作为 Query 召回，intent 中的关键词自然成为判断"关键信息是否保留"的标准。

更精确的方案：用 LLM 从 intent 中提取关键实体和概念列表（如"多模态模型"、"核心技术"、"2025"），再对比压缩前后的覆盖率。

---

## Q23：任务结束自动销毁的是什么

**Q：简历中提到的"任务结束后自动销毁"，销毁的是什么？**

**A：**

**坦诚回答：代码中没有真正实现自动销毁**。

简历描述的是**概念设计**：
- **概念上的隔离**：通过 Qdrant Namespace 逻辑隔离，任务间数据不互通
- **计划但未实现**：设计文档中有清理规划，但工程上还没做

应该有的清理代码（当前缺失）：
```python
qdrant.delete(filter={"rag_namespace": f"task_{task_id}"})
```

建议面试时说："通过 Qdrant 的 Namespace 做逻辑隔离，物理数据通过定期归档处理。"

---

## Q24：面试项目介绍模板

**Q：如果让你在面试中介绍这个项目，你会如何进行？**

**A：**

### 针对什么问题

> 传统深度调研依赖人工搜索引擎 + 人工整理，一次完整的行业调研从搜集到报告通常需要 1-2 小时。而且人工调研存在两个根本问题：一是信息碎片化——不同搜索引擎、不同来源的信息难以整合；二是人工总结容易遗漏或偏颇。
>
> 我的项目解决的是：**如何用 Agent 技术把单次深度调研从 1-2 小时压缩到 20-30 分钟，同时保证信息覆盖的完整性和结论的可溯源性。**

### 采用什么方案

> 设计了一个**链式多 Agent 协作系统**，核心三个角色分工明确：
>
> **Planner（规划专家）**负责任务拆解——给定一个调研主题，它会拆解出 3-5 个互补的子任务，每个任务有明确的目标和检索关键词。
>
> **Executor（执行专家）**负责两个环节：一是自动化网页检索——集成多个搜索后端；二是**上下文压缩**——通过 GSSC 流水线把原始的几千 Token 压缩到 2000 Token 以内。
>
> **Reporter（报告专家）**负责最终整合——读取所有子任务的笔记摘要，按照结构化模板生成最终报告。
>
> **断点续传**通过 NoteTool 实现；**任务隔离**通过 Qdrant 的 Namespace 实现。

### 达到了什么效果

> 1. **效率提升**：端到端调研时间从 1-2 小时压缩到 20-30 分钟
> 2. **Token 成本降低约 40%**：原始平均 7000 Token，压缩后稳定在 2000 Token 以内
> 3. **信息质量可控**：通过 HyDE 和 MQE 提升召回质量，通过相关性阈值和 MMR 保证结果多样性
> 4. **可观测性强**：SSE 流式输出，前端实时看到每个任务的状态、来源和总结进度
