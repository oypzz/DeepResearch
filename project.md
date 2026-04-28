# Deep Research Agent 项目文档

## 项目概述

这是一个**深度研究助手(Deep Research Agent)**项目，基于多Agent协作架构实现自动化网络研究和报告生成。用户输入研究主题后，系统会自动规划任务、执行搜索、总结信息并生成结构化报告。

---

## 技术栈

### 前端
| 技术 | 用途 |
|------|------|
| **Vue 3** | 前端框架（Composition API + `<script setup>`） |
| **TypeScript** | 类型安全 |
| **Vite** | 构建工具和开发服务器 |
| **Axios** | HTTP客户端（已引入但主要用fetch） |
| **纯CSS** | 样式与动画（极光背景、呼吸灯动画、进度条等） |

### 后端
| 技术 | 用途 |
|------|------|
| **Python 3.10+** | 后端语言 |
| **FastAPI** | Web框架，提供REST API和SSE流式响应 |
| **HelloAgents** | Agent框架核心，封装LLM调用和工具调用 |
| **Tavily Python** | 商业搜索API |
| **DuckDuckGo (ddgs)** | 开源搜索后端 |
| **Pydantic** | 配置模型和数据验证 |
| **Loguru** | 日志管理 |
| **Uvicorn** | ASGI服务器 |
| **OpenAI SDK** | LLM调用（兼容OpenAI格式） |

### 支持的LLM提供商
- **Ollama**（本地模型，base_url: localhost:11434）
- **LMStudio**（本地模型，base_url: localhost:1234）
- **ModelScope**（云端模型）
- **其他OpenAI兼容API**

### 支持的搜索后端
- `duckduckgo`（默认）
- `tavily`
- `perplexity`
- `searxng`
- `advanced`（混合模式）

### 向量存储与知识库
- **Qdrant**: 云端向量数据库（`QDRANT_URL`配置）
  - 集合: `hello_agents_vectors`
  - 向量维度: 1024（`QDRANT_VECTOR_SIZE`）
  - 距离度量: cosine（`QDRANT_DISTANCE`）
- **Neo4j**: 知识图谱存储（`NEO4J_URI`配置）
  - 用于存储文档 Chunk 之间的关系（`HAS_CHUNK`）

### Embedding模型
- 默认使用DashScope阿里云embedding（`EMBED_API_KEY`配置）
- 支持OpenAI兼容格式的Base URL配置

---

## 核心架构：多Agent协作

项目采用**三个专业Agent**协作的架构：

```
用户输入主题
     │
     ▼
┌─────────────────┐
│  规划专家Agent   │  ← 使用 TodoPlanner System Prompt
│ (研究规划专家)   │     负责任务拆分和规划
└────────┬────────┘
         │ 生成 TODO List (3-5个任务)
         ▼
┌─────────────────┐
│  执行每个任务    │  ← 搜索 + 总结（多线程并行）
│  (多线程并行)    │
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────────────────────────────┐
│ 搜索服务 │ │        总结专家Agent               │
│ (Search │ │  ┌────────────────────────────┐   │
│ Tool)   │ │  │ 1. 原始Context灌入Qdrant   │   │
└────────┘ │  │ 2. _optimize_context()     │   │
           │  │    (GSSC语义压缩)           │   │
           │  │ 3. Token硬限制2000         │   │
           │  └────────────────────────────┘   │
           │  (任务总结专家)                    │
           └────────┬──────────────────────────┘
                    │
                    ▼
         ┌─────────────────┐
         │  报告撰写专家     │  ← ReportWriter System Prompt
         │ (分析报告撰写者)  │     整合所有任务总结生成最终报告
         └────────┬────────┘
                  │
                  ▼
           最终研究报告
```

### 关键架构特点

1. **子任务命名空间隔离**: 每个任务创建独立的`rag_namespace = task_{id}`，向量数据不污染
2. **GSSC流水线**: Gather(收集) → Select(筛选) → Structure(结构化) → Compress(压缩)
3. **Token硬限制**: ContextConfig限制2000 token，`min_relevance=0.2`过滤低相关噪音
4. **HyDE+MQE**: 在RAG搜索阶段使用假设性文档和多Query扩展提升召回质量

---

## 主要功能模块

### 1. 任务规划服务 (`services/planner.py`)

**核心类**: `PlanningService`

**功能**:
- 调用规划专家Agent将研究主题拆解为3-5个互补的子任务
- 每个任务包含：`title`（标题）、`intent`（目标）、`query`（搜索关键词）
- 解析LLM输出的JSON任务列表
- 支持通过正则提取`[TOOL_CALL:note:...]`格式的工具调用
- 任务去重和fallback机制

**关键代码模式**:
```python
# 使用 hello_agents 的 ToolAwareSimpleAgent
planner_agent = ToolAwareSimpleAgent(
    name="研究规划专家",
    system_prompt=todo_planner_system_prompt.strip(),
    llm=self.llm,
    tool_registry=self.tools_registry,
)
response = planner_agent.run(prompt)
```

### 2. 搜索服务 (`services/search.py`)

**核心函数**: `dispatch_search()`

**功能**:
- 统一的搜索接口，封装多种搜索后端
- 使用`hello_agents.tools.SearchTool`执行搜索
- 支持配置`fetch_full_page`获取完整页面内容
- 返回结构化结果：`results`（来源列表）、`answer`（AI直接答案）、`notices`（警告信息）

**搜索结果格式化**:
```python
# 去重和格式化来源
sources_summary = format_sources(search_result)
context = deduplicate_and_format_sources(search_result, max_tokens_per_source=2000)
```

### 3. 总结服务 (`services/summarizer.py`)

**核心类**: `SummarizationService`

**功能**:
- 调用总结专家Agent对搜索上下文进行深度总结
- 支持流式输出(`stream_task_summary`)
- 自动过滤`<think>...</think>`思考标记
- 移除`[TOOL_CALL:...]`工具调用指令
- 支持笔记协作（读取/更新任务笔记）
- **动态RAG上下文优化**（`_optimize_context`）

**动态RAG流程**:
```python
# 1. 为任务创建独立命名空间
self.rag_tool.rag_namespace = f"task_{task.id}"

# 2. 将长文本灌入向量库
self.rag_tool.execute("add_text", text=raw_context)

# 3. GSSC召回最相关片段（2000 token硬限制）
optimized_context = self.context_builder.build(
    user_query=f"请提取满足以下意图的核心信息：{task.intent}",
)
```

**流式处理**:
```python
# 使用 yield 生成增量输出
for chunk in summary_stream:
    yield {"type": "task_summary_chunk", "content": chunk}
```

### 4. 报告生成服务 (`services/reporter.py`)

**核心类**: `ReportingService`

**功能**:
- 调用报告撰写专家Agent生成结构化报告
- 报告模板包含5个部分：
  1. 背景概览
  2. 核心洞见（3-5条）
  3. 证据与数据
  4. 风险与挑战
  5. 参考来源
- 自动读取任务笔记获取历史数据

### 5. 笔记工具 (`services/notes.py`)

**核心函数**: `build_note_guidance()`

**功能**:
- 生成笔记工具调用指引
- 支持创建(`create`)和更新(`update`)两种操作
- 使用JSON格式传递参数
- 标签系统：`deep_research` + `task_{id}`

**笔记数据结构**:
```json
{
  "action": "create/update",
  "task_id": 1,
  "title": "任务 1: xxx",
  "note_type": "task_state",  // 或 "conclusion"
  "tags": ["deep_research", "task_1"],
  "content": "..."
}
```

### 6. 工具调用追踪 (`services/tool_events.py`)

**核心类**: `ToolCallTracker`

**功能**:
- 记录所有Agent的工具调用事件
- 线程安全（使用`Lock`）
- 支持实时事件推送（event sink模式）
- 解析`note_id`和`task_id`关联

---

## 核心数据结构

### `TodoItem` (任务项)
```python
@dataclass
class TodoItem:
    id: int                    # 任务ID
    title: str                 # 任务标题
    intent: str                # 任务目标
    query: str                 # 搜索关键词
    status: str               # pending/in_progress/completed/skipped
    summary: Optional[str]     # 任务总结
    sources_summary: Optional[str]  # 来源概览
    notices: list[str]         # 系统提示
    note_id: Optional[str]     # 关联笔记ID
    note_path: Optional[str]   # 笔记文件路径
```

### `SummaryState` (研究状态)
```python
@dataclass
class SummaryState:
    research_topic: str         # 研究主题
    web_research_results: list # Web搜索结果
    sources_gathered: list     # 已收集来源
    research_loop_count: int   # 研究循环次数
    todo_items: list          # 任务列表
    structured_report: Optional[str]  # 结构化报告
```

---

## API接口 (`main.py`)

### 1. 健康检查
```
GET /healthz
Response: {"status": "ok"}
```

### 2. 同步研究
```
POST /research
Request:  {"topic": "研究主题", "search_api": "duckduckgo"}
Response: {
  "report_markdown": "# 报告...",
  "todo_items": [...]
}
```

### 3. 流式研究（SSE）
```
POST /research/stream
Request:  {"topic": "研究主题", "search_api": "duckduckgo"}
Response: SSE流
  - status: 状态消息
  - todo_list: 任务清单
  - task_status: 任务状态更新
  - sources: 来源信息
  - task_summary_chunk: 总结片段
  - tool_call: 工具调用记录
  - final_report: 最终报告
  - done: 完成标记
```

---

## 前端架构 (`App.vue`)

### 状态管理
- 使用Vue 3 ` reactive` 和 `ref` 管理表单和状态
- `isExpanded`: 控制初始/全屏两种布局
- `todoTasks`: 任务列表
- `progressLogs`: 进度日志

### 核心功能
1. **表单输入**: 研究主题 + 搜索引擎选择
2. **实时进度**: SSE流接收并解析7种事件类型
3. **任务详情**: 点击任务切换详情展示
4. **笔记复制**: 一键复制笔记路径
5. **取消功能**: AbortController中止请求

### 事件处理
```typescript
runResearchStream(payload, (event) => {
  switch(event.type) {
    case "status":        // 状态消息
    case "todo_list":     // 任务清单
    case "task_status":   // 任务状态
    case "sources":       // 来源更新
    case "task_summary_chunk":  // 总结片段
    case "tool_call":     // 工具调用
    case "final_report":  // 最终报告
    case "error":         // 错误处理
  }
})
```

---

## 工具调用机制

### HelloAgents框架
项目核心依赖`hello-agents`库，封装了：
- `HelloAgentsLLM`: LLM调用（支持tool_calling）
- `ToolAwareSimpleAgent`: 工具感知Agent
- `ToolRegistry`: 工具注册表
- `SearchTool`: 搜索工具
- `NoteTool`: 笔记工具
- `RAGTool`: 检索增强生成工具（支持Qdrant向量存储）
- `ContextBuilder`: GSSC上下文构建流水线
- `ContextConfig`: 上下文配置（Token预算、相关性阈值等）

### 工具调用格式
```
[TOOL_CALL:tool_name:{"action": "xxx", ...}]
```

### Tool Calling vs JSON Mode
```python
# 配置启用tool calling
self.tools_registry: ToolRegistry | None = None
if self.note_tool:
    registry = ToolRegistry()
    registry.register_tool(self.note_tool)
    self.tools_registry = registry

# 创建支持工具调用的Agent
ToolAwareSimpleAgent(
    enable_tool_calling=(self.tools_registry is not None),
    tool_registry=self.tools_registry,
    tool_call_listener=self._tool_tracker.record,
)
```

---

## 配置管理 (`config.py`)

使用Pydantic BaseModel管理配置：
```python
class Configuration(BaseModel):
    max_web_research_loops: int = 3      # 研究深度
    local_llm: str = "llama3.2"          # 本地模型
    llm_provider: str = "modelscope"      # LLM提供商
    search_api: SearchAPI = DUCKDUCKGO   # 搜索引擎
    enable_notes: bool = True            # 启用笔记
    notes_workspace: str = "./notes"      # 笔记目录
    fetch_full_page: bool = True         # 抓取全文
    strip_thinking_tokens: bool = True   # 过滤思考标记
```

支持从环境变量读取配置。

---

## 并发模型

### 多线程任务执行
```python
# 使用 Thread 执行多任务
threads: list[Thread] = []
for task in state.todo_items:
    thread = Thread(target=worker, args=(task, step), daemon=True)
    threads.append(thread)
    thread.start()

# 主线程从 Queue 消费事件
event_queue: Queue[dict[str, Any]] = Queue()
while finished_workers < active_workers:
    event = event_queue.get()
    yield event
```

---

## 项目亮点（面试重点）

### 1. 多Agent协作架构
- 规划、搜索、总结、报告四个阶段职责分离
- 通过System Prompt控制各Agent专业角色
- Agent间通过笔记系统共享状态

### 2. 流式SSE架构
- 后端使用Python Generator实现SSE流
- 前端实时解析7种事件类型
- 支持任务级别的增量更新

### 3. 工具调用系统
- 基于`hello_agents`框架的Tool Calling
- 笔记工具实现Agent记忆持久化
- 支持从工具输出解析`note_id`

### 4. 灵活的搜索后端
- 抽象的搜索接口（`dispatch_search`）
- 支持5种搜索后端热切换
- 混合搜索模式（`backend="hybrid"`）

### 5. 生产级配置管理
- Pydantic模型验证
- 环境变量支持
- 敏感信息脱敏日志

### 6. 线程安全的工具追踪
- `ToolCallTracker`使用Lock保护
- Event Sink模式实现实时推送
- 支持从tag解析task_id

### 7. 动态RAG上下文优化（`_optimize_context`）
- **核心位置**: `services/summarizer.py` 第45-72行
- **功能**: 对原始搜索结果进行语义压缩，剔除低相关性噪音
- **流程**:
  1. 为每个任务创建独立的RAG命名空间（`task_{id}`），避免跨任务污染
  2. 将原始网页长文本灌入Qdrant向量数据库（自动分块+Embedding）
  3. 使用任务意图(intent)作为Query，召回最相关的片段
  4. GSSC流水线完成语义筛选和压缩

### 8. GSSC上下文构建流水线
- **来源**: `hello_agents/context/builder.py`
- **四阶段**:
  1. **Gather**: 收集候选信息（记忆、RAG、对话历史）
  2. **Select**: 基于余弦相似度计算相关性分数，结合时间衰减因子
  3. **Structure**: 组织成`[Role & Policies] / [Task] / [State] / [Evidence]`结构
  4. **Compress**: 严格控制在Token预算内（默认2000 token）

### 9. HyDE + MQE高级检索（`search_vectors_expanded`）
- **来源**: `hello_agents/memory/rag/pipeline.py` 第741-805行
- **HyDE (Hypothetical Document Embeddings)**:
  - 用LLM生成"假设性答案文档"作为检索Query
  - 捕获查询的语义意图而非字面匹配
- **MQE (Multi-Query Embedding)**:
  - LLM生成n个语义等价查询（如2个）
  - 多个Query并行召回，结果合并去重
- **并行召回+去重**: 扩展候选池（`candidate_pool_multiplier=4`），合并多Query结果

### 10. Token预算动态控制
- **ContextConfig**: `max_tokens=2000`, `reserve_ratio=0.1`
- **相关性阈值**: `min_relevance=0.2` 过滤低质量内容
- **MMR (最大边际相关)**: `mmr_lambda=0.7` 平衡相关性与多样性
- **效果**: 相比原始搜索结果，Token消耗降低约40%

---

## 文件结构
```
backend/
├── src/
│   ├── agent.py          # 核心DeepResearchAgent编排器
│   ├── main.py           # FastAPI入口
│   ├── config.py         # 配置管理
│   ├── models.py         # 数据模型
│   ├── prompts.py        # System Prompt模板
│   ├── utils.py          # 工具函数
│   └── services/
│       ├── planner.py    # 任务规划服务
│       ├── search.py     # 搜索服务
│       ├── summarizer.py # 总结服务
│       ├── reporter.py   # 报告生成服务
│       ├── notes.py      # 笔记协作
│       ├── tool_events.py# 工具追踪
│       └── text_processing.py # 文本处理
frontend/
├── src/
│   ├── App.vue           # 主应用（包含完整UI和逻辑）
│   ├── main.ts           # 入口
│   ├── style.css         # 全局样式
│   └── services/
│       └── api.ts        # SSE流处理
└── package.json
```
