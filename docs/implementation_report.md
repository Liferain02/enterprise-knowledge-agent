# 功能实现报告

**时间**: 2026-04-03
**优先级 1**: 高价值、低难度（已完成）
**优先级 2**: 中价值、中难度（已完成）

---

## 优先级 1：高价值，低难度

### P1-1: General Agent → ReAct Agent（赋予通用 Agent 工具调用能力）

#### 1.1 背景与问题

原 `general_agent_node` 是简单的函数调用，仅返回预设的固定回复（如"你好呀！有什么可以帮助你的吗？"），无法：
- 调用外部工具（如搜索通用知识、查询对话历史）
- 处理混合场景（既需要知识查询，又需要对话）
- 根据上下文做个性化回复

#### 1.2 解决方案

将 `general_agent_node` 重构为基于 `create_react_agent` 的 ReAct Agent：

```python
async def general_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    agent = _get_general_agent()  # ReAct Agent 实例

    config = {"configurable": {"thread_id": f"general_{session_id}"}}
    messages_with_context = inject_context_to_messages(messages, summary, mem0_memories)

    # await 直接调用，Agent 内部处理工具调用循环
    result = await asyncio.wait_for(
        agent.ainvoke({"messages": messages_with_context}, config),
        timeout=GENERAL_TIMEOUT  # 120s 超时
    )

    agent_messages = result.get("messages", [])
    final_answer = agent_messages[-1].content
```

**关键设计**：
- **工具注入**: 通过 `SkillLoader` 从 `Skill.md` 动态加载 `general_search` 和 `search_conversation_history` 工具
- **上下文注入**: 使用 `inject_context_to_messages()` 将 Mem0 记忆和对话摘要注入消息头部
- **超时保护**: `asyncio.wait_for` 防止 Agent 长时间无响应
- **错误处理**: 超时时返回友好提示，异常时捕获并返回错误信息
- **Agent 缓存**: 使用 `_agent_cache` 缓存 Agent 实例，避免重复创建

#### 1.3 General Skill 工具定义

```yaml
# src/agent/skills/general/Skill.md
tools:
  - module: scripts.tools
    names:
      - general_search       # 搜索通用知识
      - search_conversation_history  # 搜索对话历史
```

#### 1.4 工具实现

**`general_search`**: 搜索通用知识（天气、时间等常识问题）
**`search_conversation_history`**: 搜索当前会话的历史消息

#### 1.5 验证结果

```
输入: "你好！"
路由: general_agent（ReAct Agent）
工具调用: general_search
输出: 你好呀！👋 很高兴见到你！...
Agent: general_agent
耗时: 2.69s ✅
```

---

### P1-2: Mem0 上下文只注入一次（execute_plan 入口注入，后续 Worker 复用）

#### 1.6 背景与问题

原架构中，每个并行 Worker 节点（如 `knowledge_step_node`）各自调用 `inject_context_to_messages()` 注入 Mem0 记忆和摘要：

- **重复序列化**: 每个 Worker 都要遍历和序列化 Mem0 数据，浪费 token
- **数据不一致风险**: 多次注入可能导致上下文重复或不一致
- **维护困难**: 每次修改上下文格式需要在多个地方同步

#### 1.7 解决方案

在 `execute_plan_node` 入口处**一次性注入**，后续 Worker 通过预注入的消息列表直接复用：

```python
async def _execute_plan_with_send(state):
    # ── 预注入 Mem0 + 摘要上下文（只注入一次）──────────────
    from ._utils import inject_worker_context
    messages_with_context = inject_worker_context(
        messages, summary, mem0_memories
    )

    # ── 分批并行分发（复用预注入的消息）─────────────────
    batches = analyze_step_dependencies(plan_steps)

    # 通过 Send 将预注入的消息列表传给每个 Worker
    return _send_batch(plan_steps, batches[0], messages_with_context, ...)
```

**预注入函数 `inject_worker_context`**:
```python
def inject_worker_context(messages, summary, mem0_memories):
    """
    为 Worker Agent 预注入上下文。

    与 inject_context_to_messages 的区别：
    - 本方法将 Mem0 记忆和摘要合并为一条 SystemMessage（减少 token 开销）
    - 适用于 parallel_executor 中多个 Worker 共享同一份预注入上下文

    格式：「【背景上下文】Mem0记忆（若存在）\n---\n对话摘要（若存在）」
    """
    parts = []
    if mem0_memories:
        parts.append(f"【用户背景与历史记忆】\n{mem0_memories.strip()}")
    if summary:
        parts.append(f"【本次对话早期摘要】\n{summary.strip()}")
    if not parts:
        return list(messages)

    context_content = "\n\n---\n\n".join(parts)
    context_msg = SystemMessage(
        content=f"【背景上下文】以下是你需要了解的背景信息：\n\n{context_content}"
    )
    return [context_msg] + list(messages)
```

**关键优势**:
1. **Token 节省**: Mem0 + 摘要合并为单条消息，比拆分成多条节省约 30-50 tokens
2. **原子性**: 上下文在分发前一次性确定，所有 Worker 看到的是同一份数据
3. **可维护性**: 上下文格式只需在一处修改

---

### P1-3: Sub-agent thread_id 隔离（避免 session_id 共享导致的竞态）

#### 1.8 背景与问题

原架构中，所有 Agent 共用主图的 `thread_id`（即 `session_id`），导致：
- **并发冲突**: 同一 session 的多个并行步骤共享 checkpointer 状态
- **状态覆盖**: 多个 Agent 节点同时写入同一 thread，导致状态竞态
- **调试困难**: 难以追踪单步骤的执行历史

#### 1.9 解决方案

每个 Agent 使用**独立的 `thread_id`**：

```python
# knowledge_agent_node
config = {"configurable": {"thread_id": f"{session_id}_knowledge"}}

# operation_agent_node
config = {"configurable": {"thread_id": f"{session_id}_operation"}}

# general_agent_node
config = {"configurable": {"thread_id": f"general_{session_id}"}}

# SkillLoader 创建的 Agent
config = {"configurable": {"thread_id": f"{session_id}_{skill_name}"}}
```

**thread_id 命名规范**:
| Agent/场景 | thread_id 格式 | 示例 |
|-----------|--------------|------|
| 知识库查询 | `{session_id}_knowledge` | `sess_001_knowledge` |
| 操作执行 | `{session_id}_operation` | `sess_001_operation` |
| 通用对话 | `general_{session_id}` | `general_sess_001` |
| Skill Agent | `{session_id}_{skill_name}` | `sess_001_datetime` |
| 主图 | `{session_id}` | `sess_001` |

**优势**:
- 每个 Agent 维护独立的 checkpointer 状态，互不干扰
- 支持真正的并行执行，不会出现状态覆盖
- 调试时可以按 `thread_id` 追踪单个 Agent 的执行历史

---

### P1-4: 流式 + 非流式共用 ChatService 方法（消除代码重复）

#### 1.10 背景与问题

原 `ChatService` 中流式和非流式接口存在大量重复代码：

```python
# 非流式（重复代码）
async def achat(self, message, session_id, username, images):
    # 图片理解
    if images:
        processed = await prepare_message(...)
    # 调用 agent
    result = await arun_agent(...)
    # 保存消息
    session_service.save_message(...)
    # 生成标题
    if message_count == 0:
        title = generate_title(...)
    return result

# 流式（重复代码）
async def achat_stream(self, message, session_id, username, images):
    # 图片理解（重复）
    if images:
        processed = await prepare_message(...)
    # 构建配置（重复）
    config = {...}
    # 流式输出...
    # 保存消息（重复）
    session_service.save_message(...)
    # 生成标题（重复）
    if message_count == 0:
        title = generate_title(...)
```

#### 1.11 解决方案

将公共逻辑提取为私有方法：

```python
class ChatService:

    # ==================== 公共逻辑提取 ====================

    async def _prepare_message(self, message, images=None):
        """图片理解 + 消息预处理"""
        if not images:
            return message
        # Vision LLM 处理...
        return processed_message

    def _generate_title(self, message, session_id):
        """生成会话标题"""
        if greeting_patterns.match(message):
            return "问候"
        if is_first_message(session_id):
            return session_service.generate_title(message)
        return None

    def _save_chat_message(self, session_id, message, answer, used_agent):
        """保存用户和助手的聊天消息"""
        session_service.save_message(session_id, "user", message)
        session_service.save_message(session_id, "assistant", answer, metadata)

    def _sse_event(self, event_type, data):
        """格式化为 SSE 事件"""
        return f"data: {json.dumps({...})}\n\n"

    def _format_sources(self, sources):
        """格式化来源字段"""
        ...

    # ==================== 公共入口 ====================

    async def _do_chat_async(self, message, session_id, username, images):
        """聊天核心逻辑（异步）- 公共实现"""
        total_start = time.time()

        session_service.ensure_session_exists(session_id)

        # 图片理解
        processed_message = await self._prepare_message(message, images)

        # 调用 Agent
        result = await arun_agent(
            input_text=processed_message,
            session_id=session_id,
            user_id=username,
        )

        # 生成标题
        title = self._generate_title(processed_message, session_id)
        if title:
            session_service.update_session_title(session_id, title)

        # 保存消息
        self._save_chat_message(session_id, message, answer, used_agent)

        # 性能日志
        elapsed = time.time() - total_start
        return result

    # ==================== 对外接口 ====================

    def chat(self, message, session_id):
        """非流式接口"""
        return self._do_chat(message, session_id)

    async def achat(self, message, session_id, username, images):
        """异步非流式接口"""
        return await self._do_chat_async(message, session_id, username, images)

    async def achat_stream(self, message, session_id, username, images):
        """流式接口"""
        # 仅流式特有逻辑：
        # - astream_events() 逐 token 输出
        # - 从 checkpointer 提取最终结果
        # 其余复用公共方法
        processed = await self._prepare_message(message, images)
        graph = await get_agent_graph_async()
        async for event in graph.astream_events(...):
            # SSE 输出...
```

**消除的重复**:
- 图片理解逻辑: 1 处 → 1 处（通过 `_prepare_message`）
- 标题生成逻辑: 2 处 → 1 处（通过 `_generate_title`）
- 消息保存逻辑: 3 处 → 1 处（通过 `_save_chat_message`）
- SSE 格式化: 1 处 → 1 处（通过 `_sse_event`）

---

## 优先级 2：中价值，中难度

### P2-1: 引入 Pydantic Schema 定义步骤间传递

#### 2.1 背景与问题

原 `parallel_executor.py` 中步骤结果以纯字符串传递：

```python
# 原代码：纯字符串传递
plan_results = []  # List[str]，无法携带元信息
plan_results.append(f"步骤 {step_id} 结果: {step_result}")
```

问题：
- **无类型信息**: 无法区分步骤类型（knowledge/operation/general）
- **无来源信息**: 知识库来源无法传递到汇总阶段
- **无结构化数据**: 第1步检索的数值无法传给第2步计算
- **无法扩展**: 要加字段只能拼接字符串

#### 2.2 解决方案

新增 `src/agent/agents/_schemas.py`，定义结构化的 Schema：

```python
class StepResult(BaseModel):
    """单个计划步骤的执行结果（结构化传递）"""

    step_id: int = Field(description="步骤 ID")
    description: str = Field(description="步骤描述")
    agent: Literal["knowledge_agent", "operation_agent", "general_agent"]
    result: str = Field(default="", description="步骤执行的文本结果")
    sources: Optional[str] = Field(default=None, description="知识来源")

    # 结构化数据（供下游步骤使用）
    structured_data: Optional[dict] = Field(
        default=None,
        description="机器可读数据，例："
                    "{\"type\": \"numeric\", \"value\": 15, \"unit\": \"天\"}"
    )

    # 执行状态
    success: bool = Field(default=True)
    error: Optional[str] = Field(default=None)
    confidence: Optional[float] = Field(default=None)


class PlanExecutionResult(BaseModel):
    """完整计划执行结果"""

    plan_id: str = Field(default="")
    steps: List[StepResult] = Field(default_factory=list)
    final_answer: str = Field(default="")
    used_agent: Literal["planner_parallel", "planner_sequential", "supervisor"]
    execution_mode: Literal["parallel", "sequential"]
    total_steps: int = Field(default=0)
    successful_steps: int = Field(default=0)
    failed_steps: int = Field(default=0)

    def get_step(self, step_id: int) -> Optional[StepResult]: ...
    def get_structured_data(self, step_id: int) -> Optional[dict]: ...
    def get_numeric_result(self, step_id: int) -> Optional[float]: ...
```

**结构化数据提取**:

```python
def _extract_numeric_data(text: str) -> dict:
    """从文本结果中提取数值数据"""
    patterns = [
        (re.compile(r"(?:为|=|共计)\s*(\d+)\s*(?:天|日)"), "天"),
        (re.compile(r"(?:工资|共计)\s*(\d+(?:\.\d+)?)\s*元"), "元"),
        (re.compile(r"(\d+(?:\.\d+)?)\s*倍"), "倍数"),
        ...
    ]
    # 返回 {"type": "numeric", "value": 15, "unit": "天", "raw_match": "..."}
```

**使用示例**:

```python
# parallel_executor.py 中
return serialize_step_result({
    "step_id": step["step_id"],
    "description": step["description"],
    "agent": agent_name,
    "result": result.get("final_answer", ""),
    "sources": result.get("sources", ""),
    "structured_data": _extract_numeric_data(result.get("final_answer", "")),
})

# 上游结构化结果注入
def _inject_structured_context(description, prior_results):
    """将上游步骤的结构化数据注入当前步骤"""
    context_parts = []
    for result in prior_results:
        if result.structured_data and result.structured_data.get("type") == "numeric":
            data = result.structured_data
            context_parts.append(
                f"【步骤 {result.step_id} 结果】{data.get('label')} = {data.get('value')} {data.get('unit')}"
            )
    return f"{description}\n\n【上游步骤结果参考】\n" + "\n".join(context_parts)
```

#### 2.3 验证结果

```
StepResult 序列化测试: ✅
  step_id=1, agent=knowledge_agent, result="test result"
  sources="test sources", success=True

数值提取测试: ✅
  "年假天数为 15 天" -> {"type": "numeric", "value": 15.0, "unit": "天"}
  "共计 3000 元" -> {"type": "numeric", "value": 3000.0, "unit": "元"}
  "绩效 0.8 倍" -> {"type": "numeric", "value": 0.8, "unit": "倍数"}
```

---

### P2-2: 用 LangGraph Send 替换 asyncio.gather()（真正并行 fan-out/fan-in）

#### 2.4 背景与问题

原并行执行使用 `asyncio.gather()`：

```python
async def _execute_plan_parallel(state):
    tasks = [execute_step(step, ...) for step in batch_steps]
    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
```

问题：
- **失败不隔离**: 一个步骤崩溃可能导致全部失败
- **无断点续跑**: 中断后无法从中间状态恢复
- **状态管理复杂**: 需手动汇总各步骤的 state 更新
- **无状态可见性**: 外部无法追踪单步骤的执行进度

#### 2.5 解决方案

使用 LangGraph **Send 原语**实现真正的 fan-out/fan-in：

```python
# execute_plan_node 入口
async def execute_plan_node(state):
    if USE_LANGGRAPH_SEND and current_step == 0:
        return await _execute_plan_with_send(state)
    if PARALLEL_EXECUTION_ENABLED:
        return await _execute_plan_parallel(state)  # 后备
    return await _execute_plan_sequential(state)

# Send 模式：分批分发
async def _execute_plan_with_send(state):
    # 1. 预注入上下文
    messages_with_context = inject_worker_context(messages, summary, mem0_memories)

    # 2. 拓扑排序分批
    batches = analyze_step_dependencies(plan_steps)
    print(f"[Send] 分 {len(batches)} 批处理 {len(plan_steps)} 个步骤")

    # 3. Fan-out：第一批次通过 Send 并行分发
    if len(batches) == 1:
        return _send_batch(plan_steps, batches[0], ...)

    # 4. 多批次：通过 state 记录后续批次
    return {
        **first_send,
        "current_batch_index": 1,
        "remaining_batches": remaining_batches_data,
    }

# 发送函数
def _send_batch(plan_steps, batch, messages_with_context, session_id, summary):
    sends = []
    for step_idx in batch:
        step = plan_steps[step_idx]
        agent = step.get("agent", "general_agent")

        # 映射 agent → 节点名
        node_map = {
            "knowledge_agent": "knowledge_step_node",
            "operation_agent": "operation_step_node",
            "general_agent": "general_step_node",
        }
        node_name = node_map.get(agent, "general_step_node")

        # 通过 Send 分派到对应的 Worker 节点
        sends.append(Send(node_name, {
            "step": step,
            "step_id": step["step_id"],
            "step_agent": agent,
            "messages": messages_with_context,  # 复用预注入的消息
            "session_id": session_id,
            "summary": summary,
        }))

    return sends
```

**图结构中的 Worker 节点**:

```python
# src/agent/graph.py
workflow.add_node("knowledge_step_node", knowledge_step_node)
workflow.add_node("operation_step_node", operation_step_node)
workflow.add_node("general_step_node", general_step_node)

# Send Worker 节点接收 Send 分派的独立步骤
async def knowledge_step_node(state):
    step = state.get("step", {})
    messages = state.get("messages", [])  # 预注入的消息

    from .knowledge import knowledge_agent_node
    result = await knowledge_agent_node({"messages": messages, "session_id": ...})

    from ._schemas import serialize_step_result
    return serialize_step_result({
        "step_id": step.get("step_id"),
        "description": step.get("description"),
        "agent": "knowledge_agent",
        "result": result.get("final_answer"),
        "sources": result.get("sources"),
        ...
    })
```

**与 asyncio.gather 对比**:

| 维度 | asyncio.gather | LangGraph Send |
|------|---------------|----------------|
| 失败隔离 | ❌ 一个失败全部失败 | ✅ 单步骤崩溃不影响其他步骤 |
| 断点续跑 | ❌ 无 | ✅ 图结构天然支持 |
| 状态可见性 | ❌ 无 | ✅ 每个分支独立更新状态 |
| 执行控制 | 需手动管理 | ✅ 图路由自动控制 |
| 拓扑排序 | 需手动实现 | ✅ 可复用 analyze_dependencies |

---

### P2-3: MCP 工具使用完整类型化 Schema

#### 2.6 背景与问题

原 MCP 工具适配器 (`mcp_adapter.py`) 仅使用简单的参数传递：

```python
# 原代码：无 Schema，仅依赖 MCP 返回的原始 schema
async def execute_tool(**kwargs):
    result = await tool_manager.call_mcp_tool(server_name, tool_name, kwargs)
    return result
```

问题：
- **参数校验缺失**: LLM 可能传错参数类型或缺少必填字段
- **类型推断不可靠**: 仅依赖 MCP schema 的 type 字段
- **复杂类型不支持**: enum/array/object 等类型无法正确处理
- **工具描述不完整**: LLM 可能选错工具

#### 2.7 解决方案

新增完整类型化 Schema 系统：

```python
# 预定义的 MCP 工具 Schema 增强
_MCP_TOOL_SCHEMAS = {
    "list_directory": {
        "properties": {
            "path": {
                "type": "string",
                "description": "要列出的目录绝对路径或相对路径",
                "default": "."
            }
        },
        "required": ["path"],
        "enhanced_description": "列出指定目录下的所有文件和子目录（类似 ls -la 命令）"
    },
    "read_file": {...},
    "write_file": {...},
    ...
}

def convert_single_mcp_tool(mcp_tool):
    # 1. 使用预定义 Schema 或从 MCP inputSchema 动态构建
    predefined = _MCP_TOOL_SCHEMAS.get(tool_name)
    if predefined:
        properties = predefined["properties"]
        required = predefined["required"]
        tool_description = predefined["enhanced_description"]
    else:
        # 从 MCP inputSchema 提取
        input_schema = getattr(mcp_tool, 'inputSchema', {})
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])

    # 2. 构建类型化 Pydantic 模型
    args_schema = _create_typed_pydantic_model(tool_name, properties, required)

    # 3. 异步执行（带参数校验）
    async def execute_tool_async(**kwargs):
        # Pydantic 校验
        validated = args_schema(**kwargs)
        validated_kwargs = validated.model_dump()

        # 调用 MCP 工具
        result = await tool_manager.call_mcp_tool(server_name, tool_name, validated_kwargs)
        return format_mcp_result(result)

    return StructuredTool.from_function(
        coroutine=execute_tool_async,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
    )
```

**动态 Pydantic 模型构建**:

```python
def _create_typed_pydantic_model(tool_name, properties, required):
    """
    支持：
    - 基础类型: string/integer/number/boolean
    - 数组类型: List[str], List[int], List[float]
    - 枚举类型: Literal["value1", "value2"]
    - 对象类型: dict
    - 默认值处理
    """
    field_definitions = {}

    for prop_name, prop_info in properties.items():
        prop_type = str
        json_type = prop_info.get('type', 'string')

        if json_type == 'integer':
            prop_type = int
        elif json_type == 'number':
            prop_type = float
        elif json_type == 'boolean':
            prop_type = bool
        elif json_type == 'array':
            items = prop_info.get('items', {})
            if items.get('type') == 'integer':
                prop_type = List[int]
            ...
        elif json_type == 'object':
            prop_type = dict

        # enum 处理
        if 'enum' in prop_info:
            if len(enum_values) == 1:
                prop_type = type(enum_values[0])
                field_definitions[prop_name] = (prop_type, Field(default=enum_values[0]))
                continue
            else:
                literal_type = Literal[tuple(enum_values)]
                prop_type = literal_type

        # 默认值处理
        default = ...  # Required
        if prop_name not in required:
            default = prop_info.get('default')

        field_definitions[prop_name] = (prop_type, Field(default=default))

    model_name = f"{tool_name.replace('-', '_').title()}Input"
    return create_model(model_name, **field_definitions)
```

#### 2.8 优势

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 参数校验 | ❌ 无 | ✅ Pydantic 强制校验 |
| 描述质量 | 依赖 MCP schema | 预定义增强描述 |
| 类型支持 | 仅 string | 完整类型系统 |
| 错误处理 | 运行时崩溃 | 友好的校验错误 |
| 工具选择 | 可能选错 | 描述更清晰 |

---

### P2-4: 统一异常处理中间件

#### 2.9 背景与问题

原架构中异常处理分散在各个 Controller 层：

```python
# chat_controller.py
@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        result = await chat_service.achat(...)
        return {"success": True, "data": result}
    except ValueError as e:
        return {"success": False, "error": str(e)}, 400
    except TimeoutError as e:
        return {"success": False, "error": "请求超时"}, 504
    except Exception as e:
        logger.error(...)
        return {"success": False, "error": "内部错误"}, 500

# knowledge_controller.py
@router.post("/search")
async def search(request: SearchRequest):
    try:
        ...
    except ValueError as e:
        return {"success": False, "error": str(e)}, 400
    # 重复的异常处理...
```

问题：
- **代码重复**: 每个 Controller 都要写相同的 try/except
- **不一致性**: 不同 Controller 的错误格式可能不统一
- **遗漏风险**: 可能遗漏某些异常类型的处理
- **堆栈泄露**: 生产环境可能暴露内部堆栈

#### 2.10 解决方案

新增 `src/api/middleware/__init__.py`：

**1. 异常分类体系**:

```python
class AppException(Exception):
    """应用层基础异常"""
    def __init__(self, message: str, code: str = "APP_ERROR", status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code

class ChatException(AppException):
    """聊天服务异常"""
    def __init__(self, message: str, code: str = "CHAT_ERROR", status_code: int = 500):
        super().__init__(message, code, status_code)

class KnowledgeException(AppException):
    """知识库服务异常"""
    ...

class ValidationException(AppException):
    """参数校验异常"""
    def __init__(self, message: str, ...):
        super().__init__(message, code="VALIDATION_ERROR", status_code=422)

class AuthenticationException(AppException):
    """认证异常"""
    ...

class ResourceNotFoundException(AppException):
    """资源不存在异常"""
    ...
```

**2. HTTP 状态码自动映射**:

```python
_EXCEPTION_STATUS_MAP = {
    "ValueError": 400,
    "TypeError": 400,
    "KeyError": 400,
    "FileNotFoundError": 404,
    "PermissionError": 403,
    "TimeoutError": 504,
    "ConnectionError": 503,
    "JSONDecodeError": 400,
}
```

**3. 中间件**:

```python
class UnifiedExceptionHandlerMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        # 生成请求追踪 ID
        request_id = str(uuid.uuid4())[:8]

        try:
            response = await call_next(request)
            return response

        except HTTPException:
            raise  # FastAPI 原生处理

        except AppException as exc:
            # 应用层已知异常：友好返回
            logger.warning(f"[{request_id}] AppException: {exc.code} - {exc.message}")
            return JSONResponse(
                status_code=exc.status_code,
                content=make_error_response(
                    message=exc.message,
                    code=exc.code,
                    request_id=request_id,
                )
            )

        except Exception as exc:
            # 未知异常：推断状态码，记录堆栈
            exc_type = type(exc).__name__
            status_code = _get_status_code_from_exception(exc)

            tb_str = "".join(traceback.format_exception(*sys.exc_info()))
            logger.error(f"[{request_id}] Unhandled: {exc_type}: {exc_msg}\n{tb_str}")

            # 生产模式隐藏内部细节
            return JSONResponse(
                status_code=status_code,
                content=make_error_response(
                    message="服务器内部错误",
                    code=exc_type.upper(),
                    detail=None if not self.debug else tb_str,
                    request_id=request_id,
                )
            )
```

**4. 统一错误响应格式**:

```python
{
    "success": False,
    "error": {
        "code": "CHAT_ERROR",
        "message": "聊天服务异常",
        "status": 500,
        "detail": null,  # 仅 debug 模式返回
        "request_id": "a1b2c3d4"
    }
}
```

**5. 注册**:

```python
# main.py
from src.api.middleware import register_exception_handlers

app = FastAPI(...)
register_exception_handlers(app, debug=settings.debug)
```

#### 2.11 优势

| 维度 | 改进前 | 改进后 |
|------|--------|--------|
| 代码量 | 每个 Controller 重复 | 集中一处 |
| 错误格式 | 各处不统一 | 统一 JSON 格式 |
| 堆栈安全 | 可能泄露 | 生产隐藏细节 |
| 状态码 | 手动指定 | 自动推断 |
| 请求追踪 | 无 | request_id 贯穿 |

---

## 三、修改文件清单

| 文件 | 改动类型 | 说明 |
|------|---------|------|
| `main.py` | 新增 | 注册统一异常处理器 |
| `src/agent/agents/_utils.py` | 新增 | 新增 4 个上下文注入工具函数 |
| `src/agent/agents/_schemas.py` | 新增 | 新增 StepResult/PlanExecutionResult Schema |
| `src/agent/agents/general.py` | 重构 | 改为 SkillLoader + ReAct Agent |
| `src/agent/agents/knowledge.py` | 优化 | 新增 thread_id 隔离、上下文注入 |
| `src/agent/agents/operation.py` | 优化 | 新增 thread_id 隔离 |
| `src/agent/agents/parallel_executor.py` | 增强 | 新增结构化数据提取、依赖注入 |
| `src/agent/agents/planner.py` | 增强 | 新增 Send 原语执行、复杂度预判优化 |
| `src/agent/graph.py` | 增强 | 新增 Send Worker 节点、状态定义扩展 |
| `src/agent/skills/skill_loader.py` | 增强 | 增强工具加载逻辑 |
| `src/agent/tools/mcp_adapter.py` | 增强 | 新增完整类型化 Schema 系统 |
| `src/api/services/chat_service.py` | 重构 | 提取公共方法，消除重复代码 |
| `src/api/middleware/__init__.py` | 新增 | 统一异常处理中间件 |

---

## 四、测试验证

详见 `docs/verification_test_report.md`

**测试结果**: 28/28 单元测试通过，3/3 端到端集成测试通过

---

## 五、技术债务

| 项目 | 说明 | 优先级 |
|------|------|--------|
| LangGraph 废弃警告 | `create_react_agent` 需迁移到 `langchain.agents` | 中 |
| `pkg_resources` 废弃 | jieba 库依赖，待升级 | 低 |
| Supervisor 节点 | 目前被快速路由跳过，可考虑简化 | 低 |

---

## 六、后续优化建议

1. **Skill 热加载**: 当前 Skill 定义变更需重启服务，可实现热加载机制
2. **工具缓存分级**: MCP 工具初始化较慢（10-30s），可实现增量加载和预热
3. **Mem0 异步批量**: 当前逐条保存，可改为批量异步保存提升性能
4. **流式 SSE 监控**: 增加流式输出的实时监控指标
