"""
Planner 节点
负责分析任务复杂度并拆解步骤
"""
import re
import json
from typing import Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from src.models.llm import get_llm


# 使用 TypedDict 而非 Pydantic BaseModel：
# TypedDict 是 LangGraph 的原生类型，with_structured_output 支持它，
# 且不会触发 Pydantic 在序列化 OpenAI 响应时的 "Expected none but got ..." 警告。
class PlanStep(TypedDict):
    """单个步骤的定义"""
    step_id: int
    description: str
    agent: str        # knowledge_agent / operation_agent / general_agent
    depends_on: List[int]


class TaskPlan(TypedDict):
    """任务规划结构化输出"""
    is_complex: bool
    reasoning: str
    steps: List[PlanStep]
    final_agent: str


class SummaryOutput(TypedDict):
    """汇总步骤结果的结构化输出"""
    final_answer: str


# ==================== 快速复杂度预判 ====================

# 命中任意一条 → 判定为复杂任务（走 LLM planner 拆步骤）
_COMPLEX_PATTERNS = [
    r"对比|比较|区别|差异|异同|不同点",          # 对比类
    r"\bvs\b|VS|versus",                          # 英文对比
    r"分别.*查|分别.*看|各自|各个",               # 多信息并行
    r"先.{0,10}再|先.{0,10}然后|然后再|之后再",  # 顺序执行
    r"总结.{0,15}和|汇总|综合.{0,15}和|梳理",    # 汇总类
    r"第一.{0,20}第二|①.{0,20}②",               # 列举多项
    r"多个.{0,10}政策|多个.{0,10}文档",          # 多文档
]

# 命中任意一条 → 判定为简单任务（直接跳过 LLM planner）
_SIMPLE_PATTERNS = [
    r"^(你好|hi|hello|您好|早上好|下午好|晚上好|在吗|嗨).{0,10}$",  # 问候
    r"^(谢谢|感谢|多谢|thanks|thank you).{0,15}$",                    # 致谢
    r"^(再见|拜拜|bye|晚安|好的|okay|ok|收到).{0,10}$",              # 结束语
    r"^现在(几点|时间|日期)|^今天(几号|星期|日期)|^当前时间",         # 时间查询
]

_COMPILED_COMPLEX = [re.compile(p, re.IGNORECASE) for p in _COMPLEX_PATTERNS]
_COMPILED_SIMPLE  = [re.compile(p, re.IGNORECASE) for p in _SIMPLE_PATTERNS]


def _quick_complexity_check(message: str) -> str:
    """
    快速复杂度预判（纯规则，无 LLM 调用，耗时 < 1ms）

    Returns:
        "simple"    确定是简单任务，跳过 LLM planner，节省一次 LLM 调用
        "complex"   确定是复杂任务，走 LLM planner 拆解步骤
        "uncertain" 无法确定，走 LLM planner 精确判断

    策略：宁可漏报 complex（降级为 uncertain → LLM 判断），
    不可误报 simple（跳过 LLM 导致复杂任务未被拆解）。
    """
    msg = message.strip()

    # 极短消息 → 必然简单（问候/单词回复）
    if len(msg) <= 8:
        return "simple"

    # 命中复杂信号 → 交给 LLM 拆步骤
    for pattern in _COMPILED_COMPLEX:
        if pattern.search(msg):
            return "complex"

    # 命中简单信号 → 跳过 LLM
    for pattern in _COMPILED_SIMPLE:
        if pattern.search(msg):
            return "simple"

    # 多个问号 → 多问题 → complex
    if msg.count("？") >= 2 or msg.count("?") >= 2:
        return "complex"

    # 消息较短（≤ 40字）且无复杂信号 → 大概率简单
    if len(msg) <= 40:
        return "simple"

    # 其余交给 LLM 判断
    return "uncertain"


# ==================== Planner 节点 ====================

async def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Planner 节点 - 分析任务复杂度并拆解步骤

    工作逻辑：
    1. 快速规则预判（无 LLM，< 1ms）
       - 确定简单 → 直接返回，跳过 LLM（节省 5-10 秒）
       - 确定复杂 / 不确定 → 走 LLM 精确判断
    2. LLM 判断（仅对 complex / uncertain 触发）
       - 简单任务 → 返回空步骤，后续由 Supervisor 处理
       - 复杂任务 → 拆解成步骤序列，由 execute_plan 执行

    复杂任务示例：
    - "对比 A 政策和 B 政策的差异" → 需要两步：查A政策，查B政策
    - "算一下我下个月能休几天假" → 需要多步：查假期政策，查日历，计算
    - "查询多个文档后总结" → 需要多步：检索多个文档，汇总
    """
    # 获取用户消息
    messages = state.get("messages", [])
    last_user_message = _get_last_user_message(messages)

    if not last_user_message:
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": "无用户消息输入"
        }

    # ── 快速路径：规则预判 ──────────────────────────────────────────
    quick_result = _quick_complexity_check(last_user_message)

    if quick_result == "simple":
        print(f"[Planner] 快速判断: 简单任务（跳过 LLM）")
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": "规则快速判断为简单任务",
            "current_step": 0,
        }

    if quick_result == "complex":
        print(f"[Planner] 快速判断: 复杂任务（进入 LLM 拆步骤）")
    else:
        print(f"[Planner] 快速判断: 不确定（进入 LLM 精确判断）")

    # ── LLM 路径：精确判断（仅 complex / uncertain 到达此处）─────────
    llm = get_llm()
    
    # 构建分析提示词
    prompt = f"""请分析以下用户问题，判断是否需要拆解成多个步骤来处理。

用户问题：{last_user_message}

## 复杂度判断标准

**简单任务**（不需要拆解）：
- 单一问题，可以一步回答
- 如："现在几点"、"公司年假多少天"、"你好"

**复杂任务**（需要拆解成多步骤）：
- 需要对比多个内容（如"对比A和B"）
- 需要先查询多个信息再计算或总结
- 包含多个子问题
- 如：
  - "对比年假和病假政策的差异" → 需要分别查年假政策、病假政策，然后对比
  - "我下个月能休几天假" → 需要查假期政策 + 查当前月份 + 计算
  - "总结Q1和Q2的业绩表现" → 需要分别查Q1、Q2数据，然后汇总

## Agent 类型说明

- **knowledge_agent**: 知识检索，从向量数据库查询信息
- **operation_agent**: 执行计算、时间查询、工具调用
- **general_agent**: 通用对话、汇总答案

## 输出格式

请严格按照以下 JSON 格式输出：

{{
  "is_complex": true/false,
  "reasoning": "判断理由",
  "steps": [
    {{
      "step_id": 1,
      "description": "步骤描述",
      "agent": "knowledge_agent/operation_agent/general_agent",
      "depends_on": []
    }}
    // ... 更多步骤
  ],
  "final_agent": "general_agent"
}}

**重要**：
- 如果 is_complex 为 false，steps 为空数组
- 如果 is_complex 为 true，至少有2个步骤
- depends_on 填写前置步骤的 step_id，如 [1] 表示依赖第1步
- final_agent 通常是 general_agent，用于汇总最终答案
"""
    
    try:
        llm_structured = llm.with_structured_output(TaskPlan)
        plan: TaskPlan = await llm_structured.ainvoke(prompt)

        # TypedDict 返回普通 dict，直接用 [] 访问
        is_complex = plan["is_complex"]
        reasoning = plan["reasoning"]
        steps = plan.get("steps", [])  # 每个 step 已是 dict

        print(f"[Planner] 任务复杂度: {is_complex}, 步骤数: {len(steps)}")
        print(f"[Planner] 判断理由: {reasoning}")

        if is_complex:
            for s in steps:
                print(f"[Planner] 步骤 {s['step_id']}: {s['description']} -> {s['agent']}")

        return {
            "is_complex": is_complex,
            "plan_steps": steps,
            "plan_reasoning": reasoning,
            "current_step": 0,
        }
        
    except Exception as e:
        print(f"[Planner] 规划失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 降级：认为是非复杂任务
        return {
            "is_complex": False,
            "plan_steps": [],
            "plan_reasoning": f"规划失败，降级为简单任务: {str(e)}",
            "current_step": 0
        }


def _get_last_user_message(messages: list) -> str:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


# ==================== 路由函数 ====================

def route_from_planner(state: Dict[str, Any]) -> str:
    """
    从 Planner 路由到下一个节点
    
    - 如果是复杂任务 -> 执行第一个步骤
    - 如果是简单任务 -> 交给 Supervisor 处理
    """
    is_complex = state.get("is_complex", False)
    
    if is_complex:
        # 复杂任务：开始执行步骤
        return "execute_plan"
    else:
        # 简单任务：交给 Supervisor
        return "supervisor"


# ==================== 计划执行节点 ====================

async def execute_plan_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行计划步骤的节点
    
    根据 current_step 执行对应的步骤
    步骤执行完成后，检查是否还有更多步骤：
    - 有 -> 更新 current_step，继续执行
    - 没有 -> 跳转到 Reflect 节点
    
    当前实现：简化版本，只收集结果，不实际调用 Agent
    后续可以扩展为真正调用各个 Agent
    """
    from .knowledge import knowledge_agent_node
    from .operation import operation_agent_node
    from .general import general_agent_node
    
    plan_steps = state.get("plan_steps", [])
    current_step = state.get("current_step", 0)
    messages = state.get("messages", [])
    
    if not plan_steps or current_step >= len(plan_steps):
        return {"current_step": -1}
    
    # 获取当前步骤
    step = plan_steps[current_step]
    print(f"[Execute Plan] 执行步骤 {step['step_id']}: {step['description']}")
    print(f"[Execute Plan] 分配给 Agent: {step['agent']}")
    
    # 根据步骤指定的 agent 执行
    agent_name = step.get("agent", "general_agent")
    step_result = ""
    sources = ""
    
    try:
        if agent_name == "knowledge_agent":
            # 构造子问题（只包含当前步骤的问题）
            sub_question = step['description']
            sub_messages = messages + [HumanMessage(content=sub_question)]
            
            result = await knowledge_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")
            sources = result.get("sources", "")
            
        elif agent_name == "operation_agent":
            sub_question = step['description']
            sub_messages = messages + [HumanMessage(content=sub_question)]
            
            result = await operation_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")
            
        elif agent_name == "general_agent":
            sub_question = step['description']
            sub_messages = messages + [HumanMessage(content=sub_question)]
            
            result = await general_agent_node({
                "messages": sub_messages,
                "session_id": state.get("session_id", "default")
            })
            step_result = result.get("final_answer", "")
        
        print(f"[Execute Plan] 步骤 {step['step_id']} 结果: {step_result[:100]}...")
        
    except Exception as e:
        print(f"[Execute Plan] 步骤执行失败: {e}")
        step_result = f"步骤执行出错: {str(e)}"
    
    # 收集步骤结果
    plan_results = state.get("plan_results", [])
    plan_results.append({
        "step_id": step['step_id'],
        "description": step['description'],
        "agent": agent_name,
        "result": step_result
    })
    
    # 更新已完成步骤
    completed = state.get("completed_steps", [])
    completed.append(step['step_id'])
    
    # 检查是否还有更多步骤
    next_step = current_step + 1
    if next_step >= len(plan_steps):
        # 所有步骤完成，汇总结果
        print(f"[Execute Plan] 所有步骤完成，开始汇总")
        
        # 生成最终答案（汇总所有步骤结果）
        final_answer = await _summarize_results(
            state.get("messages", []),
            plan_results
        )
        
        return {
            "current_step": -1,
            "completed_steps": completed,
            "plan_results": plan_results,
            "final_answer": final_answer,
            "used_agent": "planner"  # 标记为 planner 执行的
        }
    else:
        # 还有步骤，继续执行
        print(f"[Execute Plan] 步骤 {step['step_id']} 完成，继续执行下一步")
        return {
            "current_step": next_step,
            "completed_steps": completed,
            "plan_results": plan_results
        }


async def _summarize_results(messages: list, plan_results: list) -> str:
    """
    汇总所有步骤的结果，生成最终答案
    """
    llm = get_llm()
    user_question = _get_last_user_message(messages)
    
    # 构建汇总提示
    results_text = ""
    for r in plan_results:
        results_text += f"\n步骤 {r['step_id']}: {r['description']}\n结果: {r['result']}\n"
    
    prompt = f"""用户问题：{user_question}

各步骤执行结果：
{results_text}

请根据以上各步骤的结果，生成一个完整、连贯的最终答案。

要求：
1. 直接回答用户的问题
2. 整合各步骤的结果
3. 保持答案连贯流畅
4. 不要提及"步骤1"、"步骤2"等内部过程

最终答案："""

    try:
        llm_structured = llm.with_structured_output(SummaryOutput)
        summary: SummaryOutput = await llm_structured.ainvoke(prompt)
        # TypedDict 返回普通 dict
        return summary["final_answer"]
    except Exception as e:
        # 如果汇总失败，简单拼接结果
        print(f"[Execute Plan] 汇总失败: {e}")
        return "\n\n".join([r['result'] for r in plan_results])


def route_execute_plan(state: Dict[str, Any]) -> str:
    """
    从计划执行节点路由
    
    - current_step == -1 -> 所有步骤完成，结束
    - 否则 -> 继续执行下一个步骤
    """
    current_step = state.get("current_step", 0)
    
    if current_step == -1:
        return "END"
    else:
        return "execute_plan"
