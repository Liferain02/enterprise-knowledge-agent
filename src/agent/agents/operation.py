"""Selected local capabilities executed under explicit per-turn budgets."""
import asyncio
import json
import logging

from langchain_core.messages import AIMessage

from config.settings import get_settings
from src.models.llm import get_llm
from src.agent.harness.runtime import BoundedToolAgent, BudgetExceeded, create_run
from src.agent.routing.policy import rules_route
from src.agent.skills.registry import get_selected_tools, instructions_for_tools
from ..prompts import OPERATION_AGENT_SYSTEM_PROMPT
from ._utils import get_last_user_message, inject_user_identity_to_messages

logger = logging.getLogger(__name__)
_agent_cache = {}


def _get_operation_agent(tools):
    prompt = OPERATION_AGENT_SYSTEM_PROMPT + "\n\n" + instructions_for_tools(tools)
    settings = get_settings()
    cache_key = (prompt, settings.operation_max_output_tokens, tuple(
        (tool.name, tool.description, json.dumps(tool.args, sort_keys=True)) for tool in tools
    ))
    if cache_key not in _agent_cache:
        if len(_agent_cache) >= 32:
            _agent_cache.pop(next(iter(_agent_cache)))
        model = get_llm(max_tokens=settings.operation_max_output_tokens, max_retries=0)
        _agent_cache[cache_key] = BoundedToolAgent(model, tools, prompt)
    return _agent_cache[cache_key]


async def operation_agent_node(state):
    settings = get_settings()
    run = create_run(settings)
    messages = state.get("messages", [])
    question = get_last_user_message(messages)
    if not question:
        return {"final_answer": "请提供要处理的任务。", "harness_report": run.report("empty_input")}
    decision = state.get("route_decision") or rules_route(question).to_dict()
    try:
        tools = get_selected_tools(decision.get("skills", ()))
        agent = _get_operation_agent(tools)
        context = inject_user_identity_to_messages(messages, user_context=state.get("user_context"),
            summary=state.get("summary", "") or "", mem0_memories=state.get("mem0_memories", "") or "")
        result = await asyncio.wait_for(
            agent.ainvoke({"messages": context}, {"configurable": {"execution_run": run}}),
            timeout=run.remaining(),
        )
        answer = result["messages"][-1].content
        report = result.get("harness_report", run.report("completed"))
    except (BudgetExceeded, asyncio.TimeoutError) as exc:
        code = str(exc) if isinstance(exc, BudgetExceeded) else "timeout"
        run.event("stopped", reason=code)
        answer = "本次任务已达到执行预算或超时，请缩小输入范围后重试。已有原始材料不会被修改。"
        report = run.report("budget_exceeded")
    except Exception:
        logger.exception("操作执行失败 run_id=%s", run.run_id)
        run.event("failed", reason="execution_error")
        answer = "工具执行未能在有限步骤内完成，请换一种更明确的说法后重试。"
        report = run.report("failed")
    return {"final_answer": answer, "used_agent": "operation_agent",
            "messages": [AIMessage(content=answer)], "harness_report": report}
