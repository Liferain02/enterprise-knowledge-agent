"""
LangGraph 节点模块
定义 Agent 工作流中的各个节点
"""
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import BaseTool
from core.llm import get_llm
from core.chat_history import get_chat_history_manager
from rag.retriever import get_retriever_manager
from tools.base import get_all_tools
from config.prompts import PLANNER_SYSTEM_PROMPT, PLANNER_USER_PROMPT
from config.settings import get_settings
# ==================== 节点函数 ====================
def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    规划节点 - 分析问题并制定执行计划
    """
    settings = get_settings()
    llm = get_llm()
    
    input_text = state.get("input", "")
    session_id = state.get("session_id", "default")
    
    # 获取对话历史
    history_manager = get_chat_history_manager()
    history_text = history_manager.get_history_text(session_id, max_messages=5)
    
    # 构建提示词
    prompt = f"{PLANNER_USER_PROMPT}\n\n用户问题: {input_text}"
    
    # 调用 LLM 进行规划
    response = llm.invoke(prompt)
    
    return {
        "plan": response.content,
        "iteration": state.get("iteration", 0) + 1
    }
def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    记忆节点 - 检索相关上下文
    """
    settings = get_settings()
    
    input_text = state.get("input", "")
    session_id = state.get("session_id", "default")
    use_rag = state.get("use_rag", True)
    
    context = ""
    
    if use_rag:
        # 从向量数据库检索
        try:
            retriever_manager = get_retriever_manager()
            results = retriever_manager.search(input_text, k=settings.retrieval_top_k)
            
            if results:
                context = retriever_manager.format_search_results(results)
        except Exception as e:
            context = f"检索上下文时出错: {str(e)}"
    
    # 获取对话历史
    history_manager = get_chat_history_manager()
    history_text = history_manager.get_history_text(session_id)
    
    return {
        "context": context,
        "messages": state.get("messages", [])
    }
def tool_selection_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    工具选择节点 - 决定使用哪些工具
    """
    llm = get_llm()
    
    input_text = state.get("input", "")
    context = state.get("context", "")
    history_text = state.get("history_text", "")
    
    # 获取所有可用工具
    tools = get_all_tools()
    tools_description = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
    # 让 LLM 选择工具
    prompt = f"""根据以下问题，选择合适的工具：
    
问题: {input_text}

上下文:
{context}

历史:
{history_text}

可用工具:
{tools_description}

请选择需要使用的工具，只需返回工具名称列表，用逗号分隔。
"""
    
    response = llm.invoke(prompt)
    
    # 解析工具名称
    selected = [t.strip() for t in response.content.split(",")]
    
    return {
        "selected_tools": selected
    }
def tool_execution_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    工具执行节点 - 执行选定的工具
    """
    from tools.base import get_tool_by_name
    
    input_text = state.get("input", "")
    selected_tools = state.get("selected_tools", [])
    tool_results = state.get("tool_results", {})
    
    for tool_name in selected_tools:
        tool = get_tool_by_name(tool_name)
        
        if tool is None:
            tool_results[tool_name] = f"工具不存在: {tool_name}"
            continue
        
        try:
            # 根据工具类型设置参数
            if tool_name == "search_knowledge":
                result = tool.invoke(input_text)
            elif tool_name == "calculate":
                # 尝试从输入中提取表达式
                result = tool.invoke({"expression": input_text})
            elif tool_name == "get_date":
                result = tool.invoke({})
            else:
                result = tool.invoke(input_text)
            
            tool_results[tool_name] = result
        
        except Exception as e:
            tool_results[tool_name] = f"执行错误: {str(e)}"
    
    return {
        "tool_results": tool_results
    }
def generation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成节点 - 生成最终答案
    """
    llm = get_llm()
    
    input_text = state.get("input", "")
    context = state.get("context", "")
    tool_results = state.get("tool_results", {})
    history_manager = get_chat_history_manager()
    session_id = state.get("session_id", "default")
    history_text = history_manager.get_history_text(session_id)
    
    # 构建提示词
    prompt_parts = [f"用户问题: {input_text}"]
    
    if context:
        prompt_parts.append(f"\n知识库上下文:\n{context}")
    
    if tool_results:
        prompt_parts.append(f"\n工具执行结果:")
        for tool_name, result in tool_results.items():
            prompt_parts.append(f"- {tool_name}: {result}")
    
    if history_text:
        prompt_parts.append(f"\n对话历史:\n{history_text}")
    
    prompt_parts.append("\n请根据以上信息生成最终答案。")
    
    prompt = "\n".join(prompt_parts)
    
    # 调用 LLM 生成答案
    response = llm.invoke(prompt)
    
    # 保存到历史记录
    history_manager.add_user_message(session_id, input_text)
    history_manager.add_ai_message(session_id, response.content)
    
    return {
        "final_answer": response.content,
        "is_done": True,
        "messages": state.get("messages", []) + [
            HumanMessage(content=input_text),
            AIMessage(content=response.content)
        ]
    }
def check_completion_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    检查完成节点 - 判断是否需要继续迭代
    """
    settings = get_settings()
    
    iteration = state.get("iteration", 0)
    final_answer = state.get("final_answer")
    error = state.get("error")
    
    # 检查是否达到最大迭代
    if iteration >= settings.max_iterations:
        return {
            "is_done": True,
            "error": "达到最大迭代次数"
        }
    
    # 检查是否有错误
    if error:
        return {
            "is_done": True
        }
    
    # 检查是否有答案
    if final_answer:
        return {
            "is_done": True
        }
    
    # 继续迭代
    return {
        "is_done": False
    }
def error_handling_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    错误处理节点
    """
    error = state.get("error", "未知错误")
    
    return {
        "error": f"处理过程中发生错误: {error}",
        "is_done": True
    }
