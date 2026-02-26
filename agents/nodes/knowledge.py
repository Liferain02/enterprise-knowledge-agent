"""
Knowledge Agent 节点
负责知识检索和回答
使用 langgraph-prebuilt 的 create_tool_calling_agent
"""
from typing import Dict, Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from core.llm import get_llm
from tools import get_knowledge_tool
from agents.nodes.utils import get_last_user_message

# Agent 缓存
_agent_cache = None


def _get_knowledge_agent():
    """获取 Knowledge Agent（带缓存）"""
    global _agent_cache
    if _agent_cache is None:
        llm = get_llm()
        tools = [get_knowledge_tool()]
        checkpointer = MemorySaver()
        
        prompt = """你是一个知识专家，负责从企业知识库中检索信息并回答用户问题。

工作流程：
1. 接收用户问题
2. 使用 knowledge_search 工具检索相关文档
3. 分析检索结果
4. 生成最终答案并标注信息来源

重要：
- 只基于检索到的文档内容回答，不要编造信息
- 在答案中标注信息来源
- 如果知识库中没有相关信息，明确告知用户"""
        
        _agent_cache = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=prompt,
            checkpointer=checkpointer
        )
    return _agent_cache


def knowledge_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Knowledge Agent 节点 - 负责知识检索和回答
    
    使用 langgraph-prebuilt 的 create_tool_calling_agent
    自动处理知识库检索
    """
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    try:
        # 获取预建的 Knowledge Agent
        agent = _get_knowledge_agent()
        
        # 构建配置
        config = {"configurable": {"thread_id": "knowledge_agent"}}
        
        # 执行 Agent
        result = agent.invoke(
            {"messages": [last_user_message]},
            config
        )
        
        # 获取最终回复
        final_answer = result["messages"][-1].content
        
        print(f"[Knowledge Agent] 生成答案长度: {len(final_answer)} 字符")
        
        return {
            "final_answer": final_answer,
            "sources": "knowledge_base",
            "used_agent": "knowledge_agent"
        }
        
    except Exception as e:
        print(f"[Knowledge Agent] 执行出错: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "final_answer": f"搜索知识库时出错: {str(e)}",
            "sources": "",
            "used_agent": "knowledge_agent"
        }
