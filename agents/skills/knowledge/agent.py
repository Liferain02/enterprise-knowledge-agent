"""
Knowledge Skill Agent
使用 ReAct 模式的知识检索 Agent
"""
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from core.llm import get_llm
from .tools import get_tools


SYSTEM_PROMPT = """你是一个知识专家，负责从企业知识库中检索信息并回答用户问题。

工作流程：
1. 接收用户问题
2. 使用 knowledge_search 工具检索相关文档
3. 分析检索结果
4. 生成最终答案并标注信息来源

重要规则：
- 只基于检索到的文档内容回答，不要编造信息
- 在答案中标注信息来源（如"根据《XX文档》..."）
- 如果知识库中没有相关信息，明确告知用户
- 保持回答简洁准确，直接针对用户问题"""

# Agent 缓存
_agent = None


def get_knowledge_agent():
    """获取 Knowledge Agent"""
    global _agent
    if _agent is None:
        llm = get_llm()
        tools = get_tools()
        checkpointer = MemorySaver()
        
        _agent = create_react_agent(
            llm,
            tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=checkpointer
        )
    return _agent


def run(query: str, session_id: str = "default") -> str:
    """运行 Knowledge Agent"""
    agent = get_knowledge_agent()
    config = {"configurable": {"thread_id": f"knowledge_{session_id}"}}
    
    result = agent.invoke(
        {"messages": [query]},
        config
    )
    
    return result["messages"][-1].content

