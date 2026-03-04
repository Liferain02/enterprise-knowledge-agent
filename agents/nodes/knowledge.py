"""
Knowledge Agent 节点
负责知识检索和回答
使用 Skill Loader 动态加载
"""
from typing import Dict, Any
from agents.skills import get_skill_loader
from agents.nodes.utils import get_last_user_message


async def knowledge_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Knowledge Agent 节点 - 负责知识检索和回答
    
    使用 Skill Loader 动态加载 Skill.md 定义
    
    改为 async def，与其他节点保持一致
    
    注意：不使用 Agent 自己的 checkpointer，
    历史由主图统一管理
    """
    # 获取用户最新消息
    messages = state.get("messages", [])
    last_user_message = get_last_user_message(messages)
    
    # 获取 session_id
    session_id = state.get("session_id", "default")
    
    if not last_user_message:
        return {
            "final_answer": "抱歉，我无法理解您的问题。"
        }
    
    try:
        # 使用 SkillLoader 获取 Agent
        loader = get_skill_loader()
        agent = loader.create_agent("knowledge")
        
        # 传递完整的 messages（包含历史）
        config = {"configurable": {"thread_id": session_id}}
        
        # 执行 Agent（传递完整历史）
        result = await agent.ainvoke(
            {"messages": messages},
            config
        )
        
        # 获取 Agent 返回的所有消息
        agent_messages = result.get("messages", [])
        
        # 获取最终回复
        final_answer = agent_messages[-1].content
        
        print(f"[Knowledge Agent] 生成答案长度: {len(final_answer)} 字符")
        
        return {
            "final_answer": final_answer,
            "sources": "knowledge_base",
            "used_agent": "knowledge_agent",
            "messages": agent_messages
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
