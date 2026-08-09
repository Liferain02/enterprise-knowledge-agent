"""
General Skill 工具函数
提供通用搜索和对话历史搜索工具
"""
from datetime import datetime


def general_search():
    """创建通用搜索工具"""
    from langchain_core.tools import StructuredTool

    def _do_search(query: str) -> str:
        """
        搜索通用知识或常识性问题。

        此工具用于搜索公开的通用知识，不涉及实验室内部知识库。
        适用于回答日常闲聊、常识性问题等。

        Args:
            query: 要搜索的问题

        Returns:
            搜索结果或说明信息
        """
        # 通用知识库（内置，简单问题直接回答）
        general_knowledge = {
            "天气": "抱歉，我无法获取实时天气信息。您可以查看手机天气应用或搜索天气网站。",
            "时间": f"当前时间是 {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}。",
        }

        # 简单匹配
        for key, value in general_knowledge.items():
            if key in query:
                return value

        # 无法回答时的说明
        return (
            f"关于「{query}」，这是一个通用常识问题。\n\n"
            "如果您需要了解实验室制度、组会安排、项目资料或环境配置，请直接提问，我会帮您查询实验室知识库。\n"
            r'例如："新生入组第一周先看什么？"、"RDMA 实验前要检查哪些环境？"等。'
        )

    return StructuredTool.from_function(
        func=_do_search,
        name="general_search",
        description=(
            "搜索通用知识或常识性问题。此工具用于搜索公开的通用知识，"
            "不涉及实验室内部知识库。适用于回答日常闲聊、常识性问题等。"
            "参数 query：要搜索的问题（字符串）。"
        ),
    )


def search_conversation_history():
    """创建对话历史搜索工具"""
    from langchain_core.tools import StructuredTool

    def _do_search(
        keyword: str,
        session_id: str = "default",
    ) -> str:
        """
        搜索当前对话中之前提到的信息。

        当用户提及之前对话内容时（如"刚才说的什么"、"上次的结论"），
        可使用此工具查找对话历史。

        Args:
            keyword: 搜索关键词（如人名、主题、概念等）
            session_id: 会话 ID，默认为 "default"

        Returns:
            匹配的历史对话片段，或"未找到"说明
        """
        try:
            from src.api.services.session_service import session_service

            messages = session_service.get_messages(session_id)
            if not messages:
                return "当前会话暂无历史记录。"

            # 简单关键词匹配
            results = []
            for msg in messages[-10:]:  # 只搜索最近 10 条
                content = msg.get("content", "").lower()
                if keyword.lower() in content:
                    role = msg.get("role", "")
                    results.append(f"[{role}]: {msg.get('content', '')}")

            if results:
                return "\n".join(results[-3:])  # 最多返回 3 条
            return f"在当前会话历史中未找到与「{keyword}」相关的内容。"

        except Exception as e:
            return f"搜索对话历史时出错: {str(e)}"

    return StructuredTool.from_function(
        func=_do_search,
        name="search_conversation_history",
        description=(
            "搜索当前对话中之前提到的信息。当用户提及之前对话内容时，"
            "可使用此工具查找对话历史。参数 keyword：搜索关键词；"
            "session_id：会话 ID（可选，默认为 default）。"
        ),
    )


def get_general_tools():
    """获取所有 General Skill 工具"""
    return [
        general_search(),
        search_conversation_history(),
    ]
