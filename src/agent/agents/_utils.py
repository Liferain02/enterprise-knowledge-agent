"""
Agents 共享工具函数
"""
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage


def get_last_user_message(messages: List) -> Optional[str]:
    """获取最后一条用户消息"""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def build_user_identity_context(user_context) -> str:
    """
    构建用户身份上下文字符串，告知 LLM 当前提问者的身份。

    Args:
        user_context: UserContext 对象或 dict，包含 username/role 等

    Returns:
        格式化的身份字符串，无有效上下文时返回空字符串
    """
    if not user_context:
        return ""

    # 支持 UserContext 对象或 dict
    if hasattr(user_context, "username"):
        username = user_context.username
        role = getattr(user_context, "role", "student")
        department_name = getattr(user_context, "department_name", "")
    elif isinstance(user_context, dict):
        username = user_context.get("username", "")
        role = user_context.get("role", "student")
        department_name = user_context.get("department_name", "")
    else:
        return ""

    if not username or username == "anonymous":
        return ""

    role_display = {
        "admin": "管理员",
        "pi": "导师/PI",
        "teacher": "教师",
        "lab_admin": "实验室管理员",
        "senior_student": "高年级成员",
        "student": "研究生",
        "assistant": "助研/本科生",
        "manager": "项目负责人",
        "hr": "实验室管理员",
        "it_support": "平台支持",
        "employee": "研究组成员",
    }.get(role, role)

    parts = [f"当前用户：{username}（{role_display}）"]
    if department_name:
        parts[0] += f"，所属项目组：{department_name}"

    return "\n\n【当前用户身份】" + "\n".join(parts)


def inject_user_identity_to_messages(
    messages: List,
    user_context,
    summary: str = "",
    mem0_memories: str = ""
) -> List:
    """
    综合注入：用户身份 + Mem0 记忆 + 摘要 到消息列表。

    注入顺序（优先级从高到低）：
    1. 用户身份（最先，让 LLM 始终知道当前用户）
    2. Mem0 记忆（用户偏好、历史）
    3. 对话摘要（早期上下文）
    4. 用户消息
    """
    injected = list(messages)

    # 第3步：注入对话摘要
    if summary:
        summary_msg = SystemMessage(
            content=f"【历史对话摘要】以下是本次对话早期内容的摘要，请结合它理解用户的上下文：\n{summary}"
        )
        injected = [summary_msg] + injected

    # 第2步：注入 Mem0 记忆
    if mem0_memories:
        mem0_msg = SystemMessage(content=mem0_memories)
        injected = [mem0_msg] + injected

    # 第1步：注入用户身份（最优先）
    user_identity = build_user_identity_context(user_context)
    if user_identity:
        identity_msg = SystemMessage(content=user_identity)
        injected = [identity_msg] + injected

    return injected
