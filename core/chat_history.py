"""
对话历史管理模块
支持短期记忆（会话级）和长期记忆（向量存储）
"""
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from config.settings import get_settings


class SessionChatHistory:
    """会话级别的对话历史管理"""
    
    def __init__(self, session_id: str, max_messages: int = 20):
        self.session_id = session_id
        self.max_messages = max_messages
        self._history: InMemoryChatMessageHistory = InMemoryChatMessageHistory()
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def add_user_message(self, content: str):
        """添加用户消息"""
        self._history.add_user_message(content)
        self._trim_history()
        self.last_accessed = datetime.now()
    
    def add_ai_message(self, content: str):
        """添加AI消息"""
        self._history.add_ai_message(content)
        self._trim_history()
        self.last_accessed = datetime.now()
    
    def add_message(self, message: BaseMessage):
        """添加任意消息"""
        self._history.add_message(message)
        self._trim_history()
        self.last_accessed = datetime.now()
    
    def _trim_history(self):
        """裁剪历史消息"""
        messages = self._history.messages
        if len(messages) > self.max_messages:
            self._history.messages = messages[-self.max_messages:]
    
    def get_messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        self.last_accessed = datetime.now()
        return self._history.messages
    
    def get_history_text(self, max_messages: Optional[int] = None) -> str:
        """获取历史文本格式"""
        messages = self._history.messages
        if max_messages:
            messages = messages[-max_messages:]
        
        history_parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history_parts.append(f"用户: {msg.content}")
            elif isinstance(msg, AIMessage):
                history_parts.append(f"助手: {msg.content}")
        
        return "\n".join(history_parts)
    
    def clear(self):
        """清空历史"""
        self._history.clear()
    
    def is_expired(self, expire_seconds: int = 3600) -> bool:
        """检查是否过期"""
        return (datetime.now() - self.last_accessed).total_seconds() > expire_seconds


class ChatHistoryManager:
    """对话历史管理器"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._sessions: Dict[str, SessionChatHistory] = {}
        self._max_messages = self.settings.max_history_messages
        self._expire_seconds = self.settings.session_expire_seconds
    
    def get_session(self, session_id: str) -> SessionChatHistory:
        """获取或创建会话"""
        # 清理过期会话
        self._cleanup_expired_sessions()
        
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionChatHistory(
                session_id,
                max_messages=self._max_messages
            )
        
        return self._sessions[session_id]
    
    def add_user_message(self, session_id: str, content: str):
        """添加用户消息"""
        session = self.get_session(session_id)
        session.add_user_message(content)
    
    def add_ai_message(self, session_id: str, content: str):
        """添加AI消息"""
        session = self.get_session(session_id)
        session.add_ai_message(content)
    
    def get_history(self, session_id: str) -> List[BaseMessage]:
        """获取会话历史"""
        session = self.get_session(session_id)
        return session.get_messages()
    
    def get_history_text(
        self,
        session_id: str,
        max_messages: Optional[int] = None
    ) -> str:
        """获取历史文本"""
        session = self.get_session(session_id)
        return session.get_history_text(max_messages)
    
    def clear_session(self, session_id: str):
        """清空会话"""
        if session_id in self._sessions:
            self._sessions[session_id].clear()
    
    def _cleanup_expired_sessions(self):
        """清理过期会话"""
        expired_sessions = [
            sid for sid, session in self._sessions.items()
            if session.is_expired(self._expire_seconds)
        ]
        for sid in expired_sessions:
            del self._sessions[sid]
    
    def get_all_sessions(self) -> List[str]:
        """获取所有会话ID"""
        return list(self._sessions.keys())


class LongTermMemory:
    """长期记忆 - 基于向量存储"""
    
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore
    
    def add_memory(self, content: str, metadata: Optional[Dict] = None):
        """添加记忆"""
        if self.vectorstore is None:
            return
        # 这里可以添加长期记忆的存储逻辑
        pass
    
    def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关记忆"""
        if self.vectorstore is None:
            return []
        # 可以复用 RAG 的检索逻辑
        return []


# 全局实例
_chat_history_manager: Optional[ChatHistoryManager] = None


def get_chat_history_manager() -> ChatHistoryManager:
    """获取对话历史管理器实例"""
    global _chat_history_manager
    if _chat_history_manager is None:
        _chat_history_manager = ChatHistoryManager()
    return _chat_history_manager

