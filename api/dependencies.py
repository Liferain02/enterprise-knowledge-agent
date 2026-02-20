"""
API 依赖模块
"""
from typing import Optional
from fastapi import Depends
from core.llm import LLMManager, get_llm
from core.chat_history import ChatHistoryManager, get_chat_history_manager
from rag.vectorstore import VectorStoreManager, get_vectorstore_manager
from rag.retriever import RetrieverManager, get_retriever_manager
from config.settings import Settings, get_settings
def get_llm_manager() -> LLMManager:
    """获取LLM管理器"""
    return LLMManager()
def get_settings_dep() -> Settings:
    """获取设置依赖"""
    return get_settings()
# 可以添加更多依赖...

