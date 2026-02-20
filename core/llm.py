"""
LLM 初始化模块
支持 OpenAI 兼容的 API 和阿里千问 (Qwen)
"""
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from config.settings import get_settings


# 全局 LLM 实例
_llm_instance: Optional[ChatOpenAI] = None


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    获取 LLM 实例（单例模式）
    
    Args:
        model: 模型名称，默认使用配置
        temperature: 温度参数
        max_tokens: 最大token数
        provider: LLM 提供商 "qwen" 或 "openai"，默认使用配置
        **kwargs: 其他参数
    
    Returns:
        ChatOpenAI 实例
    """
    global _llm_instance
    
    settings = get_settings()
    
    # 确定使用哪个 LLM 提供商
    llm_provider = provider or settings.llm_provider
    
    # 根据提供商获取配置
    if llm_provider == "qwen":
        api_key = settings.dashscope_api_key
        base_url = settings.dashscope_base_url
        default_model = settings.dashscope_model
    else:
        api_key = settings.openai_api_key
        base_url = settings.openai_base_url
        default_model = settings.openai_model
    
    model = model or default_model
    
    # 如果已有实例且参数相同，直接返回
    if _llm_instance is not None:
        current_config = {
            "model": _llm_instance.model_name,
            "temperature": _llm_instance.temperature,
            "max_tokens": _llm_instance.max_tokens
        }
        new_config = {
            "model": model,
            "temperature": temperature if temperature is not None else settings.agent_temperature,
            "max_tokens": max_tokens or settings.max_token_response
        }
        
        if current_config == new_config:
            return _llm_instance
    
    # 创建新实例
    _llm_instance = ChatOpenAI(
        model=model,
        temperature=temperature if temperature is not None else settings.agent_temperature,
        max_tokens=max_tokens or settings.max_token_response,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )
    
    return _llm_instance


def reset_llm():
    """重置 LLM 实例"""
    global _llm_instance
    _llm_instance = None


class LLMManager:
    """LLM 管理器类"""
    
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._llm: Optional[ChatOpenAI] = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """获取 LLM 实例"""
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def create_chat_model(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> ChatOpenAI:
        """创建新的聊天模型实例"""
        settings = self.settings
        llm_provider = provider or settings.llm_provider
        
        if llm_provider == "qwen":
            api_key = settings.dashscope_api_key
            base_url = settings.dashscope_base_url
            default_model = settings.dashscope_model
        else:
            api_key = settings.openai_api_key
            base_url = settings.openai_base_url
            default_model = settings.openai_model
        
        return ChatOpenAI(
            model=model or default_model,
            temperature=temperature if temperature is not None else settings.agent_temperature,
            max_tokens=max_tokens or settings.max_token_response,
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        llm = self.llm
        settings = self.settings
        provider = settings.llm_provider
        
        base_url = settings.dashscope_base_url if provider == "qwen" else settings.openai_base_url
        
        return {
            "model_name": llm.model_name,
            "temperature": llm.temperature,
            "max_tokens": llm.max_tokens,
            "base_url": base_url,
            "provider": provider
        }

