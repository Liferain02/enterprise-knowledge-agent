"""
LLM 初始化模块
支持 OpenAI 兼容的 API 和阿里千问 (Qwen)
使用 functools.lru_cache 实现简单缓存
"""
from functools import lru_cache
from typing import Optional
from langchain_openai import ChatOpenAI
from config.settings import get_settings


@lru_cache(maxsize=4)
def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    provider: Optional[str] = None,
) -> ChatOpenAI:
    """
    获取 LLM 实例（带缓存）
    
    Args:
        model: 模型名称，默认使用配置
        temperature: 温度参数
        max_tokens: 最大token数
        provider: LLM 提供商 "qwen" 或 "openai"，默认使用配置
    
    Returns:
        ChatOpenAI 实例
    """
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
    
    # 创建新实例
    return ChatOpenAI(
        model=model,
        temperature=temperature if temperature is not None else settings.agent_temperature,
        max_tokens=max_tokens or settings.max_token_response,
        api_key=api_key,
        base_url=base_url,
    )


def reset_llm():
    """重置 LLM 缓存"""
    get_llm.cache_clear()
