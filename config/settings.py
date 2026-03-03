"""
配置管理模块 - 使用 Pydantic Settings
"""
import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """应用配置类"""

    model_config = SettingsConfigDict(
        env_file="config/.env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # 阿里千问 API 配置
    dashscope_api_key: str = Field(
        default="your-dashscope-api-key-here",
        description="阿里千问 API Key"
    )
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="阿里千问 API 基础URL"
    )
    dashscope_model: str = Field(
        default="qwen-plus",
        description="使用的千问模型名称"
    )

    # OpenAI API 配置 (备用)
    openai_api_key: str = Field(
        default="sk-f74df7dc93df4e0ab282668c72a834f1",
        description="OpenAI API Key"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API 基础URL"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="使用的模型名称"
    )

    # 使用的 LLM 类型: "qwen" 或 "openai"
    llm_provider: str = Field(
        default="qwen",
        description="LLM 提供商: qwen / openai"
    )

    # Chroma 向量数据库配置
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="Chroma数据库持久化目录"
    )

    # FastAPI 配置
    api_host: str = Field(
        default="127.0.0.1",
        description="API服务主机"
    )
    api_port: int = Field(
        default=8000,
        description="API服务端口"
    )
    debug: bool = Field(
        default=True,
        description="调试模式"
    )

    # Agent 配置
    max_iterations: int = Field(
        default=10,
        description="Agent最大迭代次数"
    )
    agent_temperature: float = Field(
        default=0.7,
        description="LLM温度参数"
    )
    max_token_response: int = Field(
        default=2000,
        description="LLM响应最大token数"
    )

    # RAG 配置
    retrieval_top_k: int = Field(
        default=5,
        description="检索返回的top k结果数"
    )
    similarity_threshold: float = Field(
        default=0.3,
        description="相似度阈值 (0-1，越低越宽松)"
    )
    chunk_size: int = Field(
        default=1000,
        description="文档分块大小"
    )
    chunk_overlap: int = Field(
        default=200,
        description="文档分块重叠大小"
    )

    # MCP 配置
    mcp_server_enabled: bool = Field(
        default=True,
        description="是否启用MCP服务器"
    )
    mcp_server_port: int = Field(
        default=8001,
        description="MCP服务器端口"
    )

    # 会话配置
    session_expire_seconds: int = Field(
        default=3600,
        description="会话过期时间(秒)"
    )
    max_history_messages: int = Field(
        default=20,
        description="最大历史消息数"
    )

    # 项目根目录
    @property
    def project_root(self) -> Path:
        """获取项目根目录"""
        return Path(__file__).parent.parent

    @property
    def knowledge_base_dir(self) -> Path:
        """获取知识库目录"""
        return self.project_root / "data" / "knowledge"

    @property
    def chroma_dir(self) -> Path:
        """获取Chroma数据库目录"""
        return Path(self.chroma_persist_directory)

    def ensure_directories(self):
        """确保必要的目录存在"""
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_base_dir.mkdir(parents=True, exist_ok=True)

    def load_mcp_servers_config(self) -> List[Dict[str, Any]]:
        """加载 MCP 服务器配置"""
        config_path = self.project_root / "config" / "mcp_servers.json"
        
        if not config_path.exists():
            return []
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                return config.get("servers", [])
        except Exception as e:
            print(f"加载 MCP 服务器配置失败: {e}")
            return []


# 全局配置实例
settings = Settings()


def get_settings() -> Settings:
    """获取配置实例"""
    settings.ensure_directories()
    return settings

