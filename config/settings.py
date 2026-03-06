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
        default="your-openai-api-key-here",
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
        default="0.0.0.0",
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

    # Auth 配置（单用户）
    auth_enabled: bool = Field(
        default=True,
        description="是否启用登录鉴权"
    )
    admin_username: str = Field(
        default="admin",
        description="管理员用户名"
    )
    admin_password: str = Field(
        default="change-me",
        description="管理员密码（请在 config/.env 中设置）"
    )
    jwt_secret_key: str = Field(
        default="change-me-secret",
        description="JWT 签名密钥（请在 config/.env 中设置）"
    )
    jwt_expire_minutes: int = Field(
        default=720,
        description="JWT 过期时间（分钟）"
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
        default=0.7,
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
    chunking_strategy: str = Field(
        default="recursive",
        description="分块策略: recursive(固定长度) / markdown(标题) / semantic(语义) / hybrid(混合)"
    )
    semantic_threshold: float = Field(
        default=0.3,
        description="语义分块阈值 (0-1，越高越敏感)"
    )

    # Reranker 配置
    reranker_enabled: bool = Field(
        default=True,
        description="是否启用 Reranker 重排序"
    )
    reranker_model: str = Field(
        default="qwen3-rerank",
        description="Reranker 模型名称"
    )
    reranker_provider: str = Field(
        default="qwen",
        description="Reranker 提供商: qwen(阿里百炼) 或 baai(BAAI)"
    )
    reranker_top_n: int = Field(
        default=3,
        description="Reranker 返回的 top n 结果数"
    )
    reranker_threshold: float = Field(
        default=0.1,
        description="Reranker 分数阈值，低于此分数的结果将被过滤"
    )

    # 混合检索配置
    hybrid_search_enabled: bool = Field(
        default=True,
        description="是否启用混合检索（BM25 + 向量）"
    )
    hybrid_vector_weight: float = Field(
        default=0.5,
        description="向量检索权重"
    )
    hybrid_bm25_weight: float = Field(
        default=0.5,
        description="BM25 检索权重"
    )

    # 代理配置
    http_proxy: str = Field(
        default="",
        description="HTTP 代理地址，如 http://127.0.0.1:7897"
    )
    https_proxy: str = Field(
        default="",
        description="HTTPS 代理地址，如 http://127.0.0.1:7897"
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
    mcp_init_timeout: float = Field(
        default=30.0,
        description="MCP服务器连接超时时间（秒）"
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
    summary_threshold: int = Field(
        default=20,
        description="触发对话摘要的消息数阈值（超过此数量时自动总结旧消息）"
    )
    summary_keep_recent: int = Field(
        default=6,
        description="触发摘要后保留的最近消息条数"
    )

    # Checkpointer 配置
    use_sqlite_checkpointer: bool = Field(
        default=False,
        description="是否使用 SQLite 持久化 Agent 状态（需要重启生效）"
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
        """获取Chroma数据库目录（绝对路径）"""
        # 转换为绝对路径，确保所有数据库文件在同一目录
        persist_path = Path(self.chroma_persist_directory)
        if not persist_path.is_absolute():
            persist_path = self.project_root / persist_path
        return persist_path

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


# 全局配置实例（延迟初始化）
_settings_instance = None


def get_settings() -> Settings:
    """获取配置实例（单例模式）"""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    _settings_instance.ensure_directories()
    return _settings_instance

