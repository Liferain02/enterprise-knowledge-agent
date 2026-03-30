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
        default="qwen3.5-flash",
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

    # Vision 模型配置（用于图片理解）
    vision_model: str = Field(
        default="qwen-vl-plus",
        description="Vision 模型名称: qwen-vl-plus / qwen-vl-max / gpt-4o"
    )
    vision_enabled: bool = Field(
        default=True,
        description="是否启用图片理解功能"
    )
    vision_max_image_size: int = Field(
        default=5,
        description="单张图片最大大小（MB），超过则压缩"
    )

    # ============================================================
    # 视觉入库配置（文档中图片的 Vision LLM 理解）
    # ============================================================
    vision_ingestion_enabled: bool = Field(
        default=True,
        description="入库时是否用 Vision LLM 理解文档中的图片"
    )
    vision_ingestion_model: str = Field(
        default="qwen-vl-plus",
        description="入库时使用的 Vision 模型"
    )
    vision_ingestion_max_images_per_doc: int = Field(
        default=20,
        description="单个文档最多处理图片数（防止超长文档调用过多）"
    )
    vision_ingestion_skip_small: int = Field(
        default=64,
        description="宽或高小于此像素的图片跳过（可能是图标/水印）"
    )
    vision_ingestion_prompt: str = Field(
        default=(
            "请详细描述这张图片的所有内容：\n"
            "1. 图片主体是什么（图表、截图、照片、文档等）\n"
            "2. 图片中包含的所有文字（请完整提取）\n"
            "3. 图表的标题、坐标轴标签、数据趋势\n"
            "4. 任何其他有价值的信息\n"
            "请用中文回答，语言简洁专业。"
        ),
        description="Vision LLM 图片理解提示词"
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
    # --- 字符数配置（保留兼容，仅用于 MarkdownHeader 分块） ---
    chunk_size: int = Field(
        default=2000,
        description="文档分块大小（字符数，用于 MarkdownHeader 分块策略）"
    )
    chunk_overlap: int = Field(
        default=300,
        description="文档分块重叠大小（字符数）"
    )
    chunking_strategy: str = Field(
        default="semantic",
        description="分块策略: recursive(固定长度) / markdown(标题) / semantic(语义) / hybrid(混合)"
    )
    semantic_threshold: float = Field(
        default=0.3,
        description="语义分块阈值 (0-1，越高越敏感)"
    )
    # --- Token 数配置（主导推荐） ---
    chunk_token_size: int = Field(
        default=500,
        description="分块目标 token 数（推荐 300-800，用于 Recursive/Hybrid 分块）"
    )
    chunk_token_overlap: int = Field(
        default=100,
        description="分块 overlap token 数"
    )
    # --- 块内容增强 ---
    chunk_concat_title: bool = Field(
        default=True,
        description="是否将父级 Markdown 标题拼接到每个 chunk 内容前"
    )
    chunk_semantic_overlap: bool = Field(
        default=True,
        description="是否使用语义 overlap（保留前后句）而非固定字符 overlap"
    )
    # --- Embedding 模型（用于估算 token） ---
    embedding_model_for_token: str = Field(
        default="text-embedding-3-small",
        description="Embedding 模型名称（用于 tiktoken token 估算）"
    )

    # Reranker 配置
    reranker_enabled: bool = Field(
        default=True,
        description="是否启用 Reranker 重排序"
    )
    reranker_model: str = Field(
        default="gte-rerank-v2",
        description="Reranker 模型名称 (gte-rerank-v2 / qwen3-rerank)"
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

    # Mem0 记忆配置
    mem0_enabled: bool = Field(
        default=False,
        description="是否启用 Mem0 智能记忆功能"
    )
    mem0_max_context_chars: int = Field(
        default=500,
        description="Mem0 记忆注入上下文的最大字符数"
    )

    # ==================== Corrective RAG 配置 ====================
    crag_enabled: bool = Field(
        default=True,
        description="是否启用 Corrective RAG（检索结果评估与自我纠错）"
    )
    crag_max_retries: int = Field(
        default=1,
        description="Corrective RAG 查询重写后的最大重试次数（降低以减少延迟）"
    )
    crag_grade_threshold: float = Field(
        default=0.25,
        description="CRAG HIGH 相关性阈值（0.0~1.0），>= 此值视为高相关"
    )
    crag_medium_threshold: float = Field(
        default=0.15,
        description="CRAG MEDIUM 阈值（0.0~1.0），>= 此值视为中等相关（可用于生成）"
    )
    crag_min_high_ratio: float = Field(
        default=0.2,
        description="CRAG HIGH 决策的最低 HIGH 文档占比（防止单篇高分误判）"
    )
    crag_candidate_multiplier: int = Field(
        default=2,
        description="CRAG 检索候选倍数：实际检索 k = top_k × 此值，降低以减少 LLM 评估量"
    )
    crag_no_results_low_ratio: float = Field(
        default=0.8,
        description="触发 NO_RESULTS 的最低 LOW 文档比例（0.8 = 80%以上 LOW 才判定无结果）"
    )
    crag_rerank_before_grade: bool = Field(
        default=True,
        description="CRAG LLM 评估前是否先 Rerank 精排候选文档（可减少评估 LLM 调用量）"
    )

    # ==================== 查询扩展配置 ====================
    query_expand_enabled: bool = Field(
        default=True,
        description="是否启用查询扩展分解"
    )
    query_expand_strategy: str = Field(
        default="hybrid",
        description="查询扩展策略: rule_only(纯规则) / llm_only(纯LLM) / hybrid(规则+LLM) / hyde"
    )
    query_expand_max_sub_queries: int = Field(
        default=5,
        description="最大子查询数量"
    )
    query_expand_rerank_fusion_k: int = Field(
        default=60,
        description="RRF 排序参数 k"
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
        """
        加载 MCP 服务器配置。
        args 列表中以 './' 开头的相对路径会被自动解析为项目根目录下的绝对路径，
        确保配置文件与部署路径无关。
        """
        config_path = self.project_root / "config" / "mcp_servers.json"

        if not config_path.exists():
            return []

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            servers = config.get("servers", [])

            # 将 args 中的相对路径解析为绝对路径
            for server in servers:
                resolved = []
                for arg in server.get("args", []):
                    if isinstance(arg, str) and arg.startswith("./"):
                        resolved.append(str(self.project_root / arg[2:]))
                    else:
                        resolved.append(arg)
                server["args"] = resolved

            return servers
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

