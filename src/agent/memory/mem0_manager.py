"""
Mem0 记忆管理器
提供智能记忆功能，支持用户级别和会话级别的记忆存储与检索

功能：
- 自动提取和存储对话中的关键信息
- 语义检索相关记忆
- 支持用户画像和偏好记忆
- 多会话记忆聚合
"""
import os
import logging
from typing import Dict, Any, List, Optional
from config.settings import get_settings

logger = logging.getLogger(__name__)


class Mem0MemoryManager:
    """
    Mem0 记忆管理器
    
    使用 mem0ai 提供的智能记忆层，支持：
    - 多级记忆（User、Session、Agent）
    - 自动记忆提取和更新
    - 语义相似度检索
    - 用户偏好学习
    """

    def __init__(self, provider: str = "openai", model: str = None):
        """
        初始化 Mem0 记忆管理器
        
        Args:
            provider: LLM 提供商 ("openai" 或 "qwen")
            model: 使用的模型名称
        """
        self.settings = get_settings()
        self.provider = provider
        self.model = model or self._get_default_model()
        self._client = None
        self._initialized = False

    def _get_default_model(self) -> str:
        """获取默认模型"""
        if self.provider == "qwen":
            return self.settings.dashscope_model
        return self.settings.openai_model

    def _get_api_key(self) -> str:
        """获取 API Key"""
        if self.provider == "qwen":
            return self.settings.dashscope_api_key
        return self.settings.openai_api_key

    def _get_base_url(self) -> str:
        """获取 API Base URL"""
        if self.provider == "qwen":
            return self.settings.dashscope_base_url
        return self.settings.openai_base_url

    def _initialize(self):
        """初始化 Mem0 客户端"""
        if self._initialized:
            return
        
        try:
            from mem0 import Memory
            
            # 设置环境变量（用于 embedding 模型）
            import os
            os.environ["OPENAI_API_KEY"] = self._get_api_key()
            os.environ["OPENAI_BASE_URL"] = self._get_base_url()
            
            # 配置 Mem0 - 使用千问作为 LLM
            config_dict = {
                "llm": {
                    "provider": "openai",
                    "config": {
                        "model": self.model,
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-v2"  # DashScope qwen embedding 模型
                    }
                },
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "mem0_memories",
                        "path": str(self.settings.chroma_dir / "mem0_chroma")
                    }
                }
            }
            
            self._client = Memory.from_config(config_dict=config_dict)
            self._initialized = True
            logger.info(f"Mem0 记忆管理器初始化成功，使用模型: {self.model}")
            
        except Exception as e:
            logger.error(f"Mem0 初始化失败: {e}")
            # 降级模式：创建简单封装
            self._client = None
            self._initialized = True
            logger.warning("Mem0 降级为简单模式，仅记录日志")

    async def add_conversation(
        self,
        messages: List[Dict[str, str]],
        user_id: str = "default_user",
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加对话到记忆
        
        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}, ...]
            user_id: 用户 ID
            session_id: 会话 ID（可选）
            metadata: 附加元数据
            
        Returns:
            操作结果
        """
        self._initialize()
        
        if self._client is None:
            # 降级模式
            return {"success": True, "message": "Mem0 降级模式", "memories": []}
        
        try:
            # 准备记忆数据
            memory_metadata = metadata or {}
            if session_id:
                memory_metadata["session_id"] = session_id
            
            # 添加到 mem0
            result = self._client.add(
                messages=messages,
                user_id=user_id,
                metadata=memory_metadata
            )
            
            logger.info(f"Mem0 添加记忆成功: user={user_id}, session={session_id}")
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Mem0 添加记忆失败: {e}")
            return {"success": False, "error": str(e)}

    async def search(
        self,
        query: str,
        user_id: str = "default_user",
        limit: int = 5,
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        语义检索相关记忆
        
        Args:
            query: 查询文本
            user_id: 用户 ID
            limit: 返回结果数量
            session_id: 过滤特定会话（可选）
            
        Returns:
            记忆列表
        """
        self._initialize()
        
        if self._client is None:
            # 降级模式
            return []
        
        try:
            # 搜索记忆
            filters = {}
            if session_id:
                filters["session_id"] = session_id
            
            results = self._client.search(
                query=query,
                user_id=user_id,
                limit=limit,
                filters=filters if filters else None
            )
            
            return results.get("results", [])
            
        except Exception as e:
            logger.error(f"Mem0 检索记忆失败: {e}")
            return []

    async def get_all_memories(
        self,
        user_id: str = "default_user",
        session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        获取用户的所有记忆
        
        Args:
            user_id: 用户 ID
            session_id: 过滤特定会话（可选）
            
        Returns:
            记忆列表
        """
        self._initialize()
        
        if self._client is None:
            return []
        
        try:
            filters = {}
            if session_id:
                filters["session_id"] = session_id
            
            results = self._client.get_all(
                user_id=user_id,
                filters=filters if filters else None
            )
            
            return results.get("results", [])
            
        except Exception as e:
            logger.error(f"Mem0 获取记忆失败: {e}")
            return []

    async def update_memory(
        self,
        memory_id: str,
        content: str,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        更新记忆
        
        Args:
            memory_id: 记忆 ID
            content: 新内容
            user_id: 用户 ID
            
        Returns:
            操作结果
        """
        self._initialize()
        
        if self._client is None:
            return {"success": False, "error": "Mem0 未初始化"}
        
        try:
            result = self._client.update(
                memory_id=memory_id,
                data=content,
                user_id=user_id
            )
            return {"success": True, "result": result}
            
        except Exception as e:
            logger.error(f"Mem0 更新记忆失败: {e}")
            return {"success": False, "error": str(e)}

    async def delete_memory(
        self,
        memory_id: str,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        删除记忆
        
        Args:
            memory_id: 记忆 ID
            user_id: 用户 ID
            
        Returns:
            操作结果
        """
        self._initialize()
        
        if self._client is None:
            return {"success": True}
        
        try:
            self._client.delete(memory_id=memory_id, user_id=user_id)
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Mem0 删除记忆失败: {e}")
            return {"success": False, "error": str(e)}

    async def delete_user_memories(
        self,
        user_id: str = "default_user"
    ) -> Dict[str, Any]:
        """
        删除用户的所有记忆
        
        Args:
            user_id: 用户 ID
            
        Returns:
            操作结果
        """
        self._initialize()
        
        if self._client is None:
            return {"success": True}
        
        try:
            self._client.delete_all(user_id=user_id)
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Mem0 删除用户记忆失败: {e}")
            return {"success": False, "error": str(e)}

    def format_memories_for_context(
        self,
        memories: List[Dict[str, Any]],
        max_chars: int = 500
    ) -> str:
        """
        将记忆格式化为上下文提示
        
        Args:
            memories: 记忆列表
            max_chars: 最大字符数
            
        Returns:
            格式化的记忆文本
        """
        if not memories:
            return ""
        
        lines = ["【相关记忆】"]
        total_chars = 0
        
        for mem in memories:
            # 兼容不同版本的 mem0 返回格式
            content = mem.get("content") or mem.get("memory") or ""
            if not content:
                continue
            
            # 截断过长的内容
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                if remaining > 50:
                    content = content[:remaining] + "..."
                else:
                    break
            
            lines.append(f"- {content}")
            total_chars += len(content) + 2
        
        return "\n".join(lines)


# 全局单例
_mem0_manager = None


def get_mem0_manager(provider: str = None) -> Mem0MemoryManager:
    """
    获取 Mem0 记忆管理器单例
    
    Args:
        provider: LLM 提供商（可选，默认使用配置中的 llm_provider）
        
    Returns:
        Mem0MemoryManager 实例
    """
    global _mem0_manager
    
    if _mem0_manager is None:
        settings = get_settings()
        llm_provider = provider or settings.llm_provider
        _mem0_manager = Mem0MemoryManager(provider=llm_provider)
    
    return _mem0_manager


def reset_mem0_manager():
    """重置 Mem0 记忆管理器（用于测试）"""
    global _mem0_manager
    _mem0_manager = None
