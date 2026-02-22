"""
MCP 客户端连接管理模块
负责连接外部 MCP 服务器并管理工具
"""
import asyncio
from typing import List, Optional, Dict, Any
from config.settings import get_settings
from tools.mcp_tools import MCPServerConfig, MCPClient, MCPToolManager


class MCPConnectionManager:
    """MCP 连接管理器 - 单例模式"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not MCPConnectionManager._initialized:
            self.tool_manager: Optional[MCPToolManager] = None
            self.mcp_clients: Dict[str, MCPClient] = {}
            self._initialized_connect = False
            MCPConnectionManager._initialized = True
    
    async def initialize(self):
        """初始化 MCP 连接（懒加载模式）"""
        # 如果已经初始化过，直接返回
        if self._initialized_connect:
            return
        
        self._initialized_connect = True
        settings = get_settings()
        
        # 如果未启用 MCP，直接返回
        if not settings.mcp_server_enabled:
            print("MCP 服务器未启用")
            return
        
        # 加载 MCP 服务器配置
        mcp_configs = settings.load_mcp_servers_config()
        
        if not mcp_configs:
            print("未配置 MCP 服务器")
            return
        
        # 创建工具管理器
        self.tool_manager = MCPToolManager()
        
        # 连接每个 MCP 服务器
        for config in mcp_configs:
            try:
                server_config = MCPServerConfig(**config)
                print(f"正在连接 MCP 服务器: {server_config.name}...")
                
                # 直接调用连接
                success = await self.tool_manager.connect_mcp_server(server_config)
                
                if success:
                    print(f"✓ MCP 服务器 '{server_config.name}' 连接成功")
                else:
                    print(f"✗ MCP 服务器 '{server_config.name}' 连接失败")
                    
            except Exception as e:
                print(f"连接 MCP 服务器 '{config.get('name', 'unknown')}' 时出错: {e}")
    
    def get_tools(self) -> List:
        """获取所有 MCP 工具"""
        if self.tool_manager is None:
            return []
        return self.tool_manager.get_all_tools()
    
    def get_local_tools(self) -> List:
        """获取本地工具"""
        if self.tool_manager is None:
            return []
        return self.tool_manager.get_local_tools()
    
    def get_mcp_tools(self) -> List:
        """获取 MCP 工具"""
        if self.tool_manager is None:
            return []
        return self.tool_manager.get_mcp_tools()
    
    async def close(self):
        """关闭所有 MCP 连接"""
        if self.tool_manager is not None:
            await self.tool_manager.disconnect_all()
            print("已关闭所有 MCP 服务器连接")


# 全局 MCP 连接管理器
mcp_manager = MCPConnectionManager()


async def init_mcp() -> MCPConnectionManager:
    """初始化 MCP 连接"""
    await mcp_manager.initialize()
    return mcp_manager


async def close_mcp():
    """关闭 MCP 连接"""
    await mcp_manager.close()

