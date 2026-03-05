"""
MCP 客户端连接管理模块
负责连接外部 MCP 服务器并管理工具
"""
import asyncio
import sys
from typing import List, Optional, Dict, Any

from config.settings import get_settings


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
            self.tool_manager: Optional[Any] = None
            self.mcp_clients: Dict[str, Any] = {}
            self._initialized_connect = False
            MCPConnectionManager._initialized = True
    
    async def initialize(self, timeout: float = 30.0):
        """
        初始化 MCP 连接（支持并发和超时保护）

        Args:
            timeout: 单个 MCP 服务器连接超时时间（秒）
        """
        if self._initialized_connect:
            return

        self._initialized_connect = True
        settings = get_settings()

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

        # 并发连接所有 MCP 服务器
        await self._connect_servers_concurrent(mcp_configs, timeout)

    async def _connect_servers_concurrent(
        self,
        mcp_configs: List[Dict],
        timeout: float = 30.0
    ):
        """
        并发连接多个 MCP 服务器

        Args:
            mcp_configs: MCP 服务器配置列表
            timeout: 超时时间
        """
        async def connect_with_timeout(config: Dict) -> tuple:
            """带超时的连接"""
            server_config = MCPServerConfig(**config)
            try:
                success = await asyncio.wait_for(
                    self.tool_manager.connect_mcp_server(server_config),
                    timeout=timeout
                )
                return (server_config.name, success, None)
            except asyncio.TimeoutError:
                return (server_config.name, False, f"连接超时 ({timeout}s)")
            except Exception as e:
                return (server_config.name, False, str(e))

        # 并发执行所有连接
        print(f"正在并发连接 {len(mcp_configs)} 个 MCP 服务器...")
        results = await asyncio.gather(
            *[connect_with_timeout(config) for config in mcp_configs],
            return_exceptions=False
        )

        # 汇总结果
        success_count = 0
        for name, success, error in results:
            if success:
                print(f"  ✓ MCP 服务器 '{name}' 连接成功")
                success_count += 1
            else:
                msg = error or "连接失败"
                print(f"  ✗ MCP 服务器 '{name}' {msg}")

        print(f"MCP 连接完成: {success_count}/{len(mcp_configs)} 成功")
    
    def get_tools(self) -> List:
        """获取所有 MCP 工具"""
        if self.tool_manager is None:
            print("[DEBUG] tool_manager 未初始化")
            return []
        tools = self.tool_manager.get_all_tools()
        print(f"[DEBUG] MCP Manager 返回 {len(tools)} 个工具")
        return tools
    
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


# ==================== MCP 配置模型 ====================

class MCPServerConfig:
    """MCP 服务器配置"""
    def __init__(self, name: str, command: str, args: List[str] = None, env: Dict[str, str] = None):
        self.name = name
        self.command = command
        self.args = args or []
        self.env = env or {}


# ==================== MCP 工具管理器 ====================

class MCPToolManager:
    """MCP 工具管理器 - 支持本地工具和外部 MCP 服务器"""
    
    def __init__(self):
        self._local_tools = []
        self._mcp_servers: Dict[str, Any] = {}
        self._mcp_tools: List[Any] = []
    
    async def connect_mcp_server(self, config: MCPServerConfig) -> bool:
        """连接外部 MCP 服务器"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from contextlib import AsyncExitStack
            
            # 构建服务器参数
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env or None
            )
            
            # 创建客户端
            client = MCPClient(config)
            success = await client.connect()
            
            if success:
                self._mcp_servers[config.name] = client
                tools = client.get_tools()
                self._mcp_tools.extend(tools)
                print(f"成功连接到 MCP 服务器: {config.name}, 获取 {len(tools)} 个工具")
            
            return success
            
        except Exception as e:
            print(f"连接 MCP 服务器失败: {e}")
            return False
    
    async def disconnect_all(self):
        """断开所有 MCP 服务器连接"""
        for client in self._mcp_servers.values():
            await client.disconnect()
        self._mcp_servers.clear()
        self._mcp_tools.clear()
    
    def get_all_tools(self) -> List:
        """获取所有 MCP 工具"""
        return self._mcp_tools
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 工具"""
        if server_name not in self._mcp_servers:
            raise ValueError(f"未找到 MCP 服务器: {server_name}")
        
        client = self._mcp_servers[server_name]
        return await client.call_tool(tool_name, arguments)


# ==================== MCP 客户端 ====================

class MCPClient:
    """MCP 客户端 - 连接外部 MCP 服务器"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._exit_stack = None
        self._session = None
        self._tools = []
        self._stdio_client = None
        # 旧版 API 需要保存上下文管理器
        self._stdio_ctx = None
        self._session_ctx = None
    
    async def connect(self) -> bool:
        """连接 MCP 服务器"""
        try:
            # 尝试导入 MCP 1.x 新版本 API
            try:
                from mcp import ClientSession
                from mcp.client.stdio import StdioClient
            except ImportError:
                # 回退到旧版本 API
                from mcp import ClientSession, StdioServerParameters
                from mcp.client.stdio import stdio_client

            # 构建服务器参数
            import inspect
            if 'StdioServerParameters' in dir():
                server_params = StdioServerParameters(
                    command=self.config.command,
                    args=self.config.args,
                    env=self.config.env or None
                )
            else:
                # MCP 1.6+ 新 API
                server_params = {
                    "command": self.config.command,
                    "args": self.config.args,
                    "env": self.config.env or {}
                }

            # 根据 MCP 版本选择连接方式
            if 'StdioClient' in dir():
                # MCP 1.6+ 新 API - 手动管理上下文
                stdio_client_instance = StdioClient(**server_params)
                try:
                    read, write = await stdio_client_instance.__aenter__()
                    session = ClientSession(read, write)
                    await session.initialize()
                    tools_response = await session.list_tools()
                    self._tools = tools_response.tools
                    self._session = session
                    self._stdio_client = stdio_client_instance
                    return True
                except Exception as e:
                    try:
                        await stdio_client_instance.__aexit__(type(e), e, e.__traceback__)
                    except Exception:
                        pass
                    raise
            else:
                # 旧版 API - 使用上下文管理器
                self._stdio_ctx = stdio_client(server_params)
                read, write = await self._stdio_ctx.__aenter__()
                
                self._session_ctx = ClientSession(read, write)
                self._session = await self._session_ctx.__aenter__()
                
                await self._session.initialize()
                
                tools_response = await self._session.list_tools()
                self._tools = tools_response.tools
            
            return True
            
        except NotImplementedError as e:
            import traceback
            print(f"连接错误: NotImplementedError - {e}")
            print(f"完整堆栈:\n{traceback.format_exc()}")
            return False
        except Exception as e:
            import traceback
            print(f"连接错误: {type(e).__name__} - {e}")
            print(f"完整堆栈:\n{traceback.format_exc()}")
            return False
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 工具"""
        if not self._session:
            raise RuntimeError("未连接到 MCP 服务器")
        
        result = await self._session.call_tool(tool_name, arguments)
        return result
    
    async def disconnect(self):
        """断开连接"""
        try:
            if self._session:
                try:
                    await self._session.__aexit__(None, None, None)
                except (asyncio.CancelledError, RuntimeError):
                    pass
                self._session = None

            # 正确关闭 stdio_client
            if hasattr(self, '_stdio_client') and self._stdio_client is not None:
                try:
                    await self._stdio_client.__aexit__(None, None, None)
                except (asyncio.CancelledError, RuntimeError):
                    pass
                self._stdio_client = None
        except Exception as e:
            print(f"断开连接时出错: {e}")
    
    def get_tools(self) -> List:
        """获取工具列表"""
        return self._tools

