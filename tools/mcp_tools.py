"""
MCP工具实现模块
支持连接外部 MCP 服务器，同时也保留自定义工具
"""
import asyncio
from typing import Any, Dict, List, Optional, Callable
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from config.settings import get_settings


# ==================== MCP 服务器配置 ====================
class MCPServerConfig(BaseModel):
    """MCP 服务器配置"""
    name: str = Field(..., description="服务器名称")
    command: str = Field(..., description="启动命令，如 npx, python 等")
    args: List[str] = Field(default_factory=list, description="命令参数")
    env: Optional[Dict[str, str]] = Field(default_factory=dict, description="环境变量")


# ==================== MCP 客户端连接 ====================
class MCPClient:
    """MCP 客户端 - 连接外部 MCP 服务器"""
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process = None
        self._tools = []
    
    async def connect(self):
        """连接 MCP 服务器"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            
            # 构建服务器参数
            server_params = StdioServerParameters(
                command=self.config.command,
                args=self.config.args,
                env=self.config.env or None
            )
            
            print(f"  启动命令: {self.config.command} {' '.join(self.config.args)}")
            
            # 使用上下文管理器
            async with stdio_client(server_params) as (read, write):
                self._session = ClientSession(read, write)
                await self._session.initialize()
                
                # 获取可用工具列表
                tools_response = await self._session.list_tools()
                self._tools = tools_response.tools
                
                print(f"  获取到 {len(self._tools)} 个工具")
                # 保持连接活跃
                return True
                
        except Exception as e:
            print(f"  连接错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 工具"""
        if not hasattr(self, '_session'):
            raise RuntimeError("未连接到 MCP 服务器")
        
        result = await self._session.call_tool(tool_name, arguments)
        return result
    
    async def disconnect(self):
        """断开连接"""
        try:
            if hasattr(self, '_session') and self._session:
                await self._session.close()
            # 清理stdio_client资源
            if hasattr(self, '_write') and self._write:
                await self._write.aclose()
            if hasattr(self, '_read') and self._read:
                await self._read.aclose()
        except Exception as e:
            print(f"  关闭会话时出错: {e}")
    
    def get_tools(self) -> List:
        """获取工具列表"""
        return self._tools


# ==================== MCP 工具管理器 ====================
class MCPToolManager:
    """MCP 工具管理器 - 支持本地工具和外部 MCP 服务器"""
    
    def __init__(self):
        self.settings = get_settings()
        self._local_tools: List[BaseTool] = []
        self._mcp_servers: Dict[str, MCPClient] = {}
        self._mcp_tools: List[Any] = []
    
    def add_local_tool(self, tool: BaseTool):
        """添加本地工具"""
        self._local_tools.append(tool)
    
    def add_local_tools(self, tools: List[BaseTool]):
        """批量添加本地工具"""
        self._local_tools.extend(tools)
    
    async def connect_mcp_server(self, config: MCPServerConfig) -> bool:
        """连接外部 MCP 服务器"""
        client = MCPClient(config)
        success = await client.connect()
        
        if success:
            self._mcp_servers[config.name] = client
            # 获取服务器提供的工具
            tools = client.get_tools()
            self._mcp_tools.extend(tools)
            print(f"成功连接到 MCP 服务器: {config.name}, 获取 {len(tools)} 个工具")
        
        return success
    
    async def connect_mcp_servers(self, configs: List[MCPServerConfig]):
        """批量连接多个 MCP 服务器"""
        for config in configs:
            await self.connect_mcp_server(config)
    
    async def call_mcp_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """调用 MCP 工具"""
        if server_name not in self._mcp_servers:
            raise ValueError(f"未找到 MCP 服务器: {server_name}")
        
        client = self._mcp_servers[server_name]
        return await client.call_tool(tool_name, arguments)
    
    async def disconnect_all(self):
        """断开所有 MCP 服务器连接"""
        for client in self._mcp_servers.values():
            await client.disconnect()
        self._mcp_servers.clear()
        self._mcp_tools.clear()
    
    def get_local_tools(self) -> List[BaseTool]:
        """获取本地工具"""
        return self._local_tools
    
    def get_mcp_tools(self) -> List[Any]:
        """获取 MCP 工具"""
        return self._mcp_tools
    
    def get_all_tools(self) -> List:
        """获取所有工具（本地 + MCP）"""
        return self._local_tools + self._mcp_tools
    
    def get_tools_description(self) -> str:
        """获取工具描述"""
        descriptions = []
        
        # 本地工具
        for tool in self._local_tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        # MCP 工具
        for tool in self._mcp_tools:
            input_schema = tool.inputSchema if hasattr(tool, 'inputSchema') else {}
            descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(descriptions)


# ==================== 便捷函数 ====================
def create_mcp_tool_manager() -> MCPToolManager:
    """创建 MCP 工具管理器"""
    return MCPToolManager()


async def create_mcp_client(config: dict) -> Optional[MCPClient]:
    """创建 MCP 客户端"""
    try:
        server_config = MCPServerConfig(**config)
        client = MCPClient(server_config)
        
        if await client.connect():
            return client
        return None
    except Exception as e:
        print(f"创建 MCP 客户端失败: {e}")
        return None


# ==================== 常用 MCP 服务器配置示例 ====================
# 以下是一些常用的外部 MCP 服务器配置示例
# 你可以在 config/settings.py 或 .env 中配置

EXAMPLE_MCP_SERVERS = [
    # 飞书 MCP (需要安装飞书 MCP 服务器)
    # {
    #     "name": "feishu",
    #     "command": "npx",
    #     "args": ["-y", "@feishu/mcp-server"],
    #     "env": {"FEISHU_APP_ID": "your-app-id", "FEISHU_APP_SECRET": "your-app-secret"}
    # },
    
    # 高德地图 MCP (需要安装高德 MCP 服务器)
    # {
    #     "name": "amap",
    #     "command": "npx",
    #     "args": ["-y", "@amap/mcp-server"],
    #     "env": {"AMAP_KEY": "your-amap-key"}
    # },
    
    # 文件系统 MCP
    # {
    #     "name": "filesystem",
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/directory"],
    # },
    
    # GitHub MCP
    # {
    #     "name": "github",
    #     "command": "npx",
    #     "args": ["-y", "@modelcontextprotocol/server-github"],
    #     "env": {"GITHUB_TOKEN": "your-github-token"}
    # },
]
