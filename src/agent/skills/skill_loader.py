"""
Skill Loader - 技能加载器
动态解析 Skill.md 并创建 Agent 实例
"""
import os
import sys
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Type
from pathlib import Path
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import BaseTool
from src.models.llm import get_llm
from config.settings import get_settings
import frontmatter
from ..checkpointer import get_checkpointer


class Skill:
    """技能定义"""
    def __init__(self, name: str, description: str, prompt: str, tools: List[BaseTool], metadata: Dict[str, Any]):
        self.name = name
        self.description = description
        self.prompt = prompt
        self.tools = tools
        self.metadata = metadata


class SkillLoader:
    """技能加载器"""
    
    def __init__(self, skills_dir: str = None):
        if skills_dir is None:
            # 默认路径：agents/skills/（skill_loader.py 所在目录）
            self.skills_dir = Path(__file__).parent
        else:
            self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Skill] = {}
        self._agents: Dict[str, Any] = {}
        
    def load_skill(self, skill_name: str) -> Skill:
        """加载单个技能"""
        if skill_name in self._skills:
            return self._skills[skill_name]
        
        skill_dir = self.skills_dir / skill_name
        if not skill_dir.exists():
            raise ValueError(f"Skill 目录不存在: {skill_dir}")
        
        # 解析 Skill.md
        skill_md_path = skill_dir / "Skill.md"
        if not skill_md_path.exists():
            raise ValueError(f"Skill.md 不存在: {skill_md_path}")

        with open(skill_md_path, 'r', encoding='utf-8') as f:
            post = frontmatter.parse(f.read())

        # frontmatter.parse 返回 (metadata, content) 元组
        # 注意：顺序是 metadata 在前，content 在后
        if isinstance(post, tuple) and len(post) == 2:
            metadata_raw, content_raw = post
            # metadata 可能是字符串（YAML格式）或字典
            metadata_dict = metadata_raw if isinstance(metadata_raw, dict) else {}
            # content 应该是字符串
            prompt = content_raw if isinstance(content_raw, str) else ""
        else:
            prompt = getattr(post, 'content', '') or ''
            raw_metadata = getattr(post, 'metadata', {}) or {}
            metadata_dict = raw_metadata if isinstance(raw_metadata, dict) else {}

        name = metadata_dict.get('name', skill_name)
        description = metadata_dict.get('description', '')
        
        # 加载 examples.md（如果存在）
        examples_md_path = skill_dir / "examples.md"
        if examples_md_path.exists():
            with open(examples_md_path, 'r', encoding='utf-8') as f:
                examples = f.read()
            if examples.strip():
                prompt += f"\n\n## Few-shot Examples\n{examples}"
        
        # 加载工具
        tools = self._load_tools(skill_name, metadata_dict)
        
        skill = Skill(
            name=name,
            description=description,
            prompt=prompt,
            tools=tools,
            metadata=metadata_dict
        )
        
        self._skills[skill_name] = skill
        return skill
    
    def _load_tools(self, skill_name: str, metadata: Dict[str, Any]) -> List[BaseTool]:
        """动态加载工具"""
        tools = []
        
        # 从 YAML 配置加载工具
        tools_config = metadata.get('tools', [])
        
        for tool_config in tools_config:
            module_path = tool_config.get('module')  # 例如: scripts.tools
            tool_names = tool_config.get('names', [])  # 例如: [knowledge_search]
            
            if not module_path or not tool_names:
                continue
            
            # 构建完整的模块路径
            skill_dir = self.skills_dir / skill_name
            full_module_path = f"src.agent.skills.{skill_name}.{module_path}"
            
            # 将 "scripts.tools" 转换为 "scripts/tools.py"
            module_file_path = skill_dir / f"{module_path.replace('.', '/')}.py"
            
            try:
                # 动态导入模块
                spec = importlib.util.spec_from_file_location(
                    module_path,
                    module_file_path
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[full_module_path] = module
                    spec.loader.exec_module(module)
                    
                    # 获取工具函数并创建 BaseTool
                    for tool_name in tool_names:
                        if hasattr(module, tool_name):
                            tool_func = getattr(module, tool_name)
                            # 检查是否是创建工具的函数
                            if hasattr(tool_func, '__call__') and tool_name.startswith('create_'):
                                tool = tool_func()
                            else:
                                # 创建 StructuredTool
                                from langchain_core.tools import StructuredTool
                                tool = StructuredTool.from_function(tool_func)
                            tools.append(tool)
            except Exception as e:
                print(f"加载工具失败 {skill_name}.{module_path}.{tool_names}: {e}")
                import traceback
                traceback.print_exc()
        
        # 处理 MCP 服务器工具
        mcp_servers = metadata.get('mcp_servers', [])
        if mcp_servers:
            mcp_tools = self._load_mcp_tools(mcp_servers)
            tools.extend(mcp_tools)
        
        return tools
    
    def _load_mcp_tools(self, server_names: List[str]) -> List[BaseTool]:
        """加载 MCP 工具"""
        from tools.mcp_adapter import convert_mcp_tools
        from src.models.mcp_client import mcp_manager
        
        mcp_tools = []
        try:
            all_tools = mcp_manager.get_tools()
            # 过滤指定服务器的工具
            # 这里简化处理，返回所有 MCP 工具
            mcp_tools = convert_mcp_tools(all_tools)
        except Exception as e:
            print(f"加载 MCP 工具失败: {e}")
        
        return mcp_tools
    
    def create_agent(self, skill_name: str):
        """创建 Agent 实例"""
        if skill_name in self._agents:
            return self._agents[skill_name]
        
        skill = self.load_skill(skill_name)
        
        llm = get_llm()
        
        # 使用单例 checkpointer
        checkpointer = get_checkpointer()
        
        agent = create_react_agent(
            llm,
            skill.tools,
            prompt=skill.prompt,
            checkpointer=checkpointer
        )
        
        self._agents[skill_name] = agent
        return agent
    
    def get_skill(self, skill_name: str) -> Skill:
        """获取技能定义"""
        return self.load_skill(skill_name)
    
    def list_skills(self) -> List[str]:
        """列出所有技能"""
        if not self.skills_dir.exists():
            return []
        
        skills = []
        for item in self.skills_dir.iterdir():
            if item.is_dir() and (item / "Skill.md").exists():
                skills.append(item.name)
        return skills
    
    def run(self, skill_name: str, query: str, session_id: str = "default") -> str:
        """运行技能"""
        agent = self.create_agent(skill_name)
        config = {"configurable": {"thread_id": f"{skill_name}_{session_id}"}}
        
        result = agent.invoke(
            {"messages": [query]},
            config
        )
        
        return result["messages"][-1].content


# 全局单例
_global_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """获取全局技能加载器"""
    global _global_loader
    if _global_loader is None:
        _global_loader = SkillLoader()
    return _global_loader


def create_agent(skill_name: str):
    """创建 Agent 的便捷函数"""
    loader = get_skill_loader()
    return loader.create_agent(skill_name)


def run_skill(skill_name: str, query: str, session_id: str = "default") -> str:
    """运行技能的便捷函数"""
    loader = get_skill_loader()
    return loader.run(skill_name, query, session_id)


__all__ = [
    "Skill",
    "SkillLoader", 
    "get_skill_loader",
    "create_agent",
    "run_skill"
]

