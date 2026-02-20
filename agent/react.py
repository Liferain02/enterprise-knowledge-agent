"""
ReAct 实现模块
实现 Reason + Act 推理框架
"""
import re
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from core.llm import get_llm
from core.chat_history import get_chat_history_manager
from tools.base import get_all_tools, get_tool_by_name
from config.prompts import REACT_SYSTEM_PROMPT, REACT_USER_PROMPT
from config.settings import get_settings
class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent
    
    工作流程：
    1. Thought - 分析当前问题，决定下一步行动
    2. Action - 执行一个工具来获取信息
    3. Observation - 观察工具返回的结果
    4. 重复直到得到最终答案
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        tools: Optional[List] = None
    ):
        self.settings = get_settings()
        self.max_iterations = max_iterations or self.settings.max_iterations
        self.llm = get_llm()
        self.tools = tools or get_all_tools()
        self.tools_map = {tool.name: tool for tool in self.tools}
        
        # 构建工具描述
        self.tools_description = self._build_tools_description()
        
        # ReAct 提示词
        self.prompt = self._build_prompt()
    
    def _build_tools_description(self) -> str:
        """构建工具描述"""
        descriptions = []
        for tool in self.tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """构建 ReAct 提示词"""
        return ChatPromptTemplate.from_messages([
            ("system", REACT_SYSTEM_PROMPT.format(tools=self.tools_description)),
            ("human", REACT_USER_PROMPT)
        ])
    
    def parse_response(self, response: str) -> Tuple[str, str, str]:
        """
        解析 LLM 响应
        
        Returns:
            (thought, action, observation)
        """
        thought = ""
        action = ""
        observation = ""
        
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Thought:"):
                thought = line[8:].strip()
                current_section = "thought"
            elif line.startswith("Action:"):
                action = line[7:].strip()
                current_section = "action"
            elif line.startswith("Observation:"):
                observation = line[12:].strip()
                current_section = "observation"
            elif line.startswith("Final Answer:"):
                # 最终答案格式
                return "final", line[13:].strip(), ""
            elif current_section == "thought":
                thought += " " + line
            elif current_section == "action":
                action += " " + line
            elif current_section == "observation":
                observation += " " + line
        
        return thought, action, observation
    
    def parse_action(self, action: str) -> Tuple[str, Dict[str, Any]]:
        """
        解析动作字符串
        
        格式: tool_name|param1=value1,param2=value2
        
        Returns:
            (tool_name, params)
        """
        if "|" not in action:
            # 尝试简单解析
            return action.strip(), {}
        
        tool_part, params_part = action.split("|", 1)
        tool_name = tool_part.strip()
        
        params = {}
        if params_part:
            # 解析参数
            param_pairs = params_part.split(",")
            for pair in param_pairs:
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    params[key.strip()] = value.strip()
        
        return tool_name, params
    
    def run(
        self,
        question: str,
        context: str = "",
        history: str = ""
    ) -> Dict[str, Any]:
        """
        运行 ReAct Agent
        
        Args:
            question: 用户问题
            context: 上下文信息（RAG结果）
            history: 对话历史
        
        Returns:
            执行结果
        """
        # 记录过程
        thought_history = []
        action_history = []
        observation_history = []
        
        # 构建初始提示
        prompt_value = self.prompt.format(
            question=question,
            context=context,
            history=history
        )
        
        iteration = 0
        final_answer = None
        
        while iteration < self.max_iterations and final_answer is None:
            iteration += 1
            
            # 调用 LLM
            response = self.llm.invoke(prompt_value)
            response_text = response.content
            
            # 解析响应
            thought, action, observation = self.parse_response(response_text)
            
            # 检查是否是最终答案
            if thought == "final":
                final_answer = action
                break
            
            # 记录
            thought_history.append(thought)
            action_history.append(action)
            
            # 执行动作
            if action:
                tool_name, params = self.parse_action(action)
                
                tool = self.tools_map.get(tool_name)
                
                if tool:
                    try:
                        # 根据工具类型设置参数
                        if tool_name == "search_knowledge":
                            observation = tool.invoke(question)
                        elif tool_name == "calculate":
                            observation = tool.invoke({"expression": params.get("expression", question)})
                        elif tool_name == "get_date":
                            observation = tool.invoke({})
                        else:
                            observation = tool.invoke(question)
                    except Exception as e:
                        observation = f"执行错误: {str(e)}"
                else:
                    observation = f"工具不存在: {tool_name}"
            else:
                observation = "无动作执行"
            
            observation_history.append(observation)
            
            # 构建下一轮提示
            prompt_value += f"\n\nThought: {thought}\nAction: {action}\nObservation: {observation}"
        
        if final_answer is None:
            final_answer = "抱歉，我无法在指定的迭代次数内找到答案。"
        
        return {
            "answer": final_answer,
            "iterations": iteration,
            "thought_history": thought_history,
            "action_history": action_history,
            "observation_history": observation_history
        }
    
    async def ainvoke(
        self,
        question: str,
        context: str = "",
        history: str = ""
    ) -> Dict[str, Any]:
        """异步运行 ReAct Agent"""
        # 简化版本，实际可以改写为异步
        return self.run(question, context, history)
# 便捷函数
def create_react_agent(
    max_iterations: int = 10,
    tools: Optional[List] = None
) -> ReActAgent:
    """创建 ReAct Agent"""
    return ReActAgent(max_iterations=max_iterations, tools=tools)
def run_react_agent(
    question: str,
    context: str = "",
    history: str = ""
) -> Dict[str, Any]:
    """运行 ReAct Agent 的便捷函数"""
    agent = create_react_agent()
    return agent.run(question, context, history)
