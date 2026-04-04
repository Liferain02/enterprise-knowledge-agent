#!/usr/bin/env python
"""
测试流式输出
"""
import sys
import os
import asyncio
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from src.agent.graph import get_agent_graph
from langchain_core.messages import HumanMessage

async def test_stream():
    graph = get_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="你好")],
        "session_id": "test"
    }
    run_config = {"configurable": {"thread_id": "test"}}
    
    print("开始流式输出测试...")
    async for chunk in graph.astream(initial_state, run_config):
        print(f"Chunk: {chunk}")

if __name__ == "__main__":
    asyncio.run(test_stream())
