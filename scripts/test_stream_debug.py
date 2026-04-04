#!/usr/bin/env python
"""
调试流式输出
"""
import sys
import os
import asyncio
import json
sys.path.insert(0, '/share/home/lifr/workspace/code/enterprise-knowledge-agent')
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7897'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7897'

from src.agent.graph import get_agent_graph
from langchain_core.messages import HumanMessage

async def test_stream_debug():
    graph = get_agent_graph()
    
    initial_state = {
        "messages": [HumanMessage(content="智源科技是什么时候成立的？")],
        "session_id": "test_debug"
    }
    run_config = {"configurable": {"thread_id": "test_debug"}}
    
    print("=== 开始流式输出测试 ===")
    
    full_answer = ""
    async for chunk in graph.astream(initial_state, run_config):
        print(f"--- Chunk keys: {chunk.keys()} ---")
        
        if "supervisor" in chunk:
            print(f"Supervisor: {chunk['supervisor']}")
        
        if "knowledge_agent" in chunk:
            data = chunk["knowledge_agent"]
            print(f"Knowledge Agent keys: {data.keys()}")
            if "answer" in data:
                answer = data["answer"]
                print(f"Answer chunk: {answer[:50]}...")
                full_answer += answer
            if "final_answer" in data:
                final = data["final_answer"]
                print(f"Final answer: {final[:50]}...")
                full_answer = final
                
        if "general_agent" in chunk:
            data = chunk["general_agent"]
            print(f"General Agent keys: {data.keys()}")
            if "final_answer" in data:
                full_answer = data["final_answer"]
    
    print(f"\n=== 完整答案 ({len(full_answer)} 字符) ===")
    print(full_answer[:200])

if __name__ == "__main__":
    asyncio.run(test_stream_debug())
