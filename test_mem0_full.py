"""
完整测试 Mem0 功能 - 添加和检索
"""
import asyncio
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

async def test_mem0():
    from src.agent.memory.mem0_manager import reset_mem0_manager, get_mem0_manager
    from config.settings import get_settings
    
    # 重置管理器
    reset_mem0_manager()
    settings = get_settings()
    
    # 获取 mem0 manager
    mem0_manager = get_mem0_manager(provider="qwen")
    
    # 1. 测试添加新记忆
    print("=== 测试添加新记忆 ===")
    messages = [
        {"role": "user", "content": "我今天心情非常好，因为项目上线成功了！"},
        {"role": "assistant", "content": "恭喜项目上线！太棒了！"}
    ]
    result = await mem0_manager.add_conversation(
        messages=messages,
        user_id="test_user",
        session_id="test_session"
    )
    print(f"添加结果: {result}")
    
    # 2. 测试检索记忆
    print("\n=== 测试检索记忆 ===")
    search_result = await mem0_manager.search(
        query="项目上线",
        user_id="test_user"
    )
    print(f"检索结果: {search_result}")
    
    # 3. 测试获取历史
    print("\n=== 测试获取历史 ===")
    history = await mem0_manager.get_all_memories(user_id="test_user")
    print(f"历史记忆: {history}")

if __name__ == "__main__":
    asyncio.run(test_mem0())
