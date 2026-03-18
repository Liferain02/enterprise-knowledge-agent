"""
直接测试 Mem0 功能
"""
import asyncio
import os

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

async def test_mem0():
    from src.agent.memory.mem0_manager import get_mem0_manager
    from config.settings import get_settings
    
    settings = get_settings()
    print(f"MEM0_ENABLED: {getattr(settings, 'mem0_enabled', False)}")
    
    # 获取 mem0 manager
    mem0_manager = get_mem0_manager(provider="qwen")
    
    # 测试添加记忆
    print("\n=== 测试添加记忆 ===")
    messages = [
        {"role": "user", "content": "我的名字叫李四，我最喜欢的颜色是蓝色"},
        {"role": "assistant", "content": "好的，我记住你了！"}
    ]
    result = await mem0_manager.add_conversation(
        messages=messages,
        user_id="test_user_001",
        session_id="test_session_001"
    )
    print(f"添加记忆结果: {result}")
    
    # 等待一下让记忆保存
    await asyncio.sleep(2)
    
    # 测试检索记忆
    print("\n=== 测试检索记忆 ===")
    search_results = await mem0_manager.search(
        query="用户最喜欢什么",
        user_id="test_user_001",
        limit=3
    )
    print(f"检索结果: {search_results}")
    
    # 格式化记忆
    if search_results:
        formatted = mem0_manager.format_memories_for_context(search_results)
        print(f"\n格式化后的记忆:\n{formatted}")
    
    # 获取所有记忆
    print("\n=== 获取所有记忆 ===")
    all_memories = await mem0_manager.get_all_memories(user_id="test_user_001")
    print(f"所有记忆: {all_memories}")

if __name__ == "__main__":
    asyncio.run(test_mem0())
