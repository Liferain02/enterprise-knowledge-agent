"""
Mem0 记忆功能测试

测试 Mem0 语义记忆的添加、检索、跨会话等功能
"""
import asyncio
import os

os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"


async def test_mem0_basic():
    """测试基本添加和检索"""
    from src.agent.memory.mem0_manager import reset_mem0_manager, get_mem0_manager
    
    reset_mem0_manager()
    mem0_manager = get_mem0_manager(provider="qwen")
    
    # 添加记忆
    messages = [
        {"role": "user", "content": "我的名字叫李四，我最喜欢的颜色是蓝色"},
        {"role": "assistant", "content": "好的，我记住你了！"}
    ]
    result = await mem0_manager.add_conversation(
        messages=messages,
        user_id="test_user",
        session_id="test_session"
    )
    print(f"添加记忆: {result}")
    
    await asyncio.sleep(1)
    
    # 检索记忆
    results = await mem0_manager.search(
        query="用户最喜欢什么",
        user_id="test_user"
    )
    print(f"检索结果: {results}")
    
    return results


async def test_mem0_cross_session():
    """测试跨会话记忆"""
    from src.agent.memory.mem0_manager import get_mem0_manager
    
    mem0_manager = get_mem0_manager(provider="qwen")
    
    # 会话1: 告诉AI名字
    await mem0_manager.add_conversation(
        messages=[
            {"role": "user", "content": "我叫张三，是一名软件工程师"},
            {"role": "assistant", "content": "你好张三，很高兴认识你！"}
        ],
        user_id="test_user",
        session_id="session_1"
    )
    
    # 会话2: 问AI是否记得名字
    results = await mem0_manager.search(
        query="用户叫什么",
        user_id="test_user"
    )
    print(f"跨会话检索: {results}")
    
    return results


async def test_mem0_format():
    """测试记忆格式化"""
    from src.agent.memory.mem0_manager import get_mem0_manager
    
    mem0_manager = get_mem0_manager(provider="qwen")
    
    memories = [
        {"content": "用户叫张三，是软件工程师"},
        {"content": "用户喜欢蓝色"}
    ]
    
    formatted = mem0_manager.format_memories_for_context(memories, max_chars=200)
    print(f"格式化结果:\n{formatted}")


if __name__ == "__main__":
    print("=== 测试基本功能 ===")
    asyncio.run(test_mem0_basic())
    
    print("\n=== 测试跨会话 ===")
    asyncio.run(test_mem0_cross_session())
    
    print("\n=== 测试格式化 ===")
    asyncio.run(test_mem0_format())
