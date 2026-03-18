"""
直接测试 Mem0 功能 - 使用 v3 embedding
"""
import asyncio
import os

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

async def test_mem0():
    # 重新导入以获取新配置
    import importlib
    from src.agent.memory import mem0_manager
    importlib.reload(mem0_manager)
    
    from src.agent.memory.mem0_manager import reset_mem0_manager, get_mem0_manager
    from config.settings import get_settings
    
    # 重置管理器以重新初始化
    reset_mem0_manager()
    
    settings = get_settings()
    print(f"MEM0_ENABLED: {getattr(settings, 'mem0_enabled', False)}")
    
    # 获取 mem0 manager
    mem0_manager = get_mem0_manager(provider="qwen")
    
    # 测试添加新记忆
    print("\n=== 测试添加新记忆 (v3) ===")
    messages = [
        {"role": "user", "content": "我今天的心情很开心"},
        {"role": "assistant", "content": "太好了！继续保持好心情！"}
    ]
    result = await mem0_manager.add_conversation(
        messages=messages,
        user_id="test_user_v3",
        session_id="test_session_v3"
    )
    print(f"添加记忆结果: {result}")

if __name__ == "__main__":
    asyncio.run(test_mem0())
