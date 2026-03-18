"""
直接测试 Mem0 功能 - 使用 v3 embedding with debug
"""
import asyncio
import os
import logging

# 设置代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

# 开启 debug 日志
logging.basicConfig(level=logging.DEBUG)

async def test_mem0():
    # 重新导入以获取新配置
    import importlib
    from src.agent.memory import mem0_manager as mm
    importlib.reload(mm)
    
    from src.agent.memory.mem0_manager import reset_mem0_manager, get_mem0_manager, Mem0MemoryManager
    from config.settings import get_settings
    
    # 重置管理器以重新初始化
    reset_mem0_manager()
    
    settings = get_settings()
    print(f"MEM0_ENABLED: {getattr(settings, 'mem0_enabled', False)}")
    print(f"DASHSCOPE_API_KEY: {settings.dashscope_api_key[:10]}...")
    print(f"DASHSCOPE_BASE_URL: {settings.dashscope_base_url}")
    
    # 创建新的 manager 实例并手动初始化查看配置
    manager = Mem0MemoryManager(provider="qwen")
    manager._initialized = False  # 强制重新初始化
    
    # 打印配置信息
    print(f"\nManager API Key: {manager._get_api_key()[:10]}...")
    print(f"Manager Base URL: {manager._get_base_url()}")
    
    # 先设置环境变量
    os.environ["OPENAI_API_KEY"] = manager._get_api_key()
    os.environ["OPENAI_BASE_URL"] = manager._get_base_url()
    
    # 手动初始化看配置
    print("\n=== 手动初始化 Mem0 客户端 ===")
    from mem0 import Memory
    
    config_dict = {
        "llm": {
            "provider": "openai",
            "config": {
                "model": manager.model,
            }
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v3",
                "openai_base_url": manager._get_base_url(),
                "api_key": manager._get_api_key(),
                "embedding_dims": 1024
            }
        },
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "mem0_memories_test",
                "path": str(settings.chroma_dir / "mem0_chroma_test")
            }
        }
    }
    
    print(f"Embedder config: {config_dict['embedder']}")
    
    try:
        client = Memory.from_config(config_dict=config_dict)
        print("Mem0 客户端初始化成功!")
        
        # 测试添加记忆 - 尝试不同的参数格式
        print("\n=== 测试添加记忆 ===")
        
        # 尝试直接 add 方式
        result = client.add(
            "测试记忆",
            user_id="debug_test_user"
        )
        print(f"添加结果 (简单方式): {result}")
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_mem0())
